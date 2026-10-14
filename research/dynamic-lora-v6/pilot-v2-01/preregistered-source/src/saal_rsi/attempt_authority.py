"""A custodian mirror and a separately administered monotonic witness.

The witness is the source of attempt numbers, not a controller-created database.
Restoring either custodian mirror or creating another mirror cannot reset alpha.
Protection assumes the witness state, key, enrollment and backups are outside
the controller/custodian administrator's control. Local tests do not establish
that operational separation. Missing or uncertain state always blocks progress.
"""
import json
import math
import re
import sqlite3
import time
from pathlib import Path

from .model_contract import sign, verify
from .spec import canonical, parse_unique_json, sha
from .storage import private_database

ZERO = 'sha256:' + '0' * 64
BINDINGS = {'candidate_digest', 'baseline_digest', 'protocol_digest',
            'holdout_digest', 'sampling_review_digest', 'question_digest'}


def digest(value):
    if type(value) is not str or not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
        raise ValueError('bounded digest required')
    return value


def validate_bindings(bindings):
    if type(bindings) is not dict or set(bindings) != BINDINGS:
        raise ValueError('complete frozen attempt identity required')
    for value in bindings.values():
        digest(value)
    return bindings


def validate_enrollment(enrollment):
    if type(enrollment) is not dict or set(enrollment) != {
        'tenant', 'family', 'question_digest', 'total_alpha', 'max_attempts'}:
        raise ValueError('witness enrollment shape')
    for field in ('tenant', 'family'):
        if type(enrollment[field]) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', enrollment[field]):
            raise ValueError('fixed witness namespace')
    digest(enrollment['question_digest'])
    if type(enrollment['total_alpha']) is not float or not math.isfinite(enrollment['total_alpha']) or not 0 < enrollment['total_alpha'] <= .05:
        raise ValueError('total alpha')
    if type(enrollment['max_attempts']) is not int or not 1 <= enrollment['max_attempts'] <= 30:
        raise ValueError('attempt budget')
    return enrollment


class Witness:
    def __init__(self, database, key, enrollment):
        self.path = Path(database)
        self.key = key
        self.pin = key.public_key().public_bytes_raw().hex()
        self.enrollment = validate_enrollment(enrollment)
        if not self.path.is_file():
            raise PermissionError('witness missing; never recreate it on a request')
        private_database(self.path)

    @classmethod
    def initialize(cls, database, key, enrollment):
        validate_enrollment(enrollment)
        path = Path(database)
        if path.exists() or path.is_symlink():
            raise FileExistsError('explicit witness enrollment is one-time')
        private_database(path)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE enrollment (body TEXT NOT NULL, pin TEXT NOT NULL)')
            db.execute('INSERT INTO enrollment VALUES (?,?)', (canonical(enrollment).decode(), key.public_key().public_bytes_raw().hex()))
            db.execute('CREATE TABLE events (seq INTEGER PRIMARY KEY, body TEXT NOT NULL)')
        return cls(path, key, enrollment)

    def _read(self, db):
        rows = db.execute('SELECT body,pin FROM enrollment').fetchall()
        if rows != [(canonical(self.enrollment).decode(), self.pin)]:
            raise PermissionError('witness enrollment/key substitution')
        previous = ZERO
        attempts = {}
        events = []
        for seq, raw in db.execute('SELECT seq,body FROM events ORDER BY seq'):
            envelope = parse_unique_json(raw.encode())
            event = verify(envelope, self.pin)
            if set(event) != {'schema', 'enrollment_digest', 'seq', 'previous', 'kind', 'attempt', 'bindings', 'alpha', 'time'} or event['schema'] != 'saal-attempt-event-v1':
                raise PermissionError('witness journal schema')
            if seq != len(events) + 1 or event['seq'] != seq or event['previous'] != previous or event['enrollment_digest'] != sha(self.enrollment):
                raise PermissionError('witness journal gap or changed enrollment')
            validate_bindings(event['bindings'])
            if event['kind'] == 'RESERVED':
                n = len(attempts) + 1
                if event['attempt'] != n or event['alpha'] != self.enrollment['total_alpha'] / 2 ** n:
                    raise PermissionError('counter or alpha regression')
                attempts[n] = {'reservation': envelope, 'consumption': None}
            elif event['kind'] == 'CONSUMED':
                a = attempts.get(event['attempt'])
                if not a or a['consumption'] is not None or any(event[k] != a['reservation']['payload'][k] for k in ('bindings', 'alpha')):
                    raise PermissionError('invalid witness consumption')
                a['consumption'] = envelope
            else:
                raise PermissionError('unknown attempt event')
            previous = sha(envelope)
            events.append(envelope)
        return events, attempts, previous

    def _head(self, events, attempts, previous):
        return sign({'schema': 'saal-attempt-head-v1', 'enrollment': self.enrollment,
                     'seq': len(events), 'attempt_count': len(attempts), 'head': previous}, self.key)

    def head(self):
        with sqlite3.connect(self.path.resolve().as_uri() + '?mode=rw', uri=True) as db:
            return self._head(*self._read(db))

    def transact(self, role, operation, arguments):
        if operation == 'head' and role in ('custodian', 'evaluator') and arguments == {}:
            return self.head()
        if operation not in ('reserve', 'consume') or role != {'reserve': 'custodian', 'consume': 'evaluator'}[operation]:
            raise PermissionError('role cannot perform this attempt operation')
        fields = {'expected_head', 'bindings'} if operation == 'reserve' else {'expected_head', 'reservation'}
        if type(arguments) is not dict or set(arguments) != fields:
            raise ValueError('attempt request shape')
        digest(arguments['expected_head'])
        with sqlite3.connect(self.path.resolve().as_uri() + '?mode=rw', uri=True, timeout=20) as db:
            db.execute('BEGIN IMMEDIATE')
            events, attempts, previous = self._read(db)
            if arguments['expected_head'] != previous:
                raise PermissionError('witness compare-and-swap conflict or rollback')
            if operation == 'reserve':
                bindings = validate_bindings(arguments['bindings'])
                if bindings['question_digest'] != self.enrollment['question_digest']:
                    raise PermissionError('new question requires witness administrator enrollment')
                if len(attempts) >= self.enrollment['max_attempts']:
                    raise PermissionError('global evaluation attempts exhausted')
                if any(a['reservation']['payload']['bindings'] == bindings for a in attempts.values()):
                    raise PermissionError('same frozen attempt already reserved; no implicit retry')
                number = len(attempts) + 1
                alpha = self.enrollment['total_alpha'] / 2 ** number
                kind = 'RESERVED'
            else:
                reservation = verify(arguments['reservation'], self.pin)
                number = reservation.get('attempt')
                if type(number) is not int or number not in attempts or attempts[number]['reservation'] != arguments['reservation'] or attempts[number]['consumption'] is not None:
                    raise PermissionError('unknown, substituted or consumed reservation')
                bindings = reservation['bindings']
                alpha = reservation['alpha']
                kind = 'CONSUMED'
            event = sign({'schema': 'saal-attempt-event-v1', 'enrollment_digest': sha(self.enrollment),
                          'seq': len(events) + 1, 'previous': previous, 'kind': kind,
                          'attempt': number, 'bindings': bindings, 'alpha': alpha,
                          'time': int(time.time())}, self.key)
            db.execute('INSERT INTO events VALUES (?,?)', (len(events) + 1, canonical(event).decode()))
            # Reservation spends alpha even if inference never begins or later fails.
            events, attempts, previous = self._read(db)
            return {'event': event, 'head': self._head(events, attempts, previous)}


def checked_head(envelope, pin, enrollment):
    body = verify(envelope, pin)
    if type(body) is not dict or set(body) != {'schema', 'enrollment', 'seq', 'attempt_count', 'head'} or body['schema'] != 'saal-attempt-head-v1' or body['enrollment'] != enrollment:
        raise PermissionError('wrong witness tenant, family or head schema')
    if any(type(body[k]) is not int or body[k] < 0 for k in ('seq', 'attempt_count')) or not body['attempt_count'] <= body['seq'] <= 2 * body['attempt_count']:
        raise ValueError('invalid witness counters')
    digest(body['head'])
    if (body['seq'] == 0) != (body['head'] == ZERO):
        raise ValueError('invalid witness genesis')
    return body


class CustodianMirror:
    """A fail-closed local mirror. A witness-first partial commit needs manual audit."""
    def __init__(self, path, client, enrollment, witness_pin):
        self.path = Path(path)
        self.client = client
        self.enrollment = validate_enrollment(enrollment)
        self.pin = witness_pin
        if not self.path.is_file():
            raise PermissionError('custody mirror missing; recovery is not initialization')
        private_database(self.path)

    @classmethod
    def initialize(cls, path, client, enrollment, pin):
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError('mirror exists')
        head = client('head', {})
        body = checked_head(head, pin, enrollment)
        if body['seq'] != 0:
            raise PermissionError('witness history exists; cannot reset with a new registry')
        private_database(path)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE mirror (singleton INTEGER PRIMARY KEY CHECK(singleton=1), head TEXT NOT NULL)')
            db.execute('INSERT INTO mirror VALUES (1,?)', (canonical(head).decode(),))
        return cls(path, client, enrollment, pin)

    def reserve(self, bindings):
        validate_bindings(bindings)
        with sqlite3.connect(self.path.resolve().as_uri() + '?mode=rw', uri=True, timeout=20) as db:
            db.execute('BEGIN IMMEDIATE')
            local = parse_unique_json(db.execute('SELECT head FROM mirror WHERE singleton=1').fetchone()[0].encode())
            old = checked_head(local, self.pin, self.enrollment)
            remote = checked_head(self.client('head', {}), self.pin, self.enrollment)
            if remote != old:
                raise PermissionError('custody/witness divergence; block and preserve incident')
            result = self.client('reserve', {'expected_head': old['head'], 'bindings': bindings})
            new = checked_head(result['head'], self.pin, self.enrollment)
            event = verify(result['event'], self.pin)
            if event['kind'] != 'RESERVED' or event['previous'] != old['head'] or event['seq'] != old['seq'] + 1 or event['bindings'] != bindings or new['head'] != sha(result['event']) or new['seq'] != old['seq'] + 1:
                raise PermissionError('invalid witness transition')
            db.execute('UPDATE mirror SET head=? WHERE singleton=1', (canonical(result['head']).decode(),))
            return result['event']

    def acknowledge_consumption(self, result):
        """Evaluator returns witnessed consumption; preserve custody continuity."""
        with sqlite3.connect(self.path.resolve().as_uri() + '?mode=rw', uri=True, timeout=20) as db:
            db.execute('BEGIN IMMEDIATE')
            old = checked_head(parse_unique_json(db.execute('SELECT head FROM mirror WHERE singleton=1').fetchone()[0].encode()), self.pin, self.enrollment)
            new = checked_head(result['head'], self.pin, self.enrollment)
            event = verify(result['event'], self.pin)
            remote = checked_head(self.client('head', {}), self.pin, self.enrollment)
            if event['kind'] != 'CONSUMED' or event['previous'] != old['head'] or event['seq'] != old['seq'] + 1 or new['head'] != sha(result['event']) or new['seq'] != old['seq'] + 1 or new != remote:
                raise PermissionError('invalid consumption or witness divergence')
            db.execute('UPDATE mirror SET head=? WHERE singleton=1', (canonical(result['head']).decode(),))


def evaluate_once(reservation, bindings, client, witness_pin, enrollment, inference):
    """Consume on the witness BEFORE entering the evaluator's inference callback.

    Inference implementation, protected data custody and truthful evaluator are
    still independently managed obligations; no worker can invoke this via its
    training credentials. Exceptions deliberately do not refund the attempt.
    """
    validate_bindings(bindings)
    payload = verify(reservation, witness_pin)
    if payload.get('kind') != 'RESERVED' or payload.get('enrollment_digest') != sha(enrollment) or payload.get('bindings') != bindings:
        raise PermissionError('frozen candidate/protocol/holdout substitution')
    head = checked_head(client('head', {}), witness_pin, enrollment)
    result = client('consume', {'expected_head': head['head'], 'reservation': reservation})
    event = verify(result['event'], witness_pin)
    if event.get('kind') != 'CONSUMED' or event.get('bindings') != bindings or event.get('attempt') != payload['attempt']:
        raise PermissionError('unverified consumption')
    measured = inference(payload['alpha'])
    return {'consumption': result, 'measurement': measured}
