"""Durable controller-owned public budgets, never a protected attempt authority."""
import sqlite3
from pathlib import Path

from .adaptive_contract import validate_job
from .model_contract import verify
from .spec import canonical, parse_unique_json, sha
from .storage import private_database


class AdaptiveRegistry:
    def __init__(self, path, pin, protocol, *, create=False):
        path = Path(path)
        self.pin = pin
        self.protocol = verify(protocol, pin)
        self.digest = sha(self.protocol)
        if self.protocol.get('schema') != 'saal-adaptive-campaign-v1' or self.protocol.get('qualification_status') != 'NOT_EVALUATED' or self.protocol.get('external_deployment_allowed') is not False:
            raise ValueError('controller research campaign required')
        budget = self.protocol['budget']
        if (type(budget) is not dict or set(budget) != {'campaign_id', 'max_runs', 'total_lifecycle_seconds', 'max_provisioning_attempts', 'automatic_training_retries', 'financial_cap'}
                or type(budget['max_provisioning_attempts']) is not int or budget['max_provisioning_attempts'] != 1
                or type(budget['automatic_training_retries']) is not int or budget['automatic_training_retries'] != 0
                or type(budget['max_runs']) is not int or not 1 <= budget['max_runs'] <= 25
                or type(budget['total_lifecycle_seconds']) is not int or not 30 <= budget['total_lifecycle_seconds'] <= 100000
                or budget['financial_cap'] is not False):
            raise ValueError('closed one-provision / no-retry budget')
        digests = self.protocol.get('job_digests')
        if type(digests) is not list or not 1 <= len(digests) <= budget['max_runs'] or len(set(digests)) != len(digests):
            raise ValueError('unique admitted job identities required')
        if create and path.exists():
            raise FileExistsError('preserve existing experimental budget')
        if not create and not path.exists():
            raise ValueError('signed budget not initialized')
        self.db = sqlite3.connect(str(private_database(path)), isolation_level=None, timeout=10)
        if create:
            self.db.executescript('CREATE TABLE budget (digest TEXT PRIMARY KEY,pin TEXT,protocol TEXT,provisions INTEGER,state TEXT); CREATE TABLE attempts (digest TEXT PRIMARY KEY,run_id TEXT UNIQUE,envelope TEXT,seconds INTEGER,state TEXT,outcome TEXT);')
            self.db.execute('INSERT INTO budget VALUES (?,?,?,?,?)', (self.digest, pin, canonical(protocol).decode(), 0, 'PREPARED'))
        if self.db.execute('SELECT digest,pin FROM budget').fetchall() != [(self.digest, pin)]:
            raise ValueError('budget / protocol owner mismatch')

    def reserve(self, envelope):
        job = validate_job(verify(envelope, self.pin))
        budget = job['budget']
        configured = self.protocol['budget']
        if any(budget[k] != configured[k] for k in ('campaign_id', 'max_runs', 'total_lifecycle_seconds', 'automatic_training_retries', 'financial_cap')) or sha(job) not in self.protocol['job_digests']:
            raise ValueError('unadmitted job / budget substitution')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            count, seconds = self.db.execute('SELECT COUNT(*),COALESCE(SUM(seconds),0) FROM attempts').fetchone()
            state, = self.db.execute('SELECT state FROM budget').fetchone()
            if state != 'PREPARED' or count >= configured['max_runs'] or seconds+budget['reserved_seconds'] > configured['total_lifecycle_seconds']:
                raise ValueError('durable campaign quota exhausted')
            self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,NULL)',
                            (sha(job), budget['run_id'], canonical(envelope).decode(), budget['reserved_seconds'], 'RESERVED'))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def begin_provisioning(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            digest, pin, protocol, provisions, state = self.db.execute('SELECT * FROM budget').fetchone()
            count, = self.db.execute('SELECT COUNT(*) FROM attempts').fetchone()
            if state != 'PREPARED' or provisions != 0 or count != len(self.protocol['job_digests']):
                raise ValueError('provision already consumed / incomplete reservations')
            self.db.execute("UPDATE budget SET provisions=1,state='PROVISIONING'")
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def start(self, digest):
        if self.db.execute('SELECT state FROM budget').fetchone()[0] != 'PROVISIONING':
            raise ValueError('no consumed provisioning admission')
        if self.db.execute("UPDATE attempts SET state='RUNNING' WHERE digest=? AND state='RESERVED'", (digest,)).rowcount != 1:
            raise ValueError('attempt already consumed')

    def finish(self, digest, outcome, completed):
        if self.db.execute('UPDATE attempts SET state=?,outcome=? WHERE digest=? AND state=?',
                           ('COMPLETED' if completed else 'INTERRUPTED', canonical(outcome).decode(), digest, 'RUNNING')).rowcount != 1:
            raise ValueError('invalid attempt transition')

    def end(self, succeeded):
        if succeeded and self.db.execute("SELECT COUNT(*) FROM attempts WHERE state='COMPLETED'").fetchone()[0] != len(self.protocol['job_digests']):
            raise ValueError('cannot complete an unfinished campaign')
        if self.db.execute("UPDATE budget SET state=? WHERE state='PROVISIONING'", ('COMPLETED' if succeeded else 'FAILED',)).rowcount != 1:
            raise ValueError('campaign terminal transition')

    def records(self):
        state, provisions = self.db.execute('SELECT state,provisions FROM budget').fetchone()
        return {'schema': 'saal-adaptive-budget-export-v1', 'protocol_digest': self.digest,
                'campaign_state': state, 'provisioning_attempts': provisions,
                'attempts': [{'digest': d, 'run_id': n, 'reserved_seconds': s, 'state': t,
                              'outcome': parse_unique_json(o.encode()) if o else None}
                             for d, n, s, t, o in self.db.execute('SELECT digest,run_id,seconds,state,outcome FROM attempts ORDER BY rowid')],
                'rollback_resistant': False, 'financial_cap': False, 'protected_alpha_consumed': False}

    def close(self):
        self.db.close()
