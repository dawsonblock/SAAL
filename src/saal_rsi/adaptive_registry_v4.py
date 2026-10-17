"""Separate one-use local budget for freshly signed v4 public campaigns."""
import sqlite3
from pathlib import Path

from .adaptive_evaluator_v4 import validate_jobs, validate_protocol
from .model_contract import verify
from .spec import canonical, parse_unique_json, sha
from .storage import private_database


class AdaptiveRegistryV4:
    """Local atomic accounting; explicitly not rollback-resistant custody."""
    def __init__(self, path, pin, protocol_envelope, job_envelopes, *, create=False):
        path = Path(path)
        self.pin = pin
        self.protocol = verify(protocol_envelope, pin)
        validate_protocol(self.protocol)
        self.jobs = validate_jobs(self.protocol, job_envelopes, pin)
        self.digest = sha(self.protocol)
        self.job_by_digest = {sha(job): job for job in self.jobs}
        if len(self.job_by_digest) != len(self.jobs):
            raise ValueError('duplicate v4 job digests')
        if create and path.exists():
            raise FileExistsError('preserve existing budget; never reset or reuse it')
        if not create and not path.exists():
            raise ValueError('fresh signed v4 budget is not initialized')
        self.db = sqlite3.connect(str(private_database(path)), isolation_level=None, timeout=10)
        if create:
            self.db.executescript('CREATE TABLE budget (digest TEXT PRIMARY KEY,pin TEXT,protocol TEXT,provisions INTEGER,state TEXT); CREATE TABLE attempts (digest TEXT PRIMARY KEY,run_id TEXT UNIQUE,envelope TEXT,seconds INTEGER,state TEXT,outcome TEXT);')
            self.db.execute('INSERT INTO budget VALUES (?,?,?,?,?)',
                            (self.digest, pin, canonical(self.protocol).decode(), 0, 'PREPARED'))
        if self.db.execute('SELECT digest,pin FROM budget').fetchall() != [(self.digest, pin)]:
            raise ValueError('v4 budget / signed protocol owner mismatch')

    def reserve_all(self, envelopes):
        if self.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] != 0:
            raise ValueError('v4 reservations already exist; no reset or retry')
        jobs = validate_jobs(self.protocol, envelopes, self.pin)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            total_seconds = 0
            for envelope, job in zip(envelopes, jobs):
                digest = sha(job)
                seconds = job['budget']['reserved_seconds']
                total_seconds += seconds
                if digest not in self.job_by_digest:
                    raise ValueError('job absent from signed v4 protocol')
                self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,NULL)',
                    (digest, job['budget']['run_id'], canonical(envelope).decode(), seconds, 'RESERVED'))
            if (total_seconds > self.protocol['budget']['total_lifecycle_seconds']
                    or self.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
                       != len(self.protocol['job_digests'])):
                raise ValueError('v4 campaign reservations exceed signed budget / run count')
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
                raise ValueError('v4 provisioning requires complete reservations and unused admission')
            self.db.execute("UPDATE budget SET provisions=1,state='PROVISIONING'")
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def start(self, digest):
        if self.db.execute('SELECT state FROM budget').fetchone()[0] != 'PROVISIONING':
            raise ValueError('v4 provisioning admission has not been consumed')
        if self.db.execute("UPDATE attempts SET state='RUNNING' WHERE digest=? AND state='RESERVED'", (digest,)).rowcount != 1:
            raise ValueError('v4 attempt already consumed or unknown')

    def finish(self, digest, outcome, completed):
        if type(completed) is not bool or type(outcome) is not dict:
            raise ValueError('durable v4 outcome required')
        target = 'COMPLETED' if completed else 'INTERRUPTED'
        if self.db.execute('UPDATE attempts SET state=?,outcome=? WHERE digest=? AND state=?',
                           (target, canonical(outcome).decode(), digest, 'RUNNING')).rowcount != 1:
            raise ValueError('invalid v4 attempt transition')

    def end(self, succeeded):
        if type(succeeded) is not bool:
            raise ValueError('explicit v4 terminal decision required')
        if succeeded and self.db.execute("SELECT COUNT(*) FROM attempts WHERE state='COMPLETED'").fetchone()[0] != len(self.protocol['job_digests']):
            raise ValueError('cannot complete an unfinished v4 campaign')
        if self.db.execute('UPDATE budget SET state=? WHERE state=?',
                           ('COMPLETED' if succeeded else 'FAILED', 'PROVISIONING')).rowcount != 1:
            raise ValueError('v4 campaign already terminal or not provisioned')

    def records(self):
        state, provisions = self.db.execute('SELECT state,provisions FROM budget').fetchone()
        attempts = [{'digest': d, 'run_id': n, 'reserved_seconds': s, 'state': t,
                     'outcome': parse_unique_json(o.encode()) if o else None}
                    for d, n, s, t, o in self.db.execute(
                        'SELECT digest,run_id,seconds,state,outcome FROM attempts ORDER BY rowid')]
        return {'schema': 'saal-adaptive-budget-export-v4', 'protocol_digest': self.digest,
                'campaign_state': state, 'provisioning_attempts': provisions, 'attempts': attempts,
                'rollback_resistant': False,
                'compute_units_cap': self.protocol['budget']['max_compute_units'],
                'financial_cap': False,
                'protected_alpha_consumed': False, 'qualification_status': 'NOT_EVALUATED',
                'external_deployment_allowed': False}

    def close(self):
        self.db.close()
