"""One-use signed jobs and durable negative outcomes; local trust, no promotion."""
from pathlib import Path
import sqlite3
from .storage import private_database
from .spec import canonical,sha,parse_unique_json
from .model_contract import verify
from .pretrained_contract import validate_job

class LearningExperimentRegistry:
    def __init__(self,path,pin,max_jobs=None,max_seconds=None):
        from .validation import key_bytes
        self.pin=key_bytes(pin).hex();p=Path(path);new=max_jobs is not None
        if new and p.exists():raise FileExistsError('existing budget')
        if not new and not p.exists():raise ValueError('budget not initialized')
        self.db=sqlite3.connect(str(private_database(p)),isolation_level=None,timeout=10)
        if new:
            if type(max_jobs)is not int or not 1<=max_jobs<=100 or type(max_seconds)is not int or not 30<=max_seconds<=100000:raise ValueError('campaign budget')
            self.db.executescript('CREATE TABLE budget (pin TEXT,jobs INTEGER,seconds INTEGER); CREATE TABLE attempts (digest TEXT PRIMARY KEY,envelope TEXT,seconds INTEGER,state TEXT,outcome TEXT);')
            self.db.execute('INSERT INTO budget VALUES (?,?,?)',(self.pin,max_jobs,max_seconds))
        if self.db.execute('SELECT pin FROM budget').fetchall()!=[(self.pin,)]:raise ValueError('budget owner mismatch')
    def close(self):self.db.close()
    def reserve(self,envelope):
        job=verify(envelope,self.pin);validate_job(job);digest=sha(job);seconds=job['max_seconds']+1400
        self.db.execute('BEGIN IMMEDIATE')
        try:
            _,jobs,limit=self.db.execute('SELECT * FROM budget').fetchone();used,total=self.db.execute('SELECT COUNT(*),COALESCE(SUM(seconds),0) FROM attempts').fetchone()
            if used>=jobs or total+seconds>limit:raise ValueError('durable budget exhausted')
            self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,NULL)',(digest,canonical(envelope).decode(),seconds,'RESERVED'));self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise
        return digest
    def start(self,digest):
        if self.db.execute("UPDATE attempts SET state='RUNNING' WHERE digest=? AND state='RESERVED'",(digest,)).rowcount!=1:raise ValueError('attempt consumed')
    def finish(self,digest,outcome,completed):
        if self.db.execute("UPDATE attempts SET state=?,outcome=? WHERE digest=? AND state='RUNNING'",('COMPLETED' if completed else 'INTERRUPTED',canonical(outcome).decode(),digest)).rowcount!=1:raise ValueError('attempt transition')
    def records(self):return {'attempts':[{'digest':d,'envelope':parse_unique_json(e.encode()),'reserved_seconds':s,'state':t,'outcome':parse_unique_json(o.encode()) if o else None} for d,e,s,t,o in self.db.execute('SELECT * FROM attempts ORDER BY rowid')],
       'rollback_resistant':False,'financial_cap':False,'authority':'controller-owned development registry'}
