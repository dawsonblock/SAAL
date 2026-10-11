"""Durable research attempts, including negative/interrupted results. Not rollback-proof."""
from pathlib import Path
import sqlite3
from .spec import canonical,sha,parse_unique_json
from .storage import private_database

class ExperimentRegistry:
    def __init__(self,path):
        self.db=sqlite3.connect(str(private_database(Path(path))),isolation_level=None,timeout=10)
        self.db.execute('CREATE TABLE IF NOT EXISTS experiments (digest TEXT PRIMARY KEY, protocol TEXT NOT NULL, state TEXT NOT NULL, outcome TEXT)')
    def __enter__(self):return self
    def __exit__(self,*a):self.db.close()
    def register(self,protocol):
        if type(protocol)is not dict or set(protocol)!={'schema','seeds','modes','baseline_steps','adaptation_steps','dataset_digest','worker_digest','model_spec','max_run_seconds','backend','selection_policy','controller_digest','qualification_status','external_deployment_allowed'} or protocol['schema']!='saal-neural-campaign-v1' or protocol['qualification_status']!='NOT_EVALUATED' or protocol['external_deployment_allowed'] is not False:raise ValueError('campaign protocol')
        if type(protocol['seeds'])is not list or not 1<=len(protocol['seeds'])<=5 or len(set(protocol['seeds']))!=len(protocol['seeds']) or any(type(s)is not int or not 0<=s<=2**30 for s in protocol['seeds']):raise ValueError('registered seeds')
        if protocol['modes']!=['frozen','full','lora-r4','lora-r8'] or protocol['backend'] not in ('local','colab'):raise ValueError('registered comparisons')
        for k,lo,hi in [('baseline_steps',1,2000),('adaptation_steps',1,2000),('max_run_seconds',10,3600)]:
            if type(protocol[k])is not int or not lo<=protocol[k]<=hi:raise ValueError('campaign budget')
        from .neural_contract import validate_spec,SELECTION
        validate_spec(protocol['model_spec'])
        if protocol['selection_policy']!=SELECTION:raise ValueError('campaign selection policy')
        digest=sha(protocol);self.db.execute('INSERT INTO experiments VALUES (?,?,?,NULL)',(digest,canonical(protocol).decode(),'REGISTERED'));return digest
    def start(self,digest):
        if self.db.execute("UPDATE experiments SET state='RUNNING' WHERE digest=? AND state='REGISTERED'",(digest,)).rowcount!=1:raise ValueError('unknown/consumed experiment')
    def finish(self,digest,outcome,*,completed):
        state='COMPLETED' if completed else 'INTERRUPTED'
        if self.db.execute("UPDATE experiments SET state=?, outcome=? WHERE digest=? AND state='RUNNING'",(state,canonical(outcome).decode(),digest)).rowcount!=1:raise ValueError('experiment transition')
    def records(self):return [{'digest':d,'protocol':parse_unique_json(p.encode()),'state':s,'outcome':parse_unique_json(o.encode()) if o else None} for d,p,s,o in self.db.execute('SELECT digest,protocol,state,outcome FROM experiments ORDER BY rowid')]
