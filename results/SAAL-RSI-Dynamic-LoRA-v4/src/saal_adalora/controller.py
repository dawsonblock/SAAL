"""One-use local public exploration receipt; not a protected alpha authority."""
from pathlib import Path
import json,sqlite3,sys,time,subprocess
from saal_rsi.spec import canonical,sha,parse_unique_json
from saal_rsi.model_contract import sign,verify
from saal_rsi.pretrained_contract import worker_identity
from .contract import validate,source_identity,ADAPTER,FIXED_PIN,SCREEN

def reserve(ledger,digest):
    ledger=Path(ledger)
    if ledger.exists():raise ValueError('dynamic ledger already exists; no reset/reuse')
    c=sqlite3.connect(ledger);c.execute('CREATE TABLE attempt (digest TEXT PRIMARY KEY,state TEXT NOT NULL,outcome TEXT)');c.execute('INSERT INTO attempt VALUES (?,?,NULL)',(digest,'RESERVED'));c.commit();c.close()
def consume(ledger,digest):
    if not Path(ledger).is_file():raise ValueError('missing dynamic ledger')
    c=sqlite3.connect(ledger);c.execute('BEGIN IMMEDIATE');row=c.execute('SELECT state FROM attempt WHERE digest=?',(digest,)).fetchone()
    if row!=('RESERVED',):c.rollback();c.close();raise ValueError('spent or substituted attempt')
    c.execute('UPDATE attempt SET state=? WHERE digest=?',('RUNNING',digest));c.commit();c.close()
def finish(ledger,digest,outcome):
    c=sqlite3.connect(ledger);c.execute('BEGIN IMMEDIATE')
    if c.execute('SELECT state FROM attempt WHERE digest=?',(digest,)).fetchone()!=('RUNNING',):raise ValueError('attempt transition')
    c.execute('UPDATE attempt SET state=?,outcome=? WHERE digest=?',(outcome['state'],canonical(outcome).decode(),digest));c.commit();c.close()
def prepare(root,key_path,ledger,fixed_root):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    root=Path(root);root.mkdir(exist_ok=False);fixed=Path(fixed_root);pin=(fixed/'controller.pub').read_text().strip()
    if pin!=FIXED_PIN:raise ValueError('fixed control identity')
    envelope=parse_unique_json((fixed/'lora-job.json').read_bytes());reference=verify(envelope,pin)
    key=Ed25519PrivateKey.from_private_bytes(Path(key_path).read_bytes())
    job={'schema':'saal-adalora-research-job-v1','native_configuration_reference':reference,'dynamic_adapter':ADAPTER,'dynamic_source_identity':source_identity(),'base_worker_identity':worker_identity(),'fixed_comparison_digest':sha(reference),'fixed_comparison_envelope':envelope,'fixed_comparison_pin':pin,'screen':SCREEN,'resource_budget_id':'dynamic4_'+source_identity()[7:23],'reserved_seconds':2300,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    validate(job);reserve(ledger,sha(job));(root/'signed-job.json').write_bytes(canonical(sign(job,key)));(root/'controller.pub').write_text(key.public_key().public_bytes_raw().hex()+'\n')
def run(root,ledger,cache):
    root=Path(root);pin=(root/'controller.pub').read_text().strip();job=verify(parse_unique_json((root/'signed-job.json').read_bytes()),pin);validate(job);digest=sha(job)
    consume(ledger,digest);started=time.monotonic()
    try:
        subprocess.run([sys.executable,'-m','saal_adalora.worker','train',str(root),str(cache)],check=True,timeout=1900)
        subprocess.run([sys.executable,'-m','saal_adalora.worker','replay',str(root),str(cache)],check=True,timeout=300)
        outcome={'state':'COMPLETED','job_digest':digest,'result_digest':sha(parse_unique_json((root/'result.json').read_bytes())),'fresh_replay_digest':sha(parse_unique_json((root/'fresh-replay.json').read_bytes()))}
    except Exception as exc:outcome={'state':'INTERRUPTED','job_digest':digest,'error':{'type':type(exc).__name__,'detail':str(exc)}}
    outcome.update(elapsed_seconds=time.monotonic()-started,qualification_status='NOT_EVALUATED',external_deployment_allowed=False);finish(ledger,digest,outcome)
    (root/'outcome.json').write_bytes(canonical(outcome));c=sqlite3.connect(ledger);rows=c.execute('SELECT digest,state,outcome FROM attempt').fetchall();c.close()
    (root/'ledger-export.json').write_bytes(canonical({'authority':'controller-owned public development ledger','reserved_seconds':2300,'rollback_resistant':False,'financial_cap':False,'protected_alpha_used':False,'attempts':[{'digest':d,'state':s,'outcome':json.loads(o) if o else None} for d,s,o in rows]}))
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(*sys.argv[2:])
    elif sys.argv[1]=='run':run(*sys.argv[2:])
    else:raise ValueError('closed dynamic controller operation')
