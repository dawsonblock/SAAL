"""Signed training jobs through local Python or the official Colab CLI.

Jobs are a fixed worker plus public training/development data, never a repository
upload. Every backend result remains untrusted until independently evaluated.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from .model_contract import verify,validate_job,worker_bytes,VERSIONS,LABELS,baseline_input
from .spec import canonical,sha,parse_unique_json

NAMES={'result.json','baseline.safetensors','adapter.safetensors'}
MAX_ARTIFACT=100_000_000

def validate_bundle(directory,payload):
    if payload.get("schema")=="saal-model-job-v4":
        from .neural_artifacts import validate_neural_bundle
        return validate_neural_bundle(directory,payload)
    validate_job(payload);root=Path(directory)
    if root.is_symlink() or not root.is_dir():raise ValueError('artifact directory')
    files=list(root.iterdir())
    if {p.name for p in files}!=NAMES or any(p.is_symlink() or not p.is_file() for p in files):raise ValueError('artifact inventory')
    if any(p.stat().st_size>MAX_ARTIFACT for p in files) or sum(p.stat().st_size for p in files)>MAX_ARTIFACT:raise ValueError('artifact budget')
    result=parse_unique_json((root/'result.json').read_bytes())
    fields={'schema','job_digest','worker_sha256','runtime','losses','loss_measurement','selection','training_history','development_predictions','elapsed_seconds','rank','artifact_hashes','qualification_passed','external_deployment_allowed','status'}
    if set(result)!=fields or result['schema']!='saal-model-result-v2' or result['job_digest']!=sha(payload) or result['worker_sha256']!=payload['worker_sha256']:raise ValueError('artifact job/code binding')
    if result['qualification_passed'] is not False or result['external_deployment_allowed'] is not False or result['status']!='UNTRUSTED_DEVELOPMENT_RESULT':raise ValueError('worker self-approval forbidden')
    if type(result['rank'])is not int or result['rank']!=payload['rank']:raise ValueError('adapter rank')
    if type(result['elapsed_seconds'])not in (int,float) or not math.isfinite(result['elapsed_seconds']) or not 0<=result['elapsed_seconds']<=payload['max_seconds']+30:raise ValueError('reported elapsed budget')
    if type(result['losses'])is not list or len(result['losses'])!=2 or any(type(x)not in (int,float) or not math.isfinite(x) or x<0 for x in result['losses']):raise ValueError('loss schema')
    predictions=result['development_predictions']
    if type(predictions)is not dict or set(predictions)!={'baseline','candidate'} or any(type(v)is not list or len(v)!=len(payload['development']) or any(x not in LABELS for x in v) for v in predictions.values()):raise ValueError('development prediction schema')
    from .model_worker import development_score
    selection=result['selection']
    if type(selection)is not dict or set(selection)!={'selected_adapter_step','trained_adapter_steps','development_score','policy','objective','class_weights','labels_absent_from_training'}:raise ValueError('selection schema')
    step=selection['selected_adapter_step']
    if type(step)is not int or not 0<=step<=payload['adapter_steps'] or (step%10 and step!=payload['adapter_steps']):raise ValueError('selected checkpoint step')
    if selection['trained_adapter_steps']!=payload['adapter_steps'] or selection['policy']!=payload['checkpoint_selection'] or selection['objective']!=payload['adapter_objective']:raise ValueError('selection commitment')
    if selection['development_score']!=list(development_score(predictions['candidate'],payload['development'])):raise ValueError('selected checkpoint score')
    if result['loss_measurement']!='full-training-unweighted-cross-entropy-at-exported-checkpoints':raise ValueError('loss measurement')
    from collections import Counter
    counts=Counter(r['label'] for r in payload['adapter_training'])
    expected_weights=[len(payload['adapter_training'])/(len(counts)*counts[label]) if counts[label] else 0 for label in LABELS]
    weights=selection['class_weights']
    if type(weights)is not list or len(weights)!=5 or any(type(v)not in (int,float) or not math.isfinite(v) or not math.isclose(v,w,rel_tol=1e-6,abs_tol=1e-7) for v,w in zip(weights,expected_weights)):raise ValueError('class balance weights')
    if selection['labels_absent_from_training']!=[label for label in LABELS[:4] if not counts[label]]:raise ValueError('training label inventory')
    history=result['training_history']
    if type(history)is not list or len(history)!=(payload['baseline_steps']+9)//10+(payload['adapter_steps']+9)//10:raise ValueError('training history inventory')
    expected_pairs=[(phase,step) for phase,steps in [('baseline',payload['baseline_steps']),('adapter',payload['adapter_steps'])] for step in range(1,steps+1) if step%10==0 or step==steps]
    for entry,(phase,expected_step) in zip(history,expected_pairs):
        if type(entry)is not dict or set(entry)!={'phase','step','completed_steps','total_steps','objective_loss','elapsed_seconds','development_score','gpu_allocated_bytes','gpu_reserved_bytes'}:raise ValueError('history schema')
        if entry['phase']!=phase or type(entry['step'])is not int or entry['step']!=expected_step:raise ValueError('history progress')
        if any(type(entry[k])not in (int,float) or not math.isfinite(entry[k]) or entry[k]<0 for k in ('objective_loss','elapsed_seconds')):raise ValueError('history numeric')
    versions=result['runtime']['versions']
    if {k:v.split('+')[0] for k,v in versions.items()}!=VERSIONS:raise ValueError('runtime dependency identity')
    if set(result['artifact_hashes'])!=NAMES-{'result.json'}:raise ValueError('artifact hash inventory')
    for name,h in result['artifact_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('artifact hash mismatch')
    if payload['baseline_sha256'] is not None and result['artifact_hashes']['baseline.safetensors']!=payload['baseline_sha256']:raise ValueError('baseline changed')
    return result

def bundle_digest(root,payload):
    validate_bundle(root,payload)
    return sha({name:hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() for name in sorted(artifact_names(payload))})

def artifact_names(payload):
    if payload.get("schema")=="saal-model-job-v4":
        from .neural_worker import FILES
        return FILES
    return NAMES

def unpack_result(archive,destination,*,names=None):
    names=NAMES if names is None else names
    destination=Path(destination)
    if destination.exists():raise FileExistsError('artifact destination exists')
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist()
        if len(entries)!=len(names) or {e.filename for e in entries}!=names or any(e.file_size>MAX_ARTIFACT or ((e.external_attr>>16)&0o170000)==0o120000 for e in entries) or sum(e.file_size for e in entries)>MAX_ARTIFACT:raise ValueError('unsafe artifact archive')
        destination.mkdir()
        try:
            for entry in entries:
                with z.open(entry) as incoming,(destination/entry.filename).open('xb') as output:
                    total=0
                    while chunk:=incoming.read(1024*1024):
                        total+=len(chunk)
                        if total>entry.file_size or total>MAX_ARTIFACT:raise ValueError('artifact decompression bound')
                        output.write(chunk)
        except Exception:
            shutil.rmtree(destination);raise

def run_local(envelope,pin,out,baseline_path=None):
    if verify(envelope,pin).get("schema")=="saal-model-job-v4":
        from .neural_backend import run_neural_local
        return run_neural_local(envelope,pin,out,baseline_path)
    payload=verify(envelope,pin);validate_job(payload);out=Path(out).resolve()
    if out.exists():raise FileExistsError('output exists')
    with tempfile.TemporaryDirectory(prefix='saal-model-public-') as temporary:
        root=Path(temporary);(root/'job.json').write_bytes(canonical(envelope));(root/'worker.py').write_bytes(worker_bytes())
        env={'PATH':os.environ.get('PATH',''),'HOME':str(root),'TMPDIR':str(root),'PYTHONHASHSEED':str(payload['seed'])}
        extra=[]
        if payload['baseline_sha256'] is not None:
            raw=baseline_input(baseline_path,payload['baseline_sha256']);(root/'baseline-input.safetensors').write_bytes(raw);extra=['--baseline',str(root/'baseline-input.safetensors')]
        subprocess.run([sys.executable,'-I',str(root/'worker.py'),'--job',str(root/'job.json'),'--public-key',pin,'--out',str(out),*extra],
                       cwd=root,env=env,check=True,timeout=payload['max_seconds']+30,capture_output=True)
    validate_bundle(out,payload);return {'backend':'local','artifact_digest':bundle_digest(out,payload),'qualification_status':'NOT_EVALUATED'}

def colab_plan(envelope,pin,workspace,baseline_path=None):
    if verify(envelope,pin).get("schema")=="saal-model-job-v4":
        from .neural_backend import neural_colab_plan
        return neural_colab_plan(envelope,pin,workspace,baseline_path)
    payload=verify(envelope,pin);validate_job(payload);workspace=Path(workspace).resolve();workspace.mkdir(exist_ok=False)
    session='saal-'+uuid.uuid4().hex[:20]
    (workspace/'job.json').write_bytes(canonical(envelope));(workspace/'worker.py').write_bytes(worker_bytes())
    # exec transmits only this reviewed script; uploaded files are an explicit allowlist.
    installer="""import json, shutil, subprocess, sys, threading, time
_done=threading.Event()
def heartbeat():
    while not _done.wait(5):print(json.dumps({'monitor':'dependency_heartbeat','at':time.time()}),flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
print(json.dumps({'monitor':'dependency_start'}),flush=True)
command=['uv','pip','install','--system'] if shutil.which('uv') else [sys.executable,'-m','pip','install']
try:
    subprocess.run(command+PACKAGES,check=True,timeout=300)
finally:_done.set()
print(json.dumps({'monitor':'dependency_complete'}),flush=True)
""".replace('PACKAGES',repr([k+'=='+v for k,v in VERSIONS.items()]+['numpy==2.4.6']))
    (workspace/'install.py').write_text(installer)
    runner="""import runpy, sys, zipfile
from pathlib import Path
sys.argv=['saal_worker.py','--job','/content/saal_job.json','--public-key',PIN,'--out','/content/saal_result','--require-gpu',GPU]
runpy.run_path('/content/saal_worker.py',run_name='__main__')
with zipfile.ZipFile('/content/saal_result.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ('result.json','baseline.safetensors','adapter.safetensors'):
        z.write(Path('/content/saal_result')/name,name)
""".replace('PIN',repr(pin)).replace('GPU',repr(payload['gpu']))
    if payload['baseline_sha256'] is not None:
        raw=baseline_input(baseline_path,payload['baseline_sha256']);(workspace/'baseline-input.safetensors').write_bytes(raw)
        runner=runner.replace("'--require-gpu',","'--baseline','/content/saal_baseline.safetensors','--require-gpu',")
    (workspace/'run.py').write_text(runner)
    commands=[['colab','new','-s',session,'--gpu',payload['gpu']],
      ['colab','exec','-s',session,'-f',str(workspace/'install.py'),'--timeout','330'],
      ['colab','upload','-s',session,str(workspace/'job.json'),'/content/saal_job.json'],
      ['colab','upload','-s',session,str(workspace/'worker.py'),'/content/saal_worker.py'],
      ['colab','exec','-s',session,'-f',str(workspace/'run.py'),'--timeout',str(payload['max_seconds']+30)],
      ['colab','download','-s',session,'/content/saal_result.zip',str(workspace/'result.zip')],
      ['colab','log','-s',session,'-o',str(workspace/'runtime-log.jsonl')],
      ['colab','stop','-s',session]]
    phases=['PROVISIONING','DEPENDENCIES','UPLOAD_JOB','UPLOAD_WORKER','TRAINING','DOWNLOAD','EXPORT_LOGS']
    if payload['baseline_sha256'] is not None:
        commands.insert(4,['colab','upload','-s',session,str(workspace/'baseline-input.safetensors'),'/content/saal_baseline.safetensors']);phases.insert(4,'UPLOAD_BASELINE')
    return {'phases':phases,'schema':'saal-colab-plan-v2','session':session,'job_digest':sha(payload),'commands':commands,
            'includes_final_holdout':False,'estimated_max_seconds':payload['max_seconds']+330,
            'billing_limit_enforced':False}

def run_colab(envelope,pin,out,workspace,baseline_path=None):
    if verify(envelope,pin).get("schema")=="saal-model-job-v4":
        from .neural_backend import run_neural_colab
        return run_neural_colab(envelope,pin,out,workspace,baseline_path)
    from .model_monitor import execute_colab_plan
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('output exists')
    plan=colab_plan(envelope,pin,workspace,baseline_path)
    execute_colab_plan(plan,Path(workspace))
    from .model_monitor import write_json
    state_path=Path(workspace)/'monitor-state.json';state=json.loads(state_path.read_text())
    try:
        state['phase']='VERIFYING_ARTIFACTS';state['terminal']=False;write_json(state_path,state)
        unpack_result(Path(workspace)/'result.zip',out);payload=verify(envelope,pin);result=validate_bundle(out,payload)
        if result['runtime']['device']!='cuda' or payload['gpu'] not in result['runtime']['gpu']:raise ValueError('GPU execution identity')
        from .model_worker import infer
        replay=infer(out,[r['prompt'] for r in payload['development']])
        parity={k:replay[k]==result['development_predictions'][k] for k in ('baseline','candidate')}
        if not all(parity.values()):raise ValueError('checkpoint prediction replay mismatch')
        state.update(phase='COMPLETED',terminal=True,artifact_validation='PASSED',checkpoint_reload_prediction_parity=parity,
                     runtime=result['runtime'],selection=result['selection'])
        write_json(state_path,state)
        return {'backend':'colab','session':plan['session'],'artifact_digest':bundle_digest(out,payload),'qualification_status':'NOT_EVALUATED',
            'monitor':str(state_path.resolve()),'cleanup_verified':True,'checkpoint_reload_prediction_parity':parity}
    except BaseException as exc:
        state.update(phase='FAILED',terminal=True,artifact_validation='FAILED',error={'type':type(exc).__name__,'detail':str(exc)})
        write_json(state_path,state);raise


class JobBudget:
    """One-shot local submission reservations. Not an account billing cap."""
    def __init__(self,path,pin,*,max_jobs=None,max_seconds=None):
        import sqlite3
        from .storage import private_database
        from .validation import key_bytes
        path=Path(path);create=max_jobs is not None
        if create and path.exists():raise FileExistsError('budget ledger exists')
        if not create and not path.exists():raise ValueError('initialize submission budget first')
        if create and (type(max_jobs)is not int or not 1<=max_jobs<=100 or type(max_seconds)is not int or not 10<=max_seconds<=100000):raise ValueError('budget bounds')
        self.db=sqlite3.connect(str(private_database(path)),isolation_level=None,timeout=10);self.pin=key_bytes(pin).hex()
        try:
            if create:
                self.db.executescript('CREATE TABLE config (pin TEXT, jobs INTEGER, seconds INTEGER); CREATE TABLE submissions (digest TEXT PRIMARY KEY, seconds INTEGER);')
                self.db.execute('INSERT INTO config VALUES (?,?,?)',(self.pin,max_jobs,max_seconds))
            cfg=self.db.execute('SELECT pin,jobs,seconds FROM config').fetchall()
            if len(cfg)!=1 or cfg[0][0]!=self.pin:raise ValueError('submission budget owner')
        except BaseException:self.db.close();raise
    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def reserve(self,envelope,seconds):
        payload=verify(envelope,self.pin);validate_job(payload)
        if type(seconds)is not int or seconds not in (payload['max_seconds']+30,payload['max_seconds']+(1100 if payload['schema']=='saal-model-job-v4' else 330)):raise ValueError('submission time')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            cfg=self.db.execute('SELECT jobs,seconds FROM config').fetchone()
            used=self.db.execute('SELECT COUNT(*),COALESCE(SUM(seconds),0) FROM submissions').fetchone()
            if used[0]>=cfg[0] or used[1]+seconds>cfg[1]:raise ValueError('submission budget exhausted')
            self.db.execute('INSERT INTO submissions VALUES (?,?)',(sha(payload),seconds));self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise
