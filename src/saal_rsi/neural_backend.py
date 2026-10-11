"""Versioned isolated-child and named Colab execution, preserving legacy dispatch."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from .neural_contract import worker_sources,validate_neural_job
from .neural_artifacts import validate_neural_bundle
from .neural_worker import FILES,seal_inventory
from .model_contract import verify
from .spec import canonical,sha,parse_unique_json

INSTALL_SECONDS=900
CLOUD_OVERHEAD=1100

def baseline_bytes(path,expected):
    import stat
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 1<=info.st_size<=5_000_000:raise ValueError('neural baseline bound')
        with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(5_000_001)
    finally:os.close(fd)
    if len(raw)>5_000_000 or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('neural baseline hash')
    return raw

def stage_worker(root):
    package=root/'saal_rsi';package.mkdir();(package/'__init__.py').write_text('')
    for name,raw in worker_sources().items():(package/name).write_bytes(raw)
    launcher=root/'worker.py'
    launcher.write_text("from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom saal_rsi.neural_worker import main\nmain()\n")
    return launcher

def attach_cleanup(out,evidence):
    root=Path(out);result=parse_unique_json((root/'result.json').read_bytes())
    (root/'cleanup_evidence.json').write_bytes(canonical(evidence));seal_inventory(root,result)

def run_neural_local(envelope,pin,out,baseline_path=None):
    job=verify(envelope,pin);validate_neural_job(job);out=Path(out).resolve()
    if out.exists():raise FileExistsError('output exists')
    with tempfile.TemporaryDirectory(prefix='saal-neural-worker-') as temporary:
        root=Path(temporary);launcher=stage_worker(root);(root/'job.json').write_bytes(canonical(envelope))
        args=[sys.executable,'-I',str(launcher),'--job',str(root/'job.json'),'--public-key',pin,'--out',str(out)]
        if job['baseline_sha256'] is not None:
            (root/'baseline.safetensors').write_bytes(baseline_bytes(baseline_path,job['baseline_sha256']));args+=['--baseline',str(root/'baseline.safetensors')]
        env={'PATH':os.environ.get('PATH',''),'HOME':str(root),'TMPDIR':str(root),'PYTHONHASHSEED':str(job['seed'])}
        try:
            subprocess.run(args,cwd=root,env=env,check=True,timeout=job['max_seconds']+30,capture_output=True)
        except BaseException:
            # Preserve partial artifacts for investigation; no result/cleanup success is manufactured.
            raise
    attach_cleanup(out,{'authority':'controller','confirmed':True,'status':'LOCAL_CHILD_EXITED','scope':'child process only; same-UID access is not an OS sandbox'})
    validate_neural_bundle(out,job,replay=True)
    from .model_backend import bundle_digest
    return {'backend':'local','artifact_digest':bundle_digest(out,job),'qualification_status':'NOT_EVALUATED','cleanup_verified':True}

def inference_neural(bundle,job,records):
    validate_neural_bundle(bundle,job)
    with tempfile.TemporaryDirectory(prefix='saal-neural-inference-') as temporary:
        root=Path(temporary);launcher=stage_worker(root);request={'model_spec':job['model_spec'],'mode':job['mode'],'prompts':[r['prompt'] for r in records]}
        (root/'request.json').write_bytes(canonical(request))
        subprocess.run([sys.executable,'-I',str(launcher),'--infer','--bundle',str(Path(bundle).resolve()),'--request',str(root/'request.json'),'--out',str(root/'predictions.json')],
            cwd=root,env={'PATH':os.environ.get('PATH',''),'HOME':str(root),'TMPDIR':str(root)},check=True,capture_output=True,timeout=180)
        raw=parse_unique_json((root/'predictions.json').read_bytes())
    from .neural_artifacts import validate_prediction
    for k in ('baseline','candidate'):validate_prediction(raw[k],len(records))
    if raw['runtime']['model_spec_digest']!=sha(job['model_spec']) or raw['runtime']['worker_sha256']!=job['worker_sha256']:raise ValueError('inference identity')
    return {'baseline':raw['baseline']['labels'],'candidate':raw['candidate']['labels'],'runtime':raw['runtime'],
            'probabilities':{k:raw[k]['probabilities'] for k in ('baseline','candidate')}}

def neural_colab_plan(envelope,pin,workspace,baseline_path=None):
    job=verify(envelope,pin);validate_neural_job(job);root=Path(workspace).resolve();root.mkdir(exist_ok=False)
    session='saal-'+uuid.uuid4().hex[:20];(root/'job.json').write_bytes(canonical(envelope))
    lock=Path(__file__).resolve().parents[2]/'configs/dependency-locks/linux-x86_64-py313.txt';raw=lock.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=job['dependency_lock_sha256']:raise ValueError('dependency lock substitution')
    (root/'gpu-lock.txt').write_bytes(raw)
    with tempfile.TemporaryDirectory() as temp:
        stage=Path(temp);stage_worker(stage)
        with zipfile.ZipFile(root/'worker.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(stage.rglob('*')):
                if p.is_file():archive.write(p,p.relative_to(stage).as_posix())
    installer='''import hashlib,json,shutil,subprocess,sys,threading,time
from pathlib import Path
done=threading.Event()
def heartbeat():
    while not done.wait(5):print(json.dumps({'monitor':'dependency_heartbeat','at':time.time()}),flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
try:
    if sys.version_info[:2]!=(3,13):raise ValueError('GPU lock requires Python 3.13')
    lock=Path('/content/saal_gpu_lock.txt')
    if hashlib.sha256(lock.read_bytes()).hexdigest()!=LOCK_DIGEST:raise ValueError('lock digest')
    if not shutil.which('uv'):raise ValueError('reviewed uv installer unavailable')
    subprocess.run(['uv','venv','--python',sys.executable,'/content/saal-env'],check=True,timeout=30)
    subprocess.run(['uv','pip','install','--python','/content/saal-env/bin/python','--require-hashes','-r',str(lock)],check=True,timeout=INSTALL_TIMEOUT)
finally:done.set()
print(json.dumps({'monitor':'dependency_complete','hash_lock_enforced':True}),flush=True)
'''.replace('LOCK_DIGEST',repr(job['dependency_lock_sha256'])).replace('INSTALL_TIMEOUT',str(INSTALL_SECONDS))
    (root/'install.py').write_text(installer)
    names=['worker.py','saal_rsi/__init__.py',*('saal_rsi/'+n for n in worker_sources())]
    runner='''import json,subprocess,zipfile
from pathlib import Path
with zipfile.ZipFile('/content/saal_worker.zip') as z:
    if set(z.namelist())!=set(WORKER_NAMES):raise ValueError('worker source inventory')
    z.extractall('/content/saal-neural')
command=['/content/saal-env/bin/python','-I','/content/saal-neural/worker.py','--job','/content/saal_job.json','--public-key',PIN,'--out','/content/saal_result','--require-gpu',GPU]
BASELINE_ARGUMENT
proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
for line in iter(proc.stdout.readline,b''):print(line.decode(errors='replace'),end='',flush=True)
if proc.wait(timeout=TRAIN_TIMEOUT):raise RuntimeError('training child failed')
with zipfile.ZipFile('/content/saal_result.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ARTIFACT_NAMES:z.write(Path('/content/saal_result')/name,name)
'''.replace('WORKER_NAMES',repr(names)).replace('PIN',repr(pin)).replace('GPU',repr(job['gpu'])).replace('TRAIN_TIMEOUT',str(job['max_seconds']+30)).replace('ARTIFACT_NAMES',repr(sorted(FILES)))
    baseline=job['baseline_sha256'] is not None
    if baseline:(root/'baseline-input.safetensors').write_bytes(baseline_bytes(baseline_path,job['baseline_sha256']))
    runner=runner.replace('BASELINE_ARGUMENT',"command+=['--baseline','/content/saal_baseline.safetensors']" if baseline else '')
    (root/'run.py').write_text(runner)
    commands=[['colab','new','-s',session,'--gpu',job['gpu']]];phases=['PROVISIONING']
    for local,remote in [('gpu-lock.txt','saal_gpu_lock.txt'),('job.json','saal_job.json'),('worker.zip','saal_worker.zip')]+([('baseline-input.safetensors','saal_baseline.safetensors')] if baseline else []):
        commands.append(['colab','upload','-s',session,str(root/local),'/content/'+remote]);phases.append('UPLOAD_BASELINE' if local=='baseline-input.safetensors' else 'UPLOAD_'+local.split('.')[0].upper())
    commands+=[['colab','exec','-s',session,'-f',str(root/'install.py'),'--timeout',str(INSTALL_SECONDS+60)],
              ['colab','exec','-s',session,'-f',str(root/'run.py'),'--timeout',str(job['max_seconds']+30)],
              ['colab','download','-s',session,'/content/saal_result.zip',str(root/'result.zip')],
              ['colab','log','-s',session,'-o',str(root/'runtime-log.jsonl')],['colab','stop','-s',session]]
    phases+=['DEPENDENCIES','TRAINING','DOWNLOAD','EXPORT_LOGS']
    return {'schema':'saal-colab-plan-v3','phases':phases,'session':session,'job_digest':sha(job),'commands':commands,
            'includes_final_holdout':False,'estimated_max_seconds':job['max_seconds']+CLOUD_OVERHEAD,'billing_limit_enforced':False,'dependency_hash_lock_enforced':True}

def run_neural_colab(envelope,pin,out,workspace,baseline_path=None):
    from .model_monitor import execute_colab_plan,write_json
    from .model_backend import unpack_result,bundle_digest
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('output exists')
    job=verify(envelope,pin);plan=neural_colab_plan(envelope,pin,workspace,baseline_path);root=Path(workspace)
    write_json(root/'submission-plan.json',plan);state=execute_colab_plan(plan,root)
    try:
        state.update(phase='VERIFYING_ARTIFACTS',terminal=False);write_json(root/'monitor-state.json',state)
        unpack_result(root/'result.zip',out,names=FILES)
        validate_neural_bundle(out,job,replay=True)
        result=parse_unique_json((out/'result.json').read_bytes())
        if result['runtime']['device']!='cuda' or job['gpu'] not in result['runtime']['gpu']:raise ValueError('GPU identity')
        attach_cleanup(out,{'authority':'controller','confirmed':True,'status':'NAMED_SESSION_ABSENCE_CONFIRMED','session':plan['session'],
                           'confirmation_log_sha256':hashlib.sha256((root/'cleanup-status.log').read_bytes()).hexdigest()})
        validate_neural_bundle(out,job,replay=True)
        state.update(phase='COMPLETED',terminal=True,artifact_validation='PASSED',checkpoint_reload_prediction_parity=True,runtime=result['runtime'])
        write_json(root/'monitor-state.json',state)
        return {'backend':'colab','session':plan['session'],'artifact_digest':bundle_digest(out,job),'qualification_status':'NOT_EVALUATED','cleanup_verified':True}
    except BaseException as exc:
        state.update(phase='FAILED',terminal=True,artifact_validation='FAILED',error={'type':type(exc).__name__,'detail':str(exc)});write_json(root/'monitor-state.json',state);raise
