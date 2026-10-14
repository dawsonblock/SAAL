"""Streaming recovery for large full-tuning artifacts; no learning changes.

Each authenticated chunk is appended once and then removed. The whole ZIP
hash and native inventory must verify before acceptance. This reduces disk
usage without weakening validation or allowing training retries.
"""
from pathlib import Path
import hashlib,json,sys,uuid,zipfile,subprocess,os,time
from .spec import canonical,parse_unique_json,sha
from .epoch_contract import SOURCES,validate_job
from .pretrained_contract import registry_path
from .model_contract import verify
from .model_monitor import execute_colab_plan,write_json,stream_command
from .epoch_artifacts import FILES,validate_bundle
from .pretrained_registry import file_hash

TRANSFER_SECONDS=900
STAGE_OVERHEAD=1400
CHUNK_BYTES=32_000_000

def run_remote_stage(command,log,timeout):
    # Some CLI versions report a remote Python traceback without a nonzero CLI
    # exit. Preserve explicit child failure rather than misclassify it as a
    # later missing-download error.
    failures=[];ready=[]
    def progress(value):
        print(json.dumps(value,allow_nan=False),flush=True)
        if value.get('monitor')=='worker_failure':failures.append(value)
        if value.get('monitor')=='artifact_transfer_ready':ready.append(value['manifest'])
    result=stream_command(command,log,timeout,progress)
    if failures:raise RuntimeError('Remote worker failed: '+json.dumps(failures[-1],allow_nan=False))
    if len(ready)!=1:raise RuntimeError('missing/duplicate authenticated export acknowledgement')
    return validate_transfer(ready[0])

def validate_transfer(manifest):
    if type(manifest)is not dict or set(manifest)!={'schema','job_digest','size','sha256','chunks'} or manifest['schema']!='saal-artifact-transfer-v1':raise ValueError('chunk transfer schema')
    if type(manifest['size'])is not int or not 1<=manifest['size']<=2_050_000_000 or type(manifest['chunks'])is not list or not 1<=len(manifest['chunks'])<=65:raise ValueError('transfer budget')
    import re
    if not re.fullmatch('[0-9a-f]{64}',manifest['sha256']):raise ValueError('archive hash')
    total=0
    for i,c in enumerate(manifest['chunks']):
        if type(c)is not dict or set(c)!={'name','size','sha256'} or c['name']!=f'part-{i:03d}.bin' or type(c['size'])is not int or not 1<=c['size']<=CHUNK_BYTES or not re.fullmatch('[0-9a-f]{64}',c['sha256']):raise ValueError('chunk identity')
        total+=c['size']
    if total!=manifest['size']:raise ValueError('chunk total')
    return manifest

def assemble_transfer(parts,manifest,out):
    validate_transfer(manifest);parts=Path(parts);out=Path(out)
    with out.open('xb') as target:
        for c in manifest['chunks']:
            p=parts/c['name']
            if p.is_symlink() or not p.is_file() or p.stat().st_size!=c['size'] or file_hash(p)!=c['sha256']:raise ValueError('cross-job/corrupted transfer chunk')
            with p.open('rb') as source:
                for b in iter(lambda:source.read(1024*1024),b''):target.write(b)
    if file_hash(out)!=manifest['sha256']:raise ValueError('assembled archive substitution')


def stage_worker(root):
    root=Path(root);package=root/'src/saal_rsi';package.mkdir(parents=True);(package/'__init__.py').write_text('')
    for n in SOURCES:(package/n).write_bytes(Path(__file__).with_name(n).read_bytes())
    configs=root/'configs/pretrained';configs.mkdir(parents=True)
    for n in ('registry.json','tensor_inventories.json'):(configs/n).write_bytes(registry_path().with_name(n).read_bytes())
    (root/'worker.py').write_text("from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/'src'))\nfrom saal_rsi.epoch_worker import main\nmain()\n")

def unpack_result(archive,out):
    out=Path(out)
    if out.exists():raise FileExistsError('artifact destination exists')
    with zipfile.ZipFile(archive) as z:
        if len(z.infolist())!=len(FILES) or set(z.namelist())!=FILES or sum(p.file_size for p in z.infolist())>2_050_000_000:raise ValueError('result ZIP inventory/bounds')
        if any(p.is_dir() or (p.external_attr>>16)&0o170000==0o120000 for p in z.infolist()) or z.testzip():raise ValueError('unsafe result archive')
        out.mkdir();z.extractall(out)

def setup_plan(root,max_seconds):
    root=Path(root).resolve();root.mkdir(exist_ok=False);session='saal-'+uuid.uuid4().hex[:20]
    stage=root/'stage';stage_worker(stage)
    with zipfile.ZipFile(root/'worker.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(stage).as_posix())
    lock=Path(__file__).resolve().parents[2]/'configs/dependency-locks/pretrained-linux-x86_64-py313.txt';(root/'gpu-lock.txt').write_bytes(lock.read_bytes())
    installer="""import sys,subprocess,hashlib,zipfile,json,threading,time
from pathlib import Path
stop=threading.Event()
def heart():
 while not stop.wait(5):print(json.dumps({'monitor':'dependency_heartbeat','at':time.time()}),flush=True)
threading.Thread(target=heart,daemon=True).start()
try:
 if sys.version_info[:2]!=(3,13):raise ValueError('GPU Python lock mismatch')
 lock=Path('/content/saal_pretrained_lock.txt')
 if hashlib.sha256(lock.read_bytes()).hexdigest()!=LOCK:raise ValueError('dependency hash mismatch')
 subprocess.run(['uv','venv','--python',sys.executable,'/content/saal-pretrained-env'],check=True,timeout=30)
 subprocess.run(['uv','pip','install','--python','/content/saal-pretrained-env/bin/python','--require-hashes','-r',str(lock)],check=True,timeout=900)
 with zipfile.ZipFile('/content/saal_pretrained_worker.zip') as z:z.extractall('/content/saal-pretrained')
finally:stop.set()
print(json.dumps({'monitor':'dependency_complete','hash_lock_enforced':True}),flush=True)
""".replace('LOCK',repr(hashlib.sha256(lock.read_bytes()).hexdigest()))
    # Replace the isolated placeholder only; don't alter unrelated literal text.
    (root/'install.py').write_text(installer)
    commands=[['colab','new','-s',session,'--gpu','T4'],['colab','upload','-s',session,str(root/'gpu-lock.txt'),'/content/saal_pretrained_lock.txt'],['colab','upload','-s',session,str(root/'worker.zip'),'/content/saal_pretrained_worker.zip'],['colab','exec','-s',session,'-f',str(root/'install.py'),'--timeout','960']]
    return {'schema':'saal-pretrained-colab-plan-v1','session':session,'job_digest':sha({'worker':hashlib.sha256((root/'worker.zip').read_bytes()).hexdigest(),'lock':hashlib.sha256(lock.read_bytes()).hexdigest()}),'estimated_max_seconds':max_seconds,'commands':commands,'phases':['PROVISIONING','UPLOAD_LOCK','UPLOAD_WORKER','DEPENDENCIES'],'external_deployment_allowed':False}


def download_read(command,log,deadline,expected):
    """Retry only immutable reads, never provision/train/alter a job."""
    if command[:2]!=['colab','download']:raise ValueError('read-only retry command')
    destination=Path(command[-1])
    if destination.exists():raise FileExistsError('preserve earlier transport evidence')
    last=None
    for attempt in range(1,6):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('bounded transfer deadline')
        partial=destination.with_name(destination.name+f'.attempt-{attempt}')
        try:
            if partial.exists():raise FileExistsError('earlier read attempt')
            stream_command(command[:-1]+[str(partial)],Path(str(log)+f'.attempt-{attempt}'),min(60,remaining))
            if partial.is_symlink() or not partial.is_file() or partial.stat().st_size!=expected['size'] or file_hash(partial)!=expected['sha256']:raise ValueError('transfer read authentication failed')
            partial.replace(destination);return
        except (RuntimeError,TimeoutError,ValueError) as exc:
            last=exc
            if partial.exists() and not partial.is_symlink():partial.unlink()
            print(json.dumps({'monitor':'artifact_read_retry','attempt':attempt,'max_attempts':5,'error_type':type(exc).__name__,'detail':str(exc)}),flush=True)
            if attempt<5:time.sleep(min(3,max(0,deadline-time.monotonic())))
    raise RuntimeError('immutable artifact reads exhausted') from last

class StreamingArchive:
    """One-copy assembly, with independent per-chunk and aggregate checksums."""
    def __init__(self,path,manifest):
        self.manifest=validate_transfer(manifest);self.path=Path(path)
        self.file=self.path.open('xb');self.hash=hashlib.sha256()
        self.index=0;self.written=0;self.finished=False
    def append(self,path,index):
        path=Path(path)
        if self.finished or index!=self.index or index>=len(self.manifest['chunks']):
            raise ValueError('duplicate/reordered streaming chunk')
        expected=self.manifest['chunks'][index]
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=expected['size']:
            raise ValueError('streaming chunk bounds')
        chunk_hash=hashlib.sha256();copied=0
        with path.open('rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''):
                copied+=len(block)
                if copied>expected['size']:raise ValueError('chunk grew during copy')
                chunk_hash.update(block);self.hash.update(block);self.file.write(block)
        if copied!=expected['size'] or chunk_hash.hexdigest()!=expected['sha256']:
            raise ValueError('corrupted/cross-job streaming chunk')
        self.written+=copied;self.index+=1
    def finish(self):
        if self.finished or self.index!=len(self.manifest['chunks']) or self.written!=self.manifest['size'] or self.hash.hexdigest()!=self.manifest['sha256']:
            raise ValueError('incomplete/substituted streaming archive')
        self.file.flush();os.fsync(self.file.fileno())
        if file_hash(self.path)!=self.manifest['sha256']:
            raise ValueError('written streaming archive changed')
        self.finished=True
    def close(self):
        self.file.close()

def coordinator(config_path):
    from .epoch_registry import EpochExperimentRegistry
    config=parse_unique_json(Path(config_path).read_bytes());root=Path(config['workspace']);session=config['session'];pin=config['pin'];outcomes=[]
    ledger=EpochExperimentRegistry(config['ledger'],pin)
    for i,job_path in enumerate(config['jobs']):
        envelope=parse_unique_json(Path(job_path).read_bytes());job=verify(envelope,pin);validate_job(job);digest=sha(job);name=f'run-{i:02d}-{digest[7:19]}';out=Path(config['artifacts'])/name
        # Jobs are anchored/reserved before session provisioning. Start exactly once.
        ledger.start(digest);completed=False
        try:
            if job['dependency_lock_sha256']!=hashlib.sha256((root/'gpu-lock.txt').read_bytes()).hexdigest():raise ValueError('admitted lock substitution')
            print(json.dumps({'monitor':'campaign_progress','completed_runs':i,'total_runs':len(config['jobs']),'seed':job['seed'],'mode':job['model_spec']['model_repository']+' / '+job['model_spec']['adapter_type']+' / '+job['training_policy']['sampler']}),flush=True)
            stream_command(['colab','upload','-s',session,job_path,'/content/saal_pretrained_job.json'],root/(name+'-upload.log'),60)
            remote='/content/'+name
            script=root/(name+'-run.py')
            generated_script=("import subprocess,os,zipfile,json,hashlib\nfrom pathlib import Path\nenv=dict(os.environ,SAAL_HASH_LOCK_ENFORCED='1')\n"+
                'command='+repr(['/content/saal-pretrained-env/bin/python','-I','/content/saal-pretrained/worker.py','--job','/content/saal_pretrained_job.json','--public-key',pin,'--out',remote,'--cache','/content/saal-model-cache'])+'\n'+
                "proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)\nfor line in iter(proc.stdout.readline,b''):print(line.decode(errors='replace'),end='',flush=True)\nrc=proc.wait()\nif rc:\n print(json.dumps({'monitor':'worker_failure','stage':'training','returncode':rc}),flush=True)\n raise RuntimeError('training child failed')\n"+
                "replay=subprocess.Popen(command+['--replay'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)\nfor line in iter(replay.stdout.readline,b''):print(line.decode(errors='replace'),end='',flush=True)\nrc=replay.wait()\nif rc:\n print(json.dumps({'monitor':'worker_failure','stage':'replay','returncode':rc}),flush=True)\n raise RuntimeError('fresh-process replay failed')\n"+
                f"with zipfile.ZipFile({remote+'.zip'!r},'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:\n for n in {sorted(FILES)!r}:z.write(Path({remote!r})/n,n)\n"+
                f"transfer=Path({remote+'-transfer'!r});transfer.mkdir()\nchunks=[];total=0;whole=hashlib.sha256()\nwith Path({remote+'.zip'!r}).open('rb') as source:\n i=0\n while b:=source.read({CHUNK_BYTES}):\n  name=f'part-{{i:03d}}.bin';(transfer/name).write_bytes(b);chunks.append({{'name':name,'size':len(b),'sha256':hashlib.sha256(b).hexdigest()}});whole.update(b);total+=len(b);i+=1\n(transfer/'manifest.json').write_text(json.dumps({{'schema':'saal-artifact-transfer-v1','job_digest':{digest!r},'size':total,'sha256':whole.hexdigest(),'chunks':chunks}}))\n")
            generated_script+='\nprint(json.dumps({"monitor":"artifact_transfer_ready","manifest":json.loads((transfer/"manifest.json").read_text())}),flush=True)\n' 
            guarded="import json,traceback,shutil\ntry:\n"+'\n'.join(' '+line for line in generated_script.splitlines())+"\nexcept BaseException as exc:\n print(json.dumps({'monitor':'worker_failure','stage':'export_or_child','error_type':type(exc).__name__,'detail':str(exc)}),flush=True)\n traceback.print_exc()\n raise\n"
            script.write_text(guarded)
            manifest=run_remote_stage(['colab','exec','-s',session,'-f',str(script),'--timeout',str(job['max_seconds']+300)],root/(name+'-training.log'),job['max_seconds']+330)
            transfer=root/(name+'-transfer');transfer.mkdir();deadline=time.monotonic()+TRANSFER_SECONDS
            write_json(transfer/'manifest.json',manifest)
            if manifest['job_digest']!=digest:raise ValueError('cross-job transfer manifest')
            archive=root/(name+'.zip');sink=StreamingArchive(archive,manifest)
            try:
                for k,chunk in enumerate(manifest['chunks']):
                    download_read(['colab','download','-s',session,remote+'-transfer/'+chunk['name'],str(transfer/chunk['name'])],root/(name+f'-chunk-{k:03d}.log'),deadline,chunk)
                    sink.append(transfer/chunk['name'],k)
                    (transfer/chunk['name']).unlink() # only validated task-owned redundant bytes
                    print(json.dumps({'monitor':'artifact_transfer_progress','chunk':k+1,'total_chunks':len(manifest['chunks']),'verified_bytes':sink.written}),flush=True)
                sink.finish()
            finally:
                sink.close()
            unpack_result(archive,out);result=validate_bundle(out,job,pin)
            archive.unlink() # validated bundle and manifest/receipts are retained
            cleanup_script=root/(name+'-remote-cleanup.py')
            cleanup_script.write_text('import shutil\nfrom pathlib import Path\n'+f'shutil.rmtree({remote!r})\nshutil.rmtree({remote+"-transfer"!r})\nPath({remote+".zip"!r}).unlink()\n')
            stream_command(['colab','exec','-s',session,'-f',str(cleanup_script),'--timeout','30'],root/(name+'-remote-cleanup.log'),40)
            completed=True
            outcome={'run':name,'job_digest':digest,'model_repository':job['model_spec']['model_repository'],'mode':job['model_spec']['model_repository']+' / '+job['model_spec']['adapter_type'],'seed':job['seed'],'state':'COMPLETED','artifacts':str(out),'result_digest':sha(result),'fresh_process_replay':True,'result':result}
        except BaseException as exc:
            outcome={'run':name,'job_digest':digest,'model_repository':job['model_spec']['model_repository'],'mode':job['model_spec']['model_repository']+' / '+job['model_spec']['adapter_type'],'seed':job['seed'],'state':'INTERRUPTED','error':{'type':type(exc).__name__,'detail':str(exc)}}
            # Record a failed trial; never automatically rerun or change its config.
        finally:
            ledger.finish(digest,outcome,completed);outcomes.append(outcome);write_json(root/'outcomes.json',{'runs':outcomes});write_json(root/'registry-export.json',ledger.records())
        if not completed:break
    ledger.close()
    print(json.dumps({'monitor':'campaign_progress','completed_runs':sum(o['state']=='COMPLETED' for o in outcomes),'total_runs':len(config['jobs']),'seed':None,'mode':'TERMINAL'}),flush=True)
    if len(outcomes)!=len(config['jobs']) or any(o['state']!='COMPLETED' for o in outcomes):raise RuntimeError('campaign interrupted; inspect durable negative outcome')

def execute_campaign(root,jobs,pin,artifacts,ledger,max_seconds=10000):
    required=1200+sum(j['payload']['max_seconds']+STAGE_OVERHEAD for j in jobs)
    if max_seconds<required:raise ValueError('campaign deadline smaller than registered stages')
    root=Path(root);plan=setup_plan(root,max_seconds);Path(artifacts).mkdir(parents=True,exist_ok=True)
    from .epoch_registry import EpochExperimentRegistry
    reg=EpochExperimentRegistry(ledger,pin,max_jobs=len(jobs),max_seconds=sum(j['payload']['max_seconds']+STAGE_OVERHEAD for j in jobs))
    paths=[]
    for i,e in enumerate(jobs):
        verify(e,pin);reg.reserve(e);p=root/f'job-{i:02d}.json';p.write_bytes(canonical(e));paths.append(str(p))
    reg.close();config={'workspace':str(root.resolve()),'session':plan['session'],'pin':pin,'jobs':paths,'artifacts':str(Path(artifacts).resolve()),'ledger':str(Path(ledger).resolve())};write_json(root/'coordinator.json',config)
    # Independent watchdog supervises the local coordinator and exact remote session.
    launcher=root/'coordinator.py';launcher.write_text("import sys\nsys.path.insert(0,"+repr(str(Path(__file__).resolve().parents[1]))+")\nfrom saal_rsi.epoch_backend_v3 import coordinator\ncoordinator("+repr(str((root/'coordinator.json').resolve()))+")\n")
    plan['commands']+=[[sys.executable,'-I',str(launcher)],['colab','stop','-s',plan['session']]];plan['phases']+=['CAMPAIGN'];write_json(root/'submission-plan.json',plan)
    try:
        return execute_colab_plan(plan,root)
    finally:
        state=parse_unique_json((root/'monitor-state.json').read_bytes())
        write_json(root/'cleanup-evidence.json',{'session':plan['session'],'confirmed':state['cleanup_verified'],'confirmation_sha256':hashlib.sha256((root/'cleanup-status.log').read_bytes()).hexdigest() if (root/'cleanup-status.log').exists() else None})

def run_local(envelope,pin,out,cache,max_seconds=None):
    import tempfile
    job=verify(envelope,pin);validate_job(job)
    if job['model_spec']['compute_backend']!='local':raise ValueError('local backend identity')
    with tempfile.TemporaryDirectory(prefix='saal-pretrained-') as t:
        root=Path(t);stage_worker(root);(root/'job.json').write_bytes(canonical(envelope))
        args=[sys.executable,'-I',str(root/'worker.py'),'--job',str(root/'job.json'),'--public-key',pin,'--out',str(Path(out).resolve()),'--cache',str(Path(cache).resolve())]
        subprocess.run(args,check=True,timeout=job['max_seconds']+30);subprocess.run(args+['--replay'],check=True,timeout=job['max_seconds']+30)
    return validate_bundle(out,job,pin)

