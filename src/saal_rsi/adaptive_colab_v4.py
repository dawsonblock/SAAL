"""One-use, A100/high-RAM v4 Colab engineering campaign runner.

This supervises a bounded public-data diagnostic only. It is neither a secure
hostile-code sandbox nor an independent evaluator or rollback-resistant ledger.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

from .adaptive_campaign import verify_colab_cli_version
from .adaptive_campaign_v4 import replay_command, stage_worker
from .adaptive_compute_budget_v4 import assess_budget, parse_usage
from .adaptive_contract_v4 import SOURCES as WORKER_SOURCES
from .adaptive_evaluator_v4 import validate_jobs, validate_protocol, validate_sources
from .adaptive_registry_v4 import AdaptiveRegistryV4
from .adaptive_runtime_v4 import colab_create_command
from .model_contract import verify
from .model_monitor import stream_command, write_json
from .model_monitor_v3 import execute_colab_plan
from .pretrained_contract import registry_path
from .spec import canonical, parse_unique_json, sha

REPLAY_FILE = 'fresh_process_replay.json'
MAX_ARCHIVE_BYTES = 250_000_000


def _colab_usage():
    reply = subprocess.run(['colab', 'usage'], stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=20)
    if reply.returncode:
        raise RuntimeError('Colab usage query failed: '+reply.stderr[-500:])
    return parse_usage(reply.stdout)


def _safe_unpack(archive, destination, job, pin):
    from .adaptive_artifacts_v4 import FILES, validate_bundle
    archive, destination = Path(archive), Path(destination)
    expected = FILES | {'artifact_inventory.json', REPLAY_FILE}
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('preserve existing v4 output; no overwrite')
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        if (source.testzip() is not None or len(infos) != len(expected)
                or set(source.namelist()) != expected
                or sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES
                or any(i.is_dir() or (i.external_attr >> 16) & 0o170000 == 0o120000
                       or Path(i.filename).is_absolute() or '..' in Path(i.filename).parts
                       for i in infos)):
            raise ValueError('unsafe or incomplete v4 Colab artifact archive')
        destination.mkdir(parents=True, exist_ok=False)
        source.extractall(destination)
    validate_bundle(destination, job, pin, require_replay=True)


def _remote_installer(lock_digest, worker_digest):
    source = r'''import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
lock=Path('/content/saal_v4_lock.txt'); worker=Path('/content/saal_v4_worker.zip')
if sys.version_info[:2] != (3,13): raise RuntimeError('admitted Linux Python 3.13 required')
if hashlib.sha256(lock.read_bytes()).hexdigest() != LOCK_DIGEST: raise RuntimeError('dependency lock substitution')
if hashlib.sha256(worker.read_bytes()).hexdigest() != WORKER_DIGEST: raise RuntimeError('worker archive substitution')
with zipfile.ZipFile(worker) as z:
 names=z.namelist()
 if z.testzip() is not None or len(names) != len(set(names)) or any(n.startswith('/') or '..' in Path(n).parts for n in names): raise RuntimeError('worker archive safety')
 subprocess.run(['uv','venv','--python',sys.executable,'/content/saal-v4-env'],check=True,timeout=60)
 subprocess.run(['uv','pip','install','--python','/content/saal-v4-env/bin/python','--require-hashes','--only-binary',':all:','-r',str(lock)],check=True,timeout=1200)
 with zipfile.ZipFile(worker) as z: z.extractall('/content/saal-v4-worker')
print(json.dumps({'monitor':'dependency_complete','hash_lock_enforced':True,'worker_sha256':WORKER_DIGEST}),flush=True)
'''
    return source.replace('LOCK_DIGEST', repr(lock_digest)).replace('WORKER_DIGEST', repr(worker_digest))


def _hardware_probe(requirements):
    source = r'''import json,subprocess,sys
code="""import json,torch,sys
sys.path.insert(0,'/content/saal-v4-worker/src')
from saal_rsi.adaptive_runtime_v4 import host_memory_bytes,validate_runtime_observation
req=REQ
if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
free,total=torch.cuda.mem_get_info();p=torch.cuda.get_device_properties(0);name=torch.cuda.get_device_name(0)
obs={'device':'cuda:0','gpu':name,'accelerator':req['accelerator'],'gpu_memory_bytes':int(p.total_memory),'free_gpu_bytes_preflight':int(free),'host_memory_bytes':host_memory_bytes(),'high_memory_observed':host_memory_bytes() >= 32*1024**3,'dependency_hash_lock_enforced':True}
validate_runtime_observation(req,obs)
print(json.dumps({'monitor':'hardware_preflight','observation':obs},flush=True))"""
proc=subprocess.run(['/content/saal-v4-env/bin/python','-I','-c',code],check=True,capture_output=True,text=True,timeout=60)
for line in proc.stdout.splitlines(): print(line,flush=True)
'''
    return source.replace('REQ', repr(requirements))


def _job_remote_script(job_digest, pin, output_dir, remote_job):
    from .adaptive_artifacts_v4 import FILES
    expected = sorted(FILES | {'artifact_inventory.json', REPLAY_FILE})
    command = ['/content/saal-v4-env/bin/python', '-I',
        '/content/saal-v4-worker/worker.py', '--job', remote_job,
        '--public-key', pin, '--out', output_dir, '--cache', '/content/saal-v4-model-cache']
    lines = [
        'import hashlib,json,os,subprocess,traceback,zipfile',
        'from pathlib import Path', f'command={command!r}',
        "env=dict(os.environ,SAAL_HASH_LOCK_ENFORCED='1')", 'try:',
        '    for suffix in ([], ["--replay"]):',
        '        p=subprocess.Popen(command+suffix,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)',
        '        for line in iter(p.stdout.readline,b""):',
        '            print(line.decode(errors="replace"),end="",flush=True)',
        '        rc=p.wait()',
        '        if rc: raise RuntimeError("v4 training or remote fresh-process replay failed: "+str(rc))',
        f'    files={expected!r}', f'    archive={output_dir+".zip"!r}',
        '    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:',
        f'        for n in files: z.write(Path({output_dir!r})/n,n)',
        '    size=Path(archive).stat().st_size',
        '    if not 1<=size<=250000000: raise ValueError("v4 artifact archive bound")',
        '    digest=hashlib.sha256(Path(archive).read_bytes()).hexdigest()',
        f'    print(json.dumps({{"monitor":"artifact_transfer_ready","manifest":{{"schema":"saal-v4-artifact-transfer-v1","job_digest":{job_digest!r},"size":size,"sha256":digest,"files":files}}}}),flush=True)',
        'except BaseException as exc:',
        '    print(json.dumps({"monitor":"worker_failure","stage":"training_replay_export","error_type":type(exc).__name__,"detail":str(exc)}),flush=True)',
        '    traceback.print_exc()', '    raise',
    ]
    return '\n'.join(lines)+'\n'


def _parse_ready(output, job_digest):
    rows = []
    for line in output.splitlines():
        offset = line.find('{"monitor"')
        if offset >= 0:
            try:
                record = json.loads(line[offset:])
            except ValueError:
                continue
            rows.append(record.get('manifest'))
    if len(rows) != 1:
        raise RuntimeError('missing or duplicate v4 remote artifact receipt')
    manifest = rows[0]
    if (type(manifest) is not dict or set(manifest) != {'schema','job_digest','size','sha256','files'}
            or manifest['schema'] != 'saal-v4-artifact-transfer-v1'
            or manifest['job_digest'] != job_digest or type(manifest['size']) is not int
            or not 1 <= manifest['size'] <= MAX_ARCHIVE_BYTES
            or type(manifest['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}',manifest['sha256'])):
        raise ValueError('v4 remote transfer receipt does not bind the admitted job')
    from .adaptive_artifacts_v4 import FILES
    if set(manifest['files']) != FILES | {'artifact_inventory.json',REPLAY_FILE}:
        raise ValueError('remote artifact receipt has an unexpected member inventory')
    return manifest


def _campaign_coordinator(config_path):
    """Called only after the exact runtime, lock and staged worker are ready."""
    from .adaptive_artifacts_v4 import validate_bundle
    from .adaptive_campaign_v4 import replay_command
    from .adaptive_evaluator_v4 import validate_jobs
    from .adaptive_registry_v4 import AdaptiveRegistryV4
    from .model_contract import verify
    config = parse_unique_json(Path(config_path).read_bytes())
    root, session, pin = Path(config['workspace']), config['session'], config['pin']
    protocol_envelope = parse_unique_json(Path(config['protocol']).read_bytes())
    protocol = verify(protocol_envelope, pin)
    envelopes = [parse_unique_json(Path(p).read_bytes()) for p in config['jobs']]
    jobs = validate_jobs(protocol, envelopes, pin)
    ledger = AdaptiveRegistryV4(config['ledger'],pin,protocol_envelope,envelopes)
    outcomes=[]
    try:
        for index,(job,envelope) in enumerate(zip(jobs,envelopes)):
            digest=sha(job); descriptor=protocol['runs'][index]
            name=descriptor['name']; destination=Path(config['artifacts'])/name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError('preserve prior v4 artifact path')
            ledger.start(digest)
            completed=False
            started=time.monotonic()
            try:
                remote_job='/content/'+name+'-job.json'
                stream_command(['colab','upload','-s',session,config['jobs'][index],remote_job],
                    root/(name+'-upload.log'),90)
                remote_out='/content/'+name
                script=root/(name+'-remote.py')
                script.write_text(_job_remote_script(digest,pin,remote_out,remote_job))
                ready=[];failures=[]
                def remote_event(record):
                    print(json.dumps(record,allow_nan=False),flush=True)
                    if record.get('monitor')=='artifact_transfer_ready': ready.append(record['manifest'])
                    if record.get('monitor')=='worker_failure': failures.append(record)
                timeout=job['max_seconds']+900
                output=stream_command(['colab','exec','-s',session,'-f',str(script),
                    '--timeout',str(timeout)],root/(name+'-remote.log'),timeout+60,remote_event)
                if failures or len(ready)!=1:
                    raise RuntimeError('remote training/replay/export reported failure or missing receipt')
                manifest=_parse_ready(output,digest)
                local_archive=root/(name+'.zip')
                stream_command(['colab','download','-s',session,remote_out+'.zip',str(local_archive)],
                    root/(name+'-download.log'),600)
                if (local_archive.is_symlink() or not local_archive.is_file()
                        or local_archive.stat().st_size!=manifest['size']
                        or hashlib.sha256(local_archive.read_bytes()).hexdigest()!=manifest['sha256']):
                    raise ValueError('downloaded v4 artifact archive hash/size mismatch')
                _safe_unpack(local_archive,destination,job,pin)
                validate_bundle(destination,job,pin,require_replay=True)
                replay_receipt=root/(name+'-local-replay.json')
                replay_command({'workspace':str(root),'pin':pin,'cache':config['cache']},
                    destination,Path(config['jobs'][index]),replay_receipt)
                local_replay=parse_unique_json(replay_receipt.read_bytes())
                if local_replay.get('verified') is not True:
                    raise ValueError('fresh local CPU replay did not reproduce all checkpoints')
                artifact_result=validate_bundle(destination,job,pin,require_replay=True)
                if type(artifact_result.get('snapshots')) is not dict:
                    raise ValueError('validated v4 result snapshots absent')
                completed=True
                outcome={'run':name,'job_digest':digest,'method':job['adapter']['method'],
                    'seed':job['seed'],'state':'COMPLETED','artifact_dir':str(destination),
                    'artifact_archive_sha256':manifest['sha256'],'artifact_bytes':manifest['size'],
                    'local_replay_digest':sha(local_replay),'result_digest':sha(artifact_result),
                    'initial':artifact_result['snapshots']['initial']['metrics'],
                    'best':artifact_result['snapshots']['best']['metrics'],
                    'terminal':artifact_result['snapshots']['terminal']['metrics'],
                    'selected_step':artifact_result['snapshots']['best']['step'],
                    'worker_seconds':artifact_result['total_worker_seconds'],
                    'local_replay_verified':True}
                local_archive.unlink()
                cleanup=root/(name+'-remote-cleanup.py')
                cleanup.write_text('\n'.join(['import shutil','from pathlib import Path',
                    f'shutil.rmtree({remote_out!r})',
                    f'Path({(remote_out+".zip")!r}).unlink(missing_ok=True)',
                    f'Path({remote_job!r}).unlink(missing_ok=True)'])+'\n')
                try:
                    stream_command(['colab','exec','-s',session,'-f',str(cleanup),'--timeout','45'],
                        root/(name+'-remote-cleanup.log'),60)
                    outcome['artifact_cleanup']='CONFIRMED'
                except (RuntimeError,TimeoutError) as cleanup_exc:
                    # Final exact-session stop remains mandatory and will remove the runtime.
                    outcome['artifact_cleanup']='DEFERRED_TO_EXACT_SESSION_STOP'
                    outcome['artifact_cleanup_error']=type(cleanup_exc).__name__
            except BaseException as exc:
                outcome={'run':name,'job_digest':digest,'method':job['adapter']['method'],
                    'seed':job['seed'],'state':'INTERRUPTED','error':{'type':type(exc).__name__,
                    'detail':str(exc)[:1000]},'elapsed_seconds':time.monotonic()-started}
            finally:
                ledger.finish(digest,outcome,completed)
                outcomes.append(outcome)
                write_json(root/'outcomes.json',{'schema':'saal-v4-engineering-outcomes-v1',
                    'runs':outcomes,'qualification_status':'NOT_EVALUATED',
                    'external_deployment_allowed':False})
                write_json(root/'registry-export.json',ledger.records())
            print(json.dumps({'monitor':'campaign_progress','completed_runs':sum(x['state']=='COMPLETED' for x in outcomes),
                'total_runs':len(jobs),'seed':job['seed'],'mode':job['adapter']['method']},allow_nan=False),flush=True)
            if not completed:
                raise RuntimeError('v4 attempt interrupted; do not retry or replace: '+outcome['error']['detail'])
        if len(outcomes)!=len(jobs):
            raise RuntimeError('v4 campaign missing admitted run outcomes')
    finally:
        ledger.close()


class _ComputeGuard:
    def __init__(self, root, session, start_balance, cap, deadline):
        self.root=Path(root);self.session=session;self.start_balance=start_balance
        self.cap=cap;self.deadline=deadline;self.done=threading.Event()
        self.failure=None;self.thread=None;self.log=self.root/'compute-usage.jsonl'

    def start(self):
        self.thread=threading.Thread(target=self._watch,name='saal-colab-compute-guard',daemon=True)
        self.thread.start()

    def stop(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=25)
        if self.thread and self.thread.is_alive():
            self._fail('compute guard did not stop cleanly')
        try:
            usage=_colab_usage()
            estimate=assess_budget(starting_balance=self.start_balance,usage=usage,cap=self.cap,
                remaining_seconds=0,require_single_assignment=False)
            self._record({'event':'FINAL_ACCOUNT_USAGE','usage':usage,'estimate':estimate})
            if not estimate['within_signed_cap']:
                self._fail('final reported Colab spend exceeded signed compute-unit cap')
        except BaseException as exc:
            self._fail('final usage verification failed: '+str(exc))

    def _record(self, value):
        row={'at_epoch':time.time(),'session':self.session,**value}
        with self.log.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

    def _fail(self, detail):
        if self.failure is None:self.failure=detail
        self._record({'event':'BUDGET_GUARD_FAILURE','detail':detail})
        try:
            subprocess.run(['colab','stop','-s',self.session],stdin=subprocess.DEVNULL,
                capture_output=True,text=True,timeout=60)
        except BaseException as exc:
            self._record({'event':'EXACT_SESSION_STOP_ERROR','error_type':type(exc).__name__})

    def _watch(self):
        consecutive_errors=0
        while not self.done.wait(10):
            phase='UNKNOWN'
            try:
                monitor=Path(self.root/'monitor-state.json')
                phase=parse_unique_json(monitor.read_bytes()).get('phase') if monitor.exists() else 'PREPARED'
                if phase in ('CLEANUP','COMPLETED','FAILED'):
                    continue
                usage=_colab_usage();consecutive_errors=0
                active=phase not in ('PREPARED','PROVISIONING')
                estimate=assess_budget(starting_balance=self.start_balance,usage=usage,cap=self.cap,
                    remaining_seconds=max(0.0,self.deadline-time.time()),require_single_assignment=active)
                self._record({'event':'USAGE_SAMPLE','phase':phase,'usage':usage,'estimate':estimate})
            except BaseException as exc:
                consecutive_errors+=1
                self._record({'event':'USAGE_QUERY_ERROR','phase':phase if 'phase' in locals() else 'UNKNOWN',
                    'error_type':type(exc).__name__,'consecutive_errors':consecutive_errors})
                if (phase not in ('PREPARED','PROVISIONING','CLEANUP','COMPLETED','FAILED')
                        or consecutive_errors>=2):
                    self._fail('usage telemetry unavailable or cap would be exceeded: '+str(exc))
                    return


def execute(workspace, admission, artifacts, cache):
    """Execute one freshly signed v4 diagnostic with one A100/high-RAM request."""
    workspace=Path(workspace).resolve();admission=Path(admission).resolve();artifacts=Path(artifacts).resolve()
    if workspace.exists() or artifacts.exists():
        raise FileExistsError('v4 campaign output paths must be new; preserve prior evidence')
    pin=(admission/'controller.pub').read_text().strip()
    protocol_envelope=parse_unique_json((admission/'signed_protocol.json').read_bytes())
    protocol=verify(protocol_envelope,pin);validate_protocol(protocol)
    job_envelopes=[parse_unique_json(p.read_bytes()) for p in sorted((admission/'jobs').glob('*.json'))]
    jobs=validate_jobs(protocol,job_envelopes,pin)
    if len(job_envelopes)!=len(protocol['job_digests']):
        raise ValueError('signed v4 admission is missing a job envelope')
    validate_sources(protocol,Path(__file__).resolve().parents[2])
    verify_colab_cli_version(protocol['colab_cli_version'])
    before=_colab_usage()
    if before['active_assignments']!=0:
        raise RuntimeError('preflight found another active Colab assignment; no session started')
    cap=protocol['budget']['max_compute_units']
    if before['balance']<cap+5:
        raise RuntimeError('compute-unit balance is below signed cap plus reserve')
    workspace.mkdir(mode=0o700,parents=True)
    artifacts.mkdir(mode=0o700,parents=True)
    lock=registry_path().parents[1]/'dependency-locks'/'pretrained-linux-x86_64-py313.txt'
    lock_bytes=lock.read_bytes();lock_digest=hashlib.sha256(lock_bytes).hexdigest()
    if protocol['source_files'].get('configs/dependency-locks/pretrained-linux-x86_64-py313.txt')!=lock_digest:
        raise ValueError('signed Linux dependency lock digest changed')
    stage=workspace/'stage';stage_worker(stage)
    worker_zip=workspace/'worker.zip'
    with zipfile.ZipFile(worker_zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(stage.rglob('*')):
            if path.is_symlink():raise ValueError('symlink in staged v4 worker')
            if path.is_file():archive.write(path,path.relative_to(stage).as_posix())
    worker_digest=hashlib.sha256(worker_zip.read_bytes()).hexdigest()
    (workspace/'gpu-lock.txt').write_bytes(lock_bytes)
    (workspace/'install.py').write_text(_remote_installer(lock_digest,worker_digest))
    (workspace/'hardware.py').write_text(_hardware_probe(protocol['runtime_requirements']))
    session='saal-'+uuid.uuid4().hex[:20]
    job_paths=[]
    for source in sorted((admission/'jobs').glob('*.json')):
        job_paths.append(str(source))
    config={'workspace':str(workspace),'session':session,'pin':pin,
        'protocol':str(admission/'signed_protocol.json'),'jobs':job_paths,
        'ledger':str(workspace/'budget.sqlite3'),'artifacts':str(artifacts),
        'cache':str(Path(cache).resolve())}
    (workspace/'campaign-config.json').write_bytes(canonical(config))
    coordinator=workspace/'coordinator.py'
    source_root=str(Path(__file__).resolve().parents[1])
    coordinator.write_text('import sys\n'+f'sys.path.insert(0,{source_root!r})\n'+
        'from saal_rsi.adaptive_colab_v4 import _campaign_coordinator\n'+
        f'_campaign_coordinator({str(workspace/"campaign-config.json")!r})\n')
    runtime=protocol['runtime_requirements']
    commands=[colab_create_command(session,runtime),
        ['colab','upload','-s',session,str(workspace/'gpu-lock.txt'),'/content/saal_v4_lock.txt'],
        ['colab','upload','-s',session,str(worker_zip),'/content/saal_v4_worker.zip'],
        ['colab','exec','-s',session,'-f',str(workspace/'install.py'),'--timeout','1320'],
        ['colab','exec','-s',session,'-f',str(workspace/'hardware.py'),'--timeout','90'],
        [sys.executable,'-I',str(coordinator)],['colab','stop','-s',session]]
    phases=['PROVISIONING','UPLOAD_LOCK','UPLOAD_WORKER','INSTALL_DEPENDENCIES','HARDWARE_PREFLIGHT','CAMPAIGN']
    plan={'schema':'saal-adaptive-colab-plan-v4','session':session,
        'job_digest':sha(protocol),'estimated_max_seconds':protocol['budget']['total_lifecycle_seconds'],
        'commands':commands,'phases':phases,'requested_runtime':runtime,
        'compute_units_cap':cap,'external_deployment_allowed':False}
    from .model_monitor_v3 import validate_plan
    validate_plan(plan)
    write_json(workspace/'submission-plan.json',plan)
    ledger=AdaptiveRegistryV4(workspace/'budget.sqlite3',pin,protocol_envelope,job_envelopes,create=True)
    try:
        ledger.reserve_all(job_envelopes)
        write_json(workspace/'preprovision-registry.json',ledger.records())
        ledger.begin_provisioning()
    finally:
        ledger.close()
    guard=_ComputeGuard(workspace,session,before['balance'],cap,
        time.time()+protocol['budget']['total_lifecycle_seconds'])
    guard.start()
    state=None;primary=None
    try:
        state=execute_colab_plan(plan,workspace)
    except BaseException as exc:
        primary=exc
    finally:
        guard.stop()
        ledger=AdaptiveRegistryV4(workspace/'budget.sqlite3',pin,protocol_envelope,job_envelopes)
        try:
            succeeded=(primary is None and state is not None and state.get('phase')=='COMPLETED'
                and state.get('cleanup_verified') is True and guard.failure is None)
            ledger.end(succeeded)
            write_json(workspace/'registry-export.json',ledger.records())
        finally:
            ledger.close()
    outcome={'schema':'saal-v4-colab-campaign-terminal-v1','session':session,
        'campaign_state':'COMPLETED' if primary is None and state and state.get('phase')=='COMPLETED' and guard.failure is None else 'FAILED',
        'cleanup_verified':bool(state and state.get('cleanup_verified')),
        'compute_guard_failure':guard.failure,'compute_unit_cap':cap,
        'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    write_json(workspace/'campaign-terminal.json',outcome)
    if primary is not None:raise primary
    if guard.failure is not None:raise RuntimeError('v4 compute-unit guard stopped/failed campaign: '+guard.failure)
    if outcome['campaign_state']!='COMPLETED' or not outcome['cleanup_verified']:
        raise RuntimeError('v4 Colab campaign did not reach verified cleanup')
    return outcome
