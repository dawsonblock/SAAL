"""One admitted Colab lifecycle with durable failures and no training retries."""
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

from .adaptive_contract import SOURCES
from .adaptive_artifacts import FILES, ROLES, read_json, file_hash
from .adaptive_dispatch import validate_job, validate_bundle
from .adaptive_registry import AdaptiveRegistry
from .epoch_backend_v6 import (run_remote_stage, download_read, StreamingArchive,
                               cleanup_export, CHUNK_BYTES)
from .model_contract import verify
from .model_monitor import stream_command, write_json
from .model_monitor_v3 import execute_colab_plan
from .pretrained_contract import registry_path
from .spec import canonical, parse_unique_json, sha

RECOVERED_FILES = FILES | {'artifact_inventory.json', 'fresh_process_replay.json'}


def stage_worker(destination, include_v2=False):
    destination = Path(destination)
    package = destination/'src/saal_rsi'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    if include_v2:
        from .adaptive_contract_v2 import SOURCES as sources
    else:
        sources = SOURCES
    for name in sources:
        (package/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    configs = destination/'configs/pretrained'
    configs.mkdir(parents=True)
    for name in ('registry.json', 'tensor_inventories.json'):
        (configs/name).write_bytes(registry_path().with_name(name).read_bytes())
    locks = destination/'configs/dependency-locks'
    locks.mkdir(parents=True)
    for name in ('pretrained-linux-x86_64-py313.txt', 'pretrained-macos-arm64-py312.txt'):
        (locks/name).write_bytes(registry_path().parents[1].joinpath('dependency-locks', name).read_bytes())
    bootstrap = "from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/'src'))\n"
    if include_v2:
        bootstrap += ("import argparse\nfrom saal_rsi.pretrained_contract import job_bytes\nfrom saal_rsi.spec import parse_unique_json\n"
            "parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--job');known,_=parser.parse_known_args()\n"
            "schema=parse_unique_json(job_bytes(known.job))['payload']['schema'] if known.job else 'saal-adaptive-job-v1'\n"
            "if schema=='saal-adaptive-job-v2':\n from saal_rsi.adaptive_worker_v2 import main\n"
            "elif schema=='saal-adaptive-job-v1':\n from saal_rsi.adaptive_worker import main\n"
            "else:raise ValueError('unknown signed worker version')\nmain()\n")
    else:
        bootstrap += "from saal_rsi.adaptive_worker import main\nmain()\n"
    (destination/'worker.py').write_text(bootstrap)


def require_storage(path):
    import shutil
    free = shutil.disk_usage(path).free
    if free < 768*1024**2:
        raise OSError(28, 'adaptive artifact recovery requires 768 MiB working headroom')
    return {'free_bytes': free, 'required_bytes': 768*1024**2}


def unpack(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('preserve previous native artifacts')
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if (len(infos) != len(RECOVERED_FILES) or set(z.namelist()) != RECOVERED_FILES
                or sum(i.file_size for i in infos) > 250_000_000
                or any(i.is_dir() or (i.external_attr >> 16) & 0o170000 == 0o120000 for i in infos)):
            raise ValueError('adaptive export inventory / bounds')
        destination.mkdir()
        z.extractall(destination)


def fresh_local_replay(bundle, job, pin, cache):
    from .adaptive_backend import verify_peft_implementation
    if job['schema'] == 'saal-adaptive-job-v2':
        from .adaptive_replay_v2 import PortableReplayBackendV2 as PortableReplayBackend
    else:
        from .adaptive_replay import PortableReplayBackend
    from .epoch_metrics import prediction_equal
    from .learning_backend import verify_dependencies
    from .pretrained_registry import validate_snapshot
    import torch
    torch.set_num_threads(2)
    result = validate_bundle(bundle, job, pin, require_replay=True)
    verify_dependencies()
    verify_peft_implementation()
    spec = job['model_spec']
    snapshot = Path(cache)/spec['model_repository'].split('/')[-1]/spec['model_revision']
    validate_snapshot(snapshot, spec['model_repository'], spec['model_revision'])
    backend = PortableReplayBackend(job, snapshot, Path(bundle)/'initial.safetensors',
                                    read_json(Path(bundle)/'initial_rank_pattern.json')).load_pretrained()
    if backend.unadapted_model_hash != result['unadapted_model_hash'] or backend.initial_frozen_hash != result['frozen_integrity']['initial']:
        raise ValueError('fresh local backbone / head identity mismatch')
    rows = [r for r in job['dataset']['records'] if r['split'] == 'development']
    outputs = {}
    for role in ROLES:
        backend.restore_checkpoint(Path(bundle)/(role+'.safetensors'), read_json(Path(bundle)/(role+'_rank_pattern.json')))
        outputs[role] = backend.evaluate(rows)
    agreement = all(prediction_equal(outputs[r], result['snapshots'][r]['predictions'], 'modernbert-seqcls-v1', job['replay_tolerance']) for r in ROLES)
    return {'schema': 'saal-adaptive-controller-replay-v1', 'job_digest': sha(job),
            'inventory_sha256': file_hash(Path(bundle)/'artifact_inventory.json'),
            'predictions': outputs, 'verified': agreement, 'tolerance': job['replay_tolerance'],
            'runtime': 'fresh controller process; local CPU on macOS',
            'independent_administration': False, 'qualification_status': 'NOT_EVALUATED'}


def replay_command(config, bundle, job_file, destination):
    """Fresh local process, separate from both remote training and replay."""
    script = Path(config['workspace'])/(destination.stem+'-replay.py')
    root = str(Path(__file__).resolve().parents[1])
    script.write_text('import sys,json\nfrom pathlib import Path\n'+f'sys.path.insert(0,{root!r})\n'+
        'from saal_rsi.adaptive_campaign import fresh_local_replay\nfrom saal_rsi.spec import canonical\nfrom saal_rsi.model_contract import verify\n'+
        f'pin={config["pin"]!r}\njob=verify(json.loads(Path({str(job_file)!r}).read_text()),pin)\n'+
        f'value=fresh_local_replay({str(bundle)!r},job,pin,{config["cache"]!r})\n'+
        f'with Path({str(destination)!r}).open("xb") as output:output.write(canonical(value))\n'+
        'if not value["verified"]:raise ValueError("CPU/GPU numerical replay failed at unchanged tolerance")\n')
    stream_command([sys.executable, '-I', str(script)], destination.with_suffix('.log'), 600)


def coordinator(config_path):
    config = read_json(config_path)
    root = Path(config['workspace'])
    protocol = read_json(config['protocol'])
    ledger = AdaptiveRegistry(config['ledger'], config['pin'], protocol)
    session, pin = config['session'], config['pin']
    outcomes = []
    try:
        for index, filename in enumerate(config['jobs']):
            envelope = read_json(filename)
            job = validate_job(verify(envelope, pin))
            digest = sha(job)
            name = f'run-{index:02d}-{digest[7:19]}'
            out = Path(config['artifacts'])/name
            completed = False
            ledger.start(digest)
            try:
                write_json(root/(name+'-storage.json'), require_storage(root))
                stream_command(['colab', 'upload', '-s', session, filename, '/content/saal_adaptive_job.json'], root/(name+'-upload.log'), 60)
                remote = '/content/'+name
                command = ['/content/saal-adaptive-env/bin/python', '-I', '/content/saal-adaptive/worker.py',
                           '--job', '/content/saal_adaptive_job.json', '--public-key', pin,
                           '--out', remote, '--cache', '/content/saal-model-cache']
                script = root/(name+'-run.py')
                script.write_text('import subprocess,os,json,zipfile,hashlib,traceback\nfrom pathlib import Path\ntry:\n'+
                    ' env=dict(os.environ,SAAL_HASH_LOCK_ENFORCED="1")\n'+
                    f' commands={[command, command+["--replay"]]!r}\n'+
                    ' for command in commands:\n  p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)\n  for line in iter(p.stdout.readline,b""):print(line.decode(errors="replace"),end="",flush=True)\n  if p.wait():raise RuntimeError("adaptive training or fresh replay child failed")\n'+
                    f' with zipfile.ZipFile({remote+".zip"!r},"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:\n  for n in {sorted(RECOVERED_FILES)!r}:z.write(Path({remote!r})/n,n)\n'+
                    f' transfer=Path({remote+"-transfer"!r});transfer.mkdir()\n chunks=[];total=0;whole=hashlib.sha256()\n'+
                    f' with Path({remote+".zip"!r}).open("rb") as source:\n  i=0\n  while b:=source.read({CHUNK_BYTES}):\n   n=f"part-{{i:03d}}.bin";(transfer/n).write_bytes(b);chunks.append({{"name":n,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()}});whole.update(b);total+=len(b);i+=1\n'+
                    f' print(json.dumps({{"monitor":"artifact_transfer_ready","manifest":{{"schema":"saal-artifact-transfer-v1","job_digest":{digest!r},"size":total,"sha256":whole.hexdigest(),"chunks":chunks}}}}),flush=True)\n'+
                    'except BaseException as exc:\n print(json.dumps({"monitor":"worker_failure","stage":"adaptive_train_replay_export","error_type":type(exc).__name__,"detail":str(exc)}),flush=True)\n traceback.print_exc()\n raise\n')
                manifest = run_remote_stage(['colab', 'exec', '-s', session, '-f', str(script), '--timeout', str(job['max_seconds']+300)], root/(name+'-training.log'), job['max_seconds']+330)
                if manifest['job_digest'] != digest:
                    raise ValueError('cross-job artifact transfer')
                write_json(root/(name+'-transfer.json'), manifest)
                archive = root/(name+'.zip')
                sink = StreamingArchive(archive, manifest)
                try:
                    deadline = time.monotonic()+600
                    for k, chunk in enumerate(manifest['chunks']):
                        part = root/(name+'-'+chunk['name'])
                        download_read(['colab', 'download', '-s', session, remote+'-transfer/'+chunk['name'], str(part)], root/(name+f'-download-{k}.log'), deadline, chunk)
                        sink.append(part, k)
                        part.unlink()  # validated redundant task-owned transport bytes
                    sink.finish()
                finally:
                    sink.close()
                unpack(archive, out)
                native = validate_bundle(out, job, pin, require_replay=True)
                archive.unlink()
                receipt = root/(name+'-controller-replay.json')
                replay_command(config, out, filename, receipt)
                local = read_json(receipt)
                if local['verified'] is not True:
                    raise ValueError('independent process replay failed')
                cleanup = root/(name+'-remove-export.py')
                cleanup.write_text('import shutil\nfrom pathlib import Path\n'+f'shutil.rmtree({remote!r})\nshutil.rmtree({remote+"-transfer"!r})\nPath({remote+".zip"!r}).unlink()\n')
                cleanup_result = cleanup_export(['colab', 'exec', '-s', session, '-f', str(cleanup), '--timeout', '30'], root/(name+'-export-cleanup.log'), 40)
                completed = True
                outcome = {'run': name, 'job_digest': digest, 'state': 'COMPLETED', 'artifacts': str(out),
                           'method': job['adapter']['method'], 'seed': job['seed'],
                           'result_digest': sha(native), 'unadapted_model_hash': native['unadapted_model_hash'],
                           'controller_replay': str(receipt), 'controller_replay_digest': sha(local),
                           'artifact_cleanup': cleanup_result}
            except BaseException as exc:
                outcome = {'run': name, 'job_digest': digest, 'state': 'INTERRUPTED',
                           'method': job['adapter']['method'], 'seed': job['seed'],
                           'error': {'type': type(exc).__name__, 'detail': str(exc)}}
            finally:
                ledger.finish(digest, outcome, completed)
                outcomes.append(outcome)
                write_json(root/'outcomes.json', {'runs': outcomes})
                write_json(root/'registry-export.json', ledger.records())
            if not completed:
                raise RuntimeError('adaptive campaign interrupted; no retry: '+str(outcome['error']))
            print(json.dumps({'monitor': 'campaign_progress', 'completed_runs': len(outcomes),
                              'total_runs': len(config['jobs']), 'seed': job['seed'], 'mode': job['adapter']['method']}), flush=True)
    finally:
        ledger.close()


def execute(workspace, job_paths, protocol_path, ledger_path, pin, artifacts, cache):
    protocol_envelope = read_json(protocol_path)
    protocol = verify(protocol_envelope, pin)
    jobs = [validate_job(verify(read_json(path), pin)) for path in job_paths]
    if [sha(j) for j in jobs] != protocol['job_digests']:
        raise ValueError('signed campaign / job order substitution')
    root = Path(workspace).resolve()
    root.mkdir(exist_ok=False)
    require_storage(root)
    Path(artifacts).mkdir(exist_ok=False)
    session = 'saal-'+uuid.uuid4().hex[:20]
    stage_worker(root/'stage', include_v2=any(j['schema'] == 'saal-adaptive-job-v2' for j in jobs))
    with zipfile.ZipFile(root/'worker.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root/'stage').rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(root/'stage').as_posix())
    lock = registry_path().parents[1]/'dependency-locks/pretrained-linux-x86_64-py313.txt'
    (root/'gpu-lock.txt').write_bytes(lock.read_bytes())
    installer = root/'install.py'
    installer.write_text('import sys,subprocess,hashlib,zipfile,json,threading,time\nfrom pathlib import Path\n'+
        'stop=threading.Event()\ndef heart():\n while not stop.wait(5):print(json.dumps({"monitor":"dependency_heartbeat"}),flush=True)\nthreading.Thread(target=heart,daemon=True).start()\ntry:\n'+
        ' if sys.version_info[:2]!=(3,13):raise ValueError("registered GPU Python lock mismatch")\n'+
        f' lock=Path("/content/saal_adaptive_lock.txt")\n if hashlib.sha256(lock.read_bytes()).hexdigest()!={file_hash(lock)!r}:raise ValueError("dependency lock substitution")\n'+
        ' subprocess.run(["uv","venv","--python",sys.executable,"/content/saal-adaptive-env"],check=True,timeout=30)\n subprocess.run(["uv","pip","install","--python","/content/saal-adaptive-env/bin/python","--require-hashes","--only-binary",":all:","-r",str(lock)],check=True,timeout=900)\n'+
        f' archive=Path("/content/saal_adaptive_worker.zip")\n if hashlib.sha256(archive.read_bytes()).hexdigest()!={file_hash(root/"worker.zip")!r}:raise ValueError("worker archive substitution")\n'+
        ' with zipfile.ZipFile(archive) as z:z.extractall("/content/saal-adaptive")\n print(json.dumps({"monitor":"dependency_complete","hash_lock_enforced":True}),flush=True)\nexcept BaseException as exc:\n print(json.dumps({"monitor":"worker_failure","stage":"dependencies","detail":str(exc)}),flush=True)\n raise\nfinally:stop.set()\n')
    config = {'workspace': str(root), 'session': session, 'pin': pin, 'protocol': str(Path(protocol_path).resolve()),
              'ledger': str(Path(ledger_path).resolve()), 'jobs': [str(Path(p).resolve()) for p in job_paths],
              'artifacts': str(Path(artifacts).resolve()), 'cache': str(Path(cache).resolve())}
    write_json(root/'config.json', config)
    script = root/'coordinate.py'
    script.write_text('import sys\n'+f'sys.path.insert(0,{str(Path(__file__).resolve().parents[1])!r})\n'+
                      'from saal_rsi.adaptive_campaign import coordinator\n'+f'coordinator({str(root/"config.json")!r})\n')
    hardware = root/'gpu-observation.py'
    hardware.write_text('import subprocess,json,sys\n'+
        'raw=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.free,driver_version","--format=csv,noheader,nounits"],text=True,timeout=15)\n'+
        'rows=[r.strip().split(",") for r in raw.strip().splitlines()]\n'+
        'if len(rows)!=1 or "T4" not in rows[0][0] or int(rows[0][2].strip())<6144:raise ValueError("controller T4/memory observation failed")\n'+
        'print(json.dumps({"monitor":"controller_gpu_observation","nvidia_smi":raw,"python":sys.version,"scope":"provider runtime command outside training child; not hostile-host attestation"}),flush=True)\n')
    commands = [['colab', 'new', '-s', session, '--gpu', 'T4'],
                ['colab', 'upload', '-s', session, str(root/'gpu-lock.txt'), '/content/saal_adaptive_lock.txt'],
                ['colab', 'upload', '-s', session, str(root/'worker.zip'), '/content/saal_adaptive_worker.zip'],
                ['colab', 'exec', '-s', session, '-f', str(installer), '--timeout', '960'],
                ['colab', 'exec', '-s', session, '-f', str(hardware), '--timeout', '30'],
                [sys.executable, '-I', str(script)], ['colab', 'stop', '-s', session]]
    plan = {'schema': 'saal-adaptive-colab-plan-v1', 'session': session, 'job_digest': sha(protocol),
            'estimated_max_seconds': protocol['budget']['total_lifecycle_seconds'], 'commands': commands,
            'phases': ['PROVISIONING', 'UPLOAD_LOCK', 'UPLOAD_WORKER', 'DEPENDENCIES', 'HARDWARE_OBSERVATION', 'CAMPAIGN'],
            'external_deployment_allowed': False}
    write_json(root/'plan.json', plan)
    ledger = AdaptiveRegistry(ledger_path, pin, protocol_envelope)
    succeeded = False
    ledger.begin_provisioning()
    try:
        state = execute_colab_plan(plan, root)
        succeeded = state['phase'] == 'COMPLETED' and state['cleanup_verified'] is True
        return state
    finally:
        ledger.end(succeeded)
        write_json(root/'registry-export.json', ledger.records())
        ledger.close()
