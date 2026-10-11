"""Separate signed sequence admission under the inherited exact-session watchdog."""
from pathlib import Path
import hashlib,zipfile,sys,time,json
from .spec import canonical,sha,parse_unique_json
from .model_contract import verify
from .pretrained_contract import validate_job
from .pretrained_continual import validate_sequence
from .pretrained_backend import setup_plan,validate_transfer,assemble_transfer,CHUNK_BYTES,TRANSFER_SECONDS
from .pretrained_registry import file_hash
from .model_monitor import execute_colab_plan,write_json


def execute_sequence(workspace,job_envelope,protocol_envelope,pin,learning_report,artifacts):
    job=verify(job_envelope,pin);protocol=verify(protocol_envelope,pin);validate_sequence(protocol,job)
    if job['model_spec']['compute_backend']!='colab':raise ValueError('remote sequence backend substitution')
    if protocol['positive_learning_report_digest']!=sha(learning_report):raise ValueError('sequence evidence substitution')
    root=Path(workspace);plan=setup_plan(root,protocol['max_seconds']+2600)
    # Reviewed sequence code is admitted by its distinct signed source identity.
    with zipfile.ZipFile(root/'worker.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
        for n in ('pretrained_continual.py','continual_learning.py','pretrained_sequence_worker.py'):z.writestr('src/saal_rsi/'+n,Path(__file__).with_name(n).read_bytes())
        z.writestr('sequence.py',"from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/'src'))\nfrom saal_rsi.pretrained_sequence_worker import main\nmain()\n")
    for name,value in [('sequence-job.json',job_envelope),('sequence-protocol.json',protocol_envelope),('learning-report.json',learning_report)]: (root/name).write_bytes(canonical(value))
    from .learning_experiments import LearningExperimentRegistry
    ledger=LearningExperimentRegistry(root/'budget.db',pin,max_jobs=1,max_seconds=job['max_seconds']+1400);digest=ledger.reserve(job_envelope);ledger.start(digest);ledger.close()
    session=plan['session'];uploads=[('sequence-job.json','saal_sequence_job.json'),('sequence-protocol.json','saal_sequence_protocol.json'),('learning-report.json','saal_learning_report.json')]
    for local,remote in uploads:plan['commands'].append(['colab','upload','-s',session,str(root/local),'/content/'+remote]);plan['phases'].append('UPLOAD_'+local.split('.')[0].upper())
    command=['/content/saal-pretrained-env/bin/python','-I','/content/saal-pretrained/sequence.py','--job','/content/saal_sequence_job.json','--protocol','/content/saal_sequence_protocol.json','--pin',pin,'--learning-report','/content/saal_learning_report.json','--cache','/content/saal-model-cache','--out','/content/saal_sequence_result']
    runner="import subprocess,json,zipfile,hashlib\nfrom pathlib import Path\n"+'command='+repr(command)+'\n'+"p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)\nfor line in iter(p.stdout.readline,b''):print(line.decode(errors='replace'),end='',flush=True)\nif p.wait():raise RuntimeError('sequence failed')\nwith zipfile.ZipFile('/content/saal_sequence.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:\n for p in Path('/content/saal_sequence_result').iterdir():z.write(p,p.name)\n"+f"root=Path('/content/saal_sequence_transfer');root.mkdir();chunks=[];total=0;h=hashlib.sha256()\nwith Path('/content/saal_sequence.zip').open('rb') as f:\n i=0\n while b:=f.read({CHUNK_BYTES}):\n  n=f'part-{{i:03d}}.bin';(root/n).write_bytes(b);chunks.append({{'name':n,'size':len(b),'sha256':hashlib.sha256(b).hexdigest()}});h.update(b);total+=len(b);i+=1\n(root/'manifest.json').write_text(json.dumps({{'schema':'saal-artifact-transfer-v1','job_digest':{digest!r},'size':total,'sha256':h.hexdigest(),'chunks':chunks}}))\n"
    (root/'run.py').write_text(runner);plan['commands'].append(['colab','exec','-s',session,'-f',str(root/'run.py'),'--timeout',str(protocol['max_seconds']+60)]);plan['phases'].append('SEQUENTIAL_TRAINING')
    # Retrieval runs as a separately monitored local stage.
    config={'root':str(root.resolve()),'session':session,'artifacts':str(Path(artifacts).resolve()),'job_digest':digest,'pin':pin};write_json(root/'retrieval.json',config)
    launcher=root/'retrieve.py';launcher.write_text("import sys\nsys.path.insert(0,"+repr(str(Path(__file__).resolve().parents[1]))+")\nfrom saal_rsi.pretrained_sequence_backend import retrieve\nretrieve("+repr(str((root/'retrieval.json').resolve()))+")\n")
    plan['commands']+=[[sys.executable,'-I',str(launcher)],['colab','stop','-s',session]];plan['phases']+=['DOWNLOAD_VERIFY'];write_json(root/'submission-plan.json',plan)
    complete=False
    try:
        state=execute_colab_plan(plan,root);complete=True;outcome={'cleanup_verified':state['cleanup_verified'],'result':parse_unique_json((Path(artifacts)/'result.json').read_bytes())};return outcome
    except BaseException as exc:outcome={'error':{'type':type(exc).__name__,'detail':str(exc)}};raise
    finally:
        ledger=LearningExperimentRegistry(root/'budget.db',pin);ledger.finish(digest,outcome,complete);write_json(root/'registry-export.json',ledger.records());ledger.close()
        state=parse_unique_json((root/'monitor-state.json').read_bytes());write_json(root/'cleanup-evidence.json',{'session':session,'confirmed':state['cleanup_verified']})


def retrieve(config_path):
    from .model_monitor import stream_command
    c=parse_unique_json(Path(config_path).read_bytes());root=Path(c['root']);parts=root/'transfer';parts.mkdir();session=c['session'];deadline=time.monotonic()+TRANSFER_SECONDS
    stream_command(['colab','download','-s',session,'/content/saal_sequence_transfer/manifest.json',str(parts/'manifest.json')],root/'transfer-manifest.log',60)
    m=validate_transfer(parse_unique_json((parts/'manifest.json').read_bytes()))
    if m['job_digest']!=c['job_digest']:raise ValueError('sequence transfer identity')
    for i,p in enumerate(m['chunks']):stream_command(['colab','download','-s',session,'/content/saal_sequence_transfer/'+p['name'],str(parts/p['name'])],root/f'chunk-{i}.log',min(60,deadline-time.monotonic()))
    archive=root/'sequence.zip';assemble_transfer(parts,m,archive);out=Path(c['artifacts'])
    if out.exists():raise FileExistsError('sequence artifact destination')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        allowed={'result.json','signed_job.json','signed_protocol.json','inventory.json'}|{f'{i}-{b}.safetensors' for i in range(4) for b in ('experimental','fresh_control')}
        if len(names)!=len(set(names)) or not set(names)<=allowed or not {'result.json','inventory.json','signed_job.json','signed_protocol.json'}<=set(names) or sum(p.file_size for p in z.infolist())>100_000_000 or z.testzip():raise ValueError('sequence inventory')
        out.mkdir();z.extractall(out)
    inv=parse_unique_json((out/'inventory.json').read_bytes())['files']
    if set(inv)!={p.name for p in out.iterdir() if p.name!='inventory.json'}:raise ValueError('sequence hash inventory')
    for n,r in inv.items():
        if (out/n).stat().st_size!=r['size'] or file_hash(out/n)!=r['sha256']:raise ValueError('sequence artifact bytes')
    result=parse_unique_json((out/'result.json').read_bytes());job_envelope=parse_unique_json((out/'signed_job.json').read_bytes());protocol_envelope=parse_unique_json((out/'signed_protocol.json').read_bytes())
    # Explicit pinned job/protocol binding; metrics are reconstructed from the
    # full matrix. They remain controller-owned research, not qualification.
    job=verify(job_envelope,c['pin']);protocol=verify(protocol_envelope,c['pin']);validate_sequence(protocol,job)
    if result.get('protocol_digest')!=sha(protocol) or result.get('worker_identity')!=protocol['worker_identity'] or 'T4' not in result.get('gpu',''):raise ValueError('sequence runtime/protocol identity')
    expected={'result.json','signed_job.json','signed_protocol.json','inventory.json'}|{f'{i}-{b}.safetensors' for i in range(len(protocol['groups'])) for b in ('experimental','fresh_control')}
    if {p.name for p in out.iterdir()}!=expected:raise ValueError('sequence stage inventory')
    from .pretrained_artifacts import tensor_inventory
    from safetensors import safe_open
    import torch
    tensors=tensor_inventory(job['model_spec'])
    for p in out.glob('*.safetensors'):
        with safe_open(str(p),framework='pt',device='cpu') as f:
            if set(f.keys())!=set(tensors):raise ValueError('sequence trainable tensor inventory')
            for n,shape in tensors.items():
                t=f.get_tensor(n)
                if list(t.shape)!=shape or t.dtype!=torch.float32 or not torch.isfinite(t).all():raise ValueError('sequence tensor validity')
    if job_envelope!=parse_unique_json((root/'sequence-job.json').read_bytes()) or protocol_envelope!=parse_unique_json((root/'sequence-protocol.json').read_bytes()):raise ValueError('sequence cross-job substitution')
    from .continual_learning import retention_metrics
    reconstructed=retention_metrics(result['accuracy_matrix'],result['initial_accuracy'])
    for k,v in reconstructed.items():
        if k!='limits' and result[k]!=v:raise ValueError('forged retention metrics')
    if result['qualification_status']!='NOT_EVALUATED' or result['external_deployment_allowed'] is not False:raise ValueError('sequence authority transition')
    print(json.dumps({'monitor':'sequence_artifacts_verified','verified':True,'qualification_status':'NOT_EVALUATED'}),flush=True)
