"""Distinct v6 child; signed data-only policies, no qualifying or authorizing."""
import argparse,os,json,time,platform
from pathlib import Path
from .spec import canonical,parse_unique_json,sha
from .model_contract import verify
from .pretrained_contract import job_bytes
from .epoch_contract import validate_job,worker_identity
from .epoch_artifacts import seal,validate_bundle,restoration_manifest
from .epoch_training import train
from .pretrained_registry import fetch_snapshot,validate_snapshot
from .learning_backend import create_backend,verify_dependencies
from .epoch_metrics import prediction_equal

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--public-key',required=True);p.add_argument('--out',required=True);p.add_argument('--cache',required=True);p.add_argument('--replay',action='store_true');a=p.parse_args()
    envelope=parse_unique_json(job_bytes(a.job));job=verify(envelope,a.public_key);validate_job(job);started=time.monotonic();spec=job['model_spec'];out=Path(a.out)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    versions=verify_dependencies()
    import torch
    torch.set_num_threads(2);torch.manual_seed(job['seed']);free=total=0
    if spec['compute_backend']=='colab':
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise ValueError('admitted T4 unavailable')
        free,total=torch.cuda.mem_get_info()
        if free<6*1024**3:raise ValueError('insufficient GPU memory')
        if os.environ.get('SAAL_HASH_LOCK_ENFORCED')!='1':raise ValueError('GPU hash-lock installation evidence required')
        torch.cuda.reset_peak_memory_stats()
    snapshot=fetch_snapshot(a.cache,spec['model_repository'],spec['model_revision'],lambda v:print(json.dumps(v),flush=True));manifest=validate_snapshot(snapshot,spec['model_repository'],spec['model_revision'])
    backend=create_backend(spec,snapshot,job['seed']).load_pretrained()
    for i in range(0,len(job['dataset']['records']),16):backend.tokenize(job['dataset']['records'][i:i+16])
    if a.replay:
        result=validate_bundle(out,job,a.public_key);backend.restore_checkpoint(out/'best.safetensors');fresh=backend.evaluate([r for r in job['dataset']['records'] if r['split']=='development'])
        if not prediction_equal(fresh,result['snapshots']['best']['predictions']['development'],spec['model_kind'],1e-4):raise ValueError('fresh-process GPU replay failure')
        print(json.dumps({'monitor':'fresh_process_replay','verified':True,'records':len(fresh),'job_digest':sha(job)}),flush=True);return
    if time.monotonic()-started>job['max_seconds']-30:raise TimeoutError('preflight consumed budget')
    result=train(backend,job,out,deadline=started+job['max_seconds'])
    result.update(schema='saal-model-result-v6',job_digest=sha(job),worker_identity=worker_identity(),model_identity=sha(restoration_manifest(out,job)),restoration_manifest=restoration_manifest(out,job),parameter_inventory=backend.parameter_inventory(),runtime={'versions':versions,'python':platform.python_version(),'platform':platform.platform(),'device':str(next(backend.model.parameters()).device),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'free_gpu_bytes_preflight':free,'total_gpu_bytes':total,'attention':spec['attention_implementation'],'vision_used':False,'dependency_hash_lock_enforced':os.environ.get('SAAL_HASH_LOCK_ENFORCED')=='1'},total_worker_seconds=time.monotonic()-started)
    for n,v in [('signed_training_job.json',envelope),('model_spec.json',spec),('dataset_manifest.json',job['dataset']),('snapshot_manifest.json',manifest),('result.json',result)]:(out/n).write_bytes(canonical(v))
    seal(out);validate_bundle(out,job,a.public_key)
    print(json.dumps({'monitor':'training_complete','selected_step':result['snapshots']['best']['step'],'job_digest':sha(job),'qualification_status':'NOT_EVALUATED'}),flush=True)
