"""Reviewed v5 child. No evaluation keys, protected records, or promotion API."""
import argparse,json,time,os,platform,sys,traceback
from pathlib import Path
from .spec import canonical,parse_unique_json,sha
from .model_contract import verify
from .pretrained_contract import validate_job,worker_identity,job_bytes
from .pretrained_registry import fetch_snapshot,validate_snapshot
from .learning_backend import create_backend,verify_dependencies
from .pretrained_artifacts import seal,validate_bundle,restoration_manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--public-key',required=True);p.add_argument('--out',required=True);p.add_argument('--cache',required=True);p.add_argument('--replay',action='store_true');a=p.parse_args()
    envelope=parse_unique_json(job_bytes(a.job));job=verify(envelope,a.public_key);validate_job(job);started=time.monotonic();spec=job['model_spec'];out=Path(a.out)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    versions=verify_dependencies()
    import torch
    torch.set_num_threads(2);torch.manual_seed(job['seed'])
    if spec['compute_backend']=='colab':
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise ValueError('admitted T4 unavailable')
        free,total=torch.cuda.mem_get_info()
        if free<6*1024**3:raise ValueError('insufficient free GPU memory')
        torch.cuda.reset_peak_memory_stats()
    else:free=total=0
    snapshot=fetch_snapshot(a.cache,spec['model_repository'],spec['model_revision'],lambda v:print(json.dumps(v),flush=True));manifest=validate_snapshot(snapshot,spec['model_repository'],spec['model_revision'])
    backend=create_backend(spec,snapshot,job['seed']).load_pretrained()
    # Admit every prompt through its native tokenizer before any optimizer update.
    records=job['dataset']['records']
    if spec['model_kind']=='modernbert-seqcls-v1':
        for i in range(0,len(records),16):backend.tokenize(records[i:i+16])
    else:
        from .qwen35_backend import completion_tokens
        for r in records:completion_tokens(backend.tokenizer,r['prompt'],r['expected_output'],spec['max_input_tokens'])
    if a.replay:
        result=validate_bundle(out,job,a.public_key);backend.restore_checkpoint(out/'best.safetensors');chosen=result['snapshots']['best']['predictions']['development'];fresh=backend.evaluate([r for r in records if r['split']=='development'],job['max_new_tokens'])
        if spec['model_kind']=='modernbert-seqcls-v1':
            agreement=all(x['decision']==y['decision'] and max(abs(u-v) for u,v in zip(x['probabilities'],y['probabilities']))<=1e-4 for x,y in zip(chosen,fresh))
        else:agreement=fresh==chosen
        if not agreement:raise ValueError('fresh-process checkpoint replay mismatch')
        print(json.dumps({'monitor':'fresh_process_replay','verified':True,'model_identity':result['model_identity'],'records':len(fresh)}),flush=True);return
    remaining=job['max_seconds']-(time.monotonic()-started)
    if remaining<30:raise TimeoutError('snapshot/load used admitted run budget')
    training_job=job|{'max_seconds':int(remaining)}
    result=backend.train(job['dataset'],{'job':training_job,'out':out})
    result.update(schema='saal-model-result-v4',job_digest=sha(job),model_identity=sha(restoration_manifest(out,job)),restoration_manifest=restoration_manifest(out,job),worker_identity=worker_identity(),parameter_inventory=backend.parameter_inventory(),
       runtime={'versions':versions,'python':platform.python_version(),'platform':platform.platform(),'device':str(next(backend.model.parameters()).device),'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'free_gpu_bytes_preflight':free,'total_gpu_bytes':total,'attention':spec['attention_implementation'],'vision_used':False,'dependency_hash_lock_enforced':os.environ.get('SAAL_HASH_LOCK_ENFORCED')=='1'},
       total_worker_seconds=time.monotonic()-started)
    for n,v in [('signed_training_job.json',envelope),('model_spec.json',spec),('dataset_manifest.json',job['dataset']),('snapshot_manifest.json',manifest),('result.json',result)]: (out/n).write_bytes(canonical(v))
    if not (out/'training_metrics.jsonl').exists():(out/'training_metrics.jsonl').write_text('{}\n')
    seal(out);validate_bundle(out,job,a.public_key);print(json.dumps({'monitor':'training_complete','job_digest':sha(job),'selected_step':result['snapshots']['best']['step'],'qualification_status':'NOT_EVALUATED'}),flush=True)

if __name__=='__main__':main()
