"""Reviewed worker entry point. No custody/signing/promotion keys or protected labels."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from .spec import canonical,sha,parse_unique_json
from .model_contract import verify
from .neural_contract import validate_neural_job,worker_digest
from .neural_training import train_job,load_models,predict

FILES={'experiment_manifest.json','signed_training_job.json','dataset_manifest.json','model_spec.json',
       'baseline.safetensors','adapter.safetensors','candidate.safetensors','checkpoint_selection.json',
       'training_metrics.jsonl','runtime_metrics.json','resource_accounting.json','tensor_inventory.json',
       'cleanup_evidence.json','checkpoint_replay.json','artifact_inventory.json','result.json'}

def seal_inventory(root,result):
    result['artifact_hashes']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(FILES-{'result.json','artifact_inventory.json'})}
    (root/'result.json').write_bytes(canonical(result))
    (root/'artifact_inventory.json').write_bytes(canonical({'schema':'saal-neural-artifacts-v3','files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(FILES-{'artifact_inventory.json'})}}))

def infer(bundle,spec,mode,prompts):
    torch.set_num_threads(2)
    baseline,candidate=load_models(bundle,spec,mode)
    records=[{'prompt':p} for p in prompts]
    return {'baseline':predict(baseline,records),'candidate':predict(candidate,records),
            'runtime':{'device':'cpu','gpu':None,'versions':{k:version(k) for k in ('torch','safetensors','cryptography')},
                       'python':platform.python_version(),'model_spec_digest':sha(spec),'worker_sha256':worker_digest()}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--public-key');p.add_argument('--out',required=True);p.add_argument('--baseline');p.add_argument('--require-gpu')
    p.add_argument('--infer',action='store_true');p.add_argument('--bundle');p.add_argument('--request')
    args=p.parse_args()
    if args.infer:
        request=parse_unique_json(Path(args.request).read_bytes())
        if set(request)!={'model_spec','mode','prompts'}:raise ValueError('inference request schema')
        result=infer(args.bundle,request['model_spec'],request['mode'],request['prompts']);Path(args.out).write_bytes(canonical(result));return
    envelope=parse_unique_json(Path(args.job).read_bytes());job=verify(envelope,args.public_key);validate_neural_job(job)
    result=train_job(job,args.out,args.baseline,args.require_gpu);root=Path(args.out)
    (root/'signed_training_job.json').write_bytes(canonical(envelope))
    (root/'experiment_manifest.json').write_bytes(canonical({'schema':'saal-neural-experiment-v1','job_digest':sha(job),'model_spec_digest':sha(job['model_spec']),
        'dataset_digest':job['dataset_digest'],'mode':job['mode'],'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}))
    replay=infer(root,job['model_spec'],job['mode'],[r['prompt'] for r in job['dataset']['records'] if r['split']=='development'])
    parity={k:replay[k]['labels']==result['development_predictions'][k]['labels'] and torch.allclose(torch.tensor(replay[k]['logits']),torch.tensor(result['development_predictions'][k]['logits']),atol=1e-4,rtol=1e-4) for k in ('baseline','candidate')}
    if not all(parity.values()):raise ValueError('export CPU checkpoint replay failed')
    (root/'checkpoint_replay.json').write_bytes(canonical({'scope':'worker-export-to-CPU','parity':parity,'atol':.0001,'rtol':.0001,'predictions':replay}))
    (root/'cleanup_evidence.json').write_bytes(canonical({'authority':'controller','confirmed':False,'status':'HOST_CONFIRMATION_REQUIRED'}))
    seal_inventory(root,result)

if __name__=='__main__':main()
