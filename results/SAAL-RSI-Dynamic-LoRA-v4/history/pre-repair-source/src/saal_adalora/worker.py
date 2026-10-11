"""Separate experimental worker and reconstruction; no qualification authority."""
from pathlib import Path
import json,sys
from saal_rsi.spec import canonical,parse_unique_json,sha
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_registry import validate_snapshot
from saal_rsi.learning_backend import verify_dependencies
from saal_rsi.checkpoint_selection import metrics
from saal_rsi.sequence_replay_v2 import prediction_equal
from .contract import validate,source_identity,ADAPTER
from .backend import DynamicBackend
from .training import train_backend

def load(root,cache):
    import torch
    torch.set_num_threads(2)
    root=Path(root);pin=(root/'controller.pub').read_text().strip();job=verify(parse_unique_json((root/'signed-job.json').read_bytes()),pin);reference=validate(job)
    spec=reference['model_spec'];snapshot=Path(cache)/spec['model_repository'].split('/')[-1]/spec['model_revision']
    validate_snapshot(snapshot,spec['model_repository'],spec['model_revision']);verify_dependencies()
    return root,job,reference,DynamicBackend(reference,snapshot).load_pretrained()

def train(root,cache):
    root,job,reference,backend=load(root,cache);inventory=backend.parameter_inventory()
    result=train_backend(backend,reference['dataset'],{'job':reference,'out':root/'model'})
    result.update(schema='saal-adalora-research-result-v1',job_digest=sha(job),source_identity=source_identity(),dynamic_adapter=ADAPTER,tensor_inventory=inventory,model_identity=reference['model_spec'],identity_note='Native reference pins the base/tokenizer and fixed comparison; this new result is AdaLoRA, never native fixed-LoRA v5 evidence.')
    if not result['real_parameter_updates'] or not result['frozen_integrity']['unchanged']:raise ValueError('dynamic updates/freeze gate')
    (root/'result.json').write_bytes(canonical(result))

def replay(root,cache):
    root,job,reference,backend=load(root,cache);result=parse_unique_json((root/'result.json').read_bytes())
    if result['schema']!='saal-adalora-research-result-v1' or result['job_digest']!=sha(job) or result['source_identity']!=source_identity() or result['dynamic_adapter']!=ADAPTER or result['tensor_inventory']!=backend.parameter_inventory() or result['qualification_status']!='NOT_EVALUATED' or result['external_deployment_allowed']is not False:raise ValueError('dynamic result identity/inventory')
    parts={s:[r for r in reference['dataset']['records'] if r['split']==s] for s in ('train','development','regression')};evidence={}
    for name in ('initial','best','terminal'):
        backend.restore_checkpoint(root/'model'/f'{name}.safetensors')
        predictions={s:backend.evaluate(records) for s,records in parts.items()}
        for split in ('development','regression'):
            if not prediction_equal(predictions[split],result['snapshots'][name]['predictions'][split],'modernbert-seqcls-v1',1e-4):raise ValueError('dynamic fresh-process prediction mismatch')
        allocation=result['snapshot_allocations'][name];pattern=allocation['assigned_rank_pattern']
        expected_names={n for n in backend.trainable if 'lora_E.default' in n}
        if pattern and (set(pattern)!=expected_names or any(type(m)is not list or len(m)!=ADAPTER['init_r'] or any(type(v)is not bool for v in m) for m in pattern.values())):raise ValueError('rank mask inventory')
        if pattern:
            for n,mask in pattern.items():
                if any(not keep and backend.trainable[n][i].detach().item()!=0 for i,keep in enumerate(mask)):raise ValueError('rank mask/weight substitution')
        actual={n:int((p.detach()!=0).sum()) for n,p in backend.trainable.items() if n in expected_names}
        if actual!=allocation['effective_nonzero_singular_values']:raise ValueError('effective rank mismatch')
        evidence[name]={'metrics':{s:metrics(parts[s],v,'modernbert-seqcls-v1') for s,v in predictions.items()},'predictions':predictions,'allocation':allocation}
    backend.verify_frozen_weights()
    (root/'fresh-replay.json').write_bytes(canonical({'schema':'saal-adalora-replay-v1','job_digest':sha(job),'snapshots':evidence,'all_snapshots_reproduced':True,'probability_tolerance':1e-4,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}))

if __name__=='__main__':
    if sys.argv[1]=='train':train(*sys.argv[2:])
    elif sys.argv[1]=='replay':replay(*sys.argv[2:])
    else:raise ValueError('closed dynamic worker operation')
