"""Strict v4 result inventory, snapshot reconstruction, and public metric checks."""
from pathlib import Path
import hashlib
from .spec import parse_unique_json,canonical,sha
from .pretrained_contract import validate_job,PretrainedModelSpec,registry_path,VERSIONS,entry
from .pretrained_registry import file_hash
from .checkpoint_selection import metrics,selection_key

FILES={'signed_training_job.json','model_spec.json','dataset_manifest.json','snapshot_manifest.json','initial.safetensors','best.safetensors','terminal.safetensors','training_metrics.jsonl','result.json','artifact_inventory.json'}
def tensor_inventory(spec):
    s=PretrainedModelSpec.parse(spec);base=parse_unique_json(registry_path().with_name('tensor_inventories.json').read_bytes())[s.model_repository]
    if not base:raise ValueError('unreviewed base tensor inventory')
    heads={'head.dense.weight':[768,768],'head.norm.weight':[768],'classifier.weight':[4,768],'classifier.bias':[4]};out={}
    if s.adapter_type=='none':return {'__no_adaptation__':[1]}
    if s.adapter_type=='head':return heads
    if s.adapter_type=='full':return {k:v['shape'] for k,v in base.items() if k.startswith('model.')}|heads
    for target in s.adapter_targets:
        shape=base[target+'.weight']['shape'];prefix='base_model.model.'+target
        out[prefix+'.lora_A.default.weight']=[s.adapter_rank,shape[1]];out[prefix+'.lora_B.default.weight']=[shape[0],s.adapter_rank]
    if s.model_kind=='modernbert-seqcls-v1':
        for k,shape in heads.items():
            module,tail=k.split('.',1);out['base_model.model.'+module+'.modules_to_save.default.'+tail]=shape
    return out

def restoration_manifest(root,job):
    root=Path(root)
    from safetensors import safe_open
    import torch
    heads={}
    with safe_open(str(root/'best.safetensors'),framework='pt',device='cpu') as f:
        for n in f.keys():
            if n.startswith(('head.','classifier.')) or '.head.modules_to_save.' in n or '.classifier.modules_to_save.' in n:
                t=f.get_tensor(n);heads[n]=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    s=job['model_spec']
    return {'schema':'saal-restoration-manifest-v1','model_spec_digest':sha(s),'base_files_digest':s['model_files_digest'],'base_revision':s['model_revision'],'architecture':s['architecture'],'tokenizer_digest':s['tokenizer_digest'],'processor_digest':s['processor_digest'],'chat_template_digest':s['chat_template_digest'],'checkpoint_sha256':file_hash(root/'best.safetensors'),'head_digest':sha(heads) if heads else None,'head_tensor_hashes':heads,'initialization_seed':job['seed'],'runtime_dependencies':s['dependency_identity'],'worker_identity':job['worker_identity'],'output_schema':s['output_schema_version'],'external_deployment_allowed':False}

def seal(root):
    root=Path(root);files={n:{'sha256':file_hash(root/n),'size':(root/n).stat().st_size} for n in sorted(FILES-{'artifact_inventory.json'})}
    (root/'artifact_inventory.json').write_bytes(canonical({'schema':'saal-pretrained-artifacts-v1','files':files}))

def validate_bundle(root,job,pin=None):
    import torch
    from safetensors import safe_open
    from .model_contract import verify
    root=Path(root);validate_job(job,execution=False)
    if root.is_symlink() or {p.name for p in root.iterdir()}!=FILES or any(p.is_symlink() or not p.is_file() for p in root.iterdir()):raise ValueError('v5 artifact inventory')
    inventory=parse_unique_json((root/'artifact_inventory.json').read_bytes())
    if inventory.get('schema')!='saal-pretrained-artifacts-v1' or set(inventory.get('files',{}))!=FILES-{'artifact_inventory.json'}:raise ValueError('inventory schema')
    limits={'initial.safetensors':650_000_000,'best.safetensors':650_000_000,'terminal.safetensors':650_000_000,'training_metrics.jsonl':5_000_000,'dataset_manifest.json':50_000_000,'signed_training_job.json':50_000_000}
    for n,record in inventory['files'].items():
        p=root/n
        if set(record)!={'sha256','size'} or p.stat().st_size!=record['size'] or not 1<=record['size']<=limits.get(n,10_000_000) or file_hash(p)!=record['sha256']:raise ValueError('artifact substitution/bounds')
    envelope=parse_unique_json((root/'signed_training_job.json').read_bytes())
    if envelope.get('payload')!=job or pin is not None and verify(envelope,pin)!=job:raise ValueError('signed job substitution')
    if parse_unique_json((root/'model_spec.json').read_bytes())!=job['model_spec'] or parse_unique_json((root/'dataset_manifest.json').read_bytes())!=job['dataset']:raise ValueError('configuration/data substitution')
    snap=parse_unique_json((root/'snapshot_manifest.json').read_bytes());spec=job['model_spec'];base=parse_unique_json(registry_path().with_name('tensor_inventories.json').read_bytes())[spec['model_repository']]
    if snap!={'repository':spec['model_repository'],'revision':spec['model_revision'],'files_digest':spec['model_files_digest'],'tensor_inventory':base}:raise ValueError('base tensor provenance substitution')
    expected=tensor_inventory(job['model_spec'])
    for n in ('initial.safetensors','best.safetensors','terminal.safetensors'):
        with safe_open(str(root/n),framework='pt',device='cpu') as f:
            if set(f.keys())!=set(expected):raise ValueError('checkpoint tensor names')
            for k,shape in expected.items():
                t=f.get_tensor(k)
                if list(t.shape)!=shape or t.dtype!=torch.float32 or not torch.isfinite(t).all():raise ValueError('checkpoint shape/dtype/finite')
    r=parse_unique_json((root/'result.json').read_bytes())
    if r.get('schema')!='saal-model-result-v4' or r.get('job_digest')!=sha(job) or r.get('worker_identity')!=job['worker_identity'] or r.get('model_identity')!=sha(restoration_manifest(root,job)) or r.get('restoration_manifest')!=restoration_manifest(root,job) or r.get('qualification_status')!='NOT_EVALUATED' or r.get('external_deployment_allowed') is not False:raise ValueError('result identity/authority')
    if r.get('runtime',{}).get('versions')!=VERSIONS or r['runtime'].get('vision_used') is not False:raise ValueError('runtime identity')
    if r.get('parameter_inventory')!={k:{'shape':v,'dtype':'torch.float32','numel':__import__('math').prod(v)} for k,v in expected.items() if k!='__no_adaptation__'}:raise ValueError('reported parameter inventory')
    snaps=r['snapshots']
    if set(snaps)!={'initial','best','terminal'} or snaps['initial']['step']!=0 or snaps['terminal']['step']!=job['steps'] or r['resource_accounting']['steps']!=job['steps']:raise ValueError('checkpoint/work budget')
    for snapshot in snaps.values():
        for split in ('development','regression'):
            records=[x for x in job['dataset']['records'] if x['split']==split]
            if metrics(records,snapshot['predictions'][split],job['model_spec']['model_kind'])!=snapshot['metrics'][split]:raise ValueError('forged public metrics')
    history=r['selection_history']
    if not history or history[0]['step']!=0 or history[-1]['step']!=job['steps'] or any(a['step']>=b['step'] for a,b in zip(history,history[1:])):raise ValueError('selection history')
    selected=max(history,key=lambda x:selection_key(x['metrics']['development']))
    if selected['step']!=snaps['best']['step'] or selected['metrics']!=snaps['best']['metrics']:raise ValueError('checkpoint selection substitution')
    if job['steps'] and inventory['files']['terminal.safetensors']['sha256']==inventory['files']['initial.safetensors']['sha256'] or job['steps'] and r.get('real_parameter_updates')is not True:raise ValueError('no measurable parameter updates')
    if r['frozen_integrity'].get('unchanged')is not True or r['frozen_integrity']['initial']!=r['frozen_integrity']['final'] or r.get('reload_parity')is not True:raise ValueError('freeze/reload failure')
    return r
