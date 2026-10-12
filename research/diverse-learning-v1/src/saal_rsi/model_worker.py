"""Standalone compute worker. No authority keys, holdout labels, or qualification logic.

Byte-encoded tiny attention classifier; genuine supervised baseline and LoRA updates.
Not a pretrained LLM, language generation benchmark, or alignment result.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from importlib.metadata import version

LABELS=('CHEAP','FAST','ASK','REFUSE','FORBIDDEN')

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return 'sha256:'+hashlib.sha256(canonical(x)).hexdigest()

def loaded_versions():
    import torch,safetensors,cryptography
    return {'torch':torch.__version__.split('+')[0],'safetensors':safetensors.__version__,'cryptography':cryptography.__version__}

def runtime():
    import torch
    return {'python':sys.version,'platform':platform.platform(),'versions':loaded_versions(),
            'device':'cuda' if torch.cuda.is_available() else 'cpu','cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'limits':'Worker self-report; not remote attestation.'}

def setup(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    torch.set_num_threads(1);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def architecture(rank):
    import torch
    from torch import nn
    class LowRank(nn.Module):
        def __init__(self,inputs,outputs):
            super().__init__();self.base=nn.Linear(inputs,outputs)
            self.a=nn.Parameter(torch.randn(rank,inputs)*.02)
            self.b=nn.Parameter(torch.zeros(outputs,rank));self.enabled=False
        def forward(self,x):
            out=self.base(x)
            return out+(x@self.a.T@self.b.T)/rank if self.enabled else out
    class Attention(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(257,32,padding_idx=0)
            self.attention=LowRank(32,1);self.classifier=LowRank(32,5)
        def forward(self,ids):
            e=self.embedding(ids);scores=self.attention(e).squeeze(-1).masked_fill(ids==0,-1e4)
            pooled=(e*torch.softmax(scores,dim=-1).unsqueeze(-1)).sum(1)
            return self.classifier(pooled)
        def enable_adapter(self,flag):
            self.attention.enabled=flag;self.classifier.enabled=flag
    return Attention()

def encode(prompts,device):
    import torch
    result=torch.zeros((len(prompts),128),dtype=torch.long)
    for i,prompt in enumerate(prompts):
        values=list(prompt.encode())
        if not 1<=len(values)<=128:raise ValueError('prompt length')
        result[i,:len(values)]=torch.tensor([x+1 for x in values])
    return result.to(device)

def predict(model,prompts,device):
    import torch
    model.eval();output=[]
    with torch.no_grad():
        for start in range(0,len(prompts),64):
            output.extend(model(encode(prompts[start:start+64],device)).argmax(-1).cpu().tolist())
    return [LABELS[x] for x in output]

def development_score(predictions,records):
    categories={r['category'] for r in records}
    scores={k:sum(x==r['label'] for x,r in zip(predictions,records) if r['category']==k)/sum(r['category']==k for r in records) for k in categories}
    return (min(scores.values()),sum(scores.values())/len(scores),sum(x==r['label'] for x,r in zip(predictions,records))/len(records))

def class_weights(records,device):
    import torch
    from collections import Counter
    counts=Counter(r['label'] for r in records);present=len(counts)
    return torch.tensor([len(records)/(present*counts[label]) if counts[label] else 0 for label in LABELS],device=device)

def emit(event,**values):
    print(json.dumps({'monitor':event,**values},allow_nan=False),flush=True)

def train(payload,out,baseline_path=None):
    import torch
    from safetensors.torch import save_file
    device=setup(payload['seed']);model=architecture(payload['rank']).to(device)
    started=time.monotonic();records=payload['training'];dev=payload['development']
    x=encode([r['prompt'] for r in records],device);y=torch.tensor([LABELS.index(r['label']) for r in records],device=device)
    adapter_records=payload['adapter_training'];adapter_x=encode([r['prompt'] for r in adapter_records],device)
    adapter_y=torch.tensor([LABELS.index(r['label']) for r in adapter_records],device=device)
    weights=class_weights(adapter_records,device);losses=[];history=[];selected=None;best_score=None;selected_step=0
    total=payload['baseline_steps']+payload['adapter_steps'];completed=0
    if payload['baseline_sha256'] is not None:
        from safetensors.torch import load_file
        raw=Path(baseline_path).read_bytes()
        if len(raw)>100000 or hashlib.sha256(raw).hexdigest()!=payload['baseline_sha256']:raise ValueError('frozen baseline hash')
        base=load_file(str(baseline_path));expected={k:v for k,v in model.state_dict().items() if not k.endswith(('.a','.b'))}
        if set(base)!=set(expected) or any(base[k].shape!=v.shape or base[k].dtype!=v.dtype or not torch.isfinite(base[k]).all() for k,v in expected.items()):raise ValueError('frozen baseline tensor schema')
        model.load_state_dict(base,strict=False);(out/'baseline.safetensors').write_bytes(raw)
    emit('worker_start',runtime=runtime(),total_steps=total)
    def score_candidate(step):
        nonlocal selected,best_score,selected_step
        predictions=predict(model,[r['prompt'] for r in dev],device);score=development_score(predictions,dev)
        if best_score is None or score>best_score:
            best_score=score;selected_step=step
            selected={k:v.detach().cpu().clone().contiguous() for k,v in model.state_dict().items() if k.endswith(('.a','.b'))}
        model.train();return score
    for adapter,steps in ((False,payload['baseline_steps']),(True,payload['adapter_steps'])):
        model.enable_adapter(adapter)
        for name,param in model.named_parameters():param.requires_grad_(name.endswith(('.a','.b')) if adapter else not name.endswith(('.a','.b')))
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=payload['adapter_learning_rate'] if adapter else payload['learning_rate'])
        model.train()
        if adapter:score_candidate(0)  # retain the incumbent if no learned checkpoint is better
        for step in range(1,steps+1):
            if time.monotonic()-started>payload['max_seconds']:raise TimeoutError('training budget elapsed')
            if adapter:
                # Full bounded data batch: no rare-class dropout from random minibatches.
                indices=torch.arange(len(adapter_records),device=device)
            else:
                # Preserve the original incumbent training procedure exactly.
                indices=torch.randperm(len(records),device=device)[:32]
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(adapter_x[indices] if adapter else x[indices]),adapter_y[indices] if adapter else y[indices],weight=weights if adapter else None)
            loss.backward();optimizer.step();completed+=1
            score=score_candidate(step) if adapter and (step%10==0 or step==steps) else None
            if step%10==0 or step==steps:
                entry={'phase':'adapter' if adapter else 'baseline','step':step,'completed_steps':completed,'total_steps':total,
                    'objective_loss':float(loss.detach().cpu()),'elapsed_seconds':time.monotonic()-started,
                    'development_score':list(score) if score is not None else None,
                    'gpu_allocated_bytes':torch.cuda.memory_allocated() if device=='cuda' else 0,
                    'gpu_reserved_bytes':torch.cuda.memory_reserved() if device=='cuda' else 0}
                history.append(entry);emit('training_progress',**entry)
        if not adapter:
            model.eval()
            with torch.no_grad():losses.append(float(torch.nn.functional.cross_entropy(model(x),y).cpu()))
            base={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if not k.endswith(('.a','.b'))}
            if payload['baseline_sha256'] is None:save_file(base,str(out/'baseline.safetensors'))
    model.load_state_dict(selected,strict=False);model.enable_adapter(True);model.eval()
    with torch.no_grad():losses.append(float(torch.nn.functional.cross_entropy(model(x),y).cpu()))
    save_file(selected,str(out/'adapter.safetensors'))
    prompts=[r['prompt'] for r in dev];candidate=predict(model,prompts,device)
    model.enable_adapter(False);baseline=predict(model,prompts,device)
    result={'schema':'saal-model-result-v2','job_digest':digest(payload),'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runtime':runtime(),'losses':losses,'loss_measurement':'full-training-unweighted-cross-entropy-at-exported-checkpoints',
            'selection':{'selected_adapter_step':selected_step,'trained_adapter_steps':payload['adapter_steps'],
                 'development_score':list(best_score),'policy':payload['checkpoint_selection'],
                 'objective':payload['adapter_objective'],'class_weights':weights.cpu().tolist(),
                 'labels_absent_from_training':[label for label in LABELS[:4] if not any(r['label']==label for r in adapter_records)]},
            'training_history':history,'development_predictions':{'baseline':baseline,'candidate':candidate},
            'elapsed_seconds':time.monotonic()-started,'rank':payload['rank'],
            'artifact_hashes':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('baseline.safetensors','adapter.safetensors')},
            'qualification_passed':False,'external_deployment_allowed':False,'status':'UNTRUSTED_DEVELOPMENT_RESULT'}
    (out/'result.json').write_bytes(canonical(result));emit('worker_complete',selected_adapter_step=selected_step,completed_steps=completed)
    return result


def infer(bundle,prompts):
    from safetensors.torch import load_file
    result=json.loads((bundle/'result.json').read_text());device=setup(0);model=architecture(result['rank'])
    base=load_file(str(bundle/'baseline.safetensors'));adapter=load_file(str(bundle/'adapter.safetensors'))
    expected=model.state_dict()
    if set(base)|set(adapter)!=set(expected) or set(base)&set(adapter):raise ValueError('tensor inventory')
    import torch
    for name,tensor in {**base,**adapter}.items():
        if tensor.shape!=expected[name].shape or tensor.dtype!=expected[name].dtype or not torch.isfinite(tensor).all():raise ValueError('tensor schema/nonfinite')
    model.load_state_dict({**base,**adapter},strict=True);model.to(device)
    model.enable_adapter(False);baseline=predict(model,prompts,device)
    model.enable_adapter(True);candidate=predict(model,prompts,device)
    return {'baseline':baseline,'candidate':candidate,'runtime':runtime()}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--baseline');p.add_argument('--require-gpu');p.add_argument('--infer',action='store_true');p.add_argument('--job');p.add_argument('--public-key');p.add_argument('--out',required=True);p.add_argument('--bundle');p.add_argument('--prompts')
    a=p.parse_args();out=Path(a.out)
    if a.infer:
        prompts=json.loads(Path(a.prompts).read_text())['prompts']
        out.write_bytes(canonical(infer(Path(a.bundle),prompts)))
    else:
        envelope=json.loads(Path(a.job).read_text());payload=envelope['payload']
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        import base64
        pub=bytes.fromhex(a.public_key)
        if envelope['key_sha256']!=hashlib.sha256(pub).hexdigest():raise ValueError('controller identity')
        Ed25519PublicKey.from_public_bytes(pub).verify(base64.b64decode(envelope['signature_b64'],validate=True),canonical(payload))
        if payload['worker_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('worker hash')
        if payload['versions']!=loaded_versions():raise ValueError('dependency versions')
        if payload['final_holdout_included'] is not False or payload['external_deployment_allowed'] is not False:raise ValueError('restricted job')
        if a.require_gpu:
            import torch
            if not torch.cuda.is_available() or a.require_gpu not in torch.cuda.get_device_name(0):raise RuntimeError('Required CUDA GPU unavailable')
        out.mkdir(exist_ok=False);train(payload,out,a.baseline)
