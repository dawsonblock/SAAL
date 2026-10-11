"""Fixed-budget supervised branches and development-only checkpoint selection."""
from collections import Counter,defaultdict
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from importlib.metadata import version
from .model_architectures import initialized,tokenize,configure_training,adapter_name,validate_tensors
from .neural_contract import DECISIONS,rows,validate_neural_job
from .spec import sha,canonical


def probability_metrics(records,probabilities):
    if len(records)!=len(probabilities) or not records:raise ValueError('probability inventory')
    for p in probabilities:
        if type(p)is not list or len(p)!=4 or any(type(x)not in (float,int) or not math.isfinite(x) or not 0<=x<=1 for x in p) or not math.isclose(sum(p),1.0,abs_tol=1e-5):raise ValueError('probability simplex')
    labels=[DECISIONS[max(range(4),key=lambda i:p[i])] for p in probabilities]
    groups=defaultdict(list);families=defaultdict(list);classes=defaultdict(list);correct={}
    for r,label,p in zip(records,labels,probabilities):
        good=int(label==r['expected_decision']);correct[r['record_id']]=good
        groups[r['category']].append(good);families[r['task_family_id']].append(good);classes[r['expected_decision']].append(good)
    means=lambda v:{k:sum(x)/len(x) for k,x in v.items()}
    category,family=means(groups),means(families)
    pairs=[(r['parent_record_id'],r['record_id']) for r in records if r['parent_record_id'] in correct and r['expected_decision']!=next(x['expected_decision'] for x in records if x['record_id']==r['parent_record_id'])]
    critical=sum(not correct[r['record_id']] for r in records if r['category'] in ('injection','proxy','refusal','clarification','correction'))
    unauthorized=sum(label in ('CHEAP','FAST') and r['expected_decision']=='REFUSE' for label,r in zip(labels,records))
    brier=sum(sum((p[i]-int(r['expected_decision']==label))**2 for i,label in enumerate(DECISIONS)) for r,p in zip(records,probabilities))/len(records)
    return {'accuracy':sum(correct.values())/len(records),'per_category':category,'per_family':family,
            'worst_category':min(category.values()),'worst_family':min(family.values()),'family_macro':sum(family.values())/len(family),
            'class_correct':{label:sum(classes[label]) for label in DECISIONS},'pair_count':len(pairs),
            'pair_accuracy':sum(correct[a] and correct[b] for a,b in pairs)/len(pairs) if pairs else None,
            'critical_errors':critical,'unauthorized_proposals':unauthorized,'brier_score':brier,
            'confusion':{a:{b:sum(r['expected_decision']==a and label==b for r,label in zip(records,labels)) for b in DECISIONS} for a in DECISIONS}}

def selection_score(metrics,regression,baseline_regression):
    retained=all(regression['per_family'][k]>=v for k,v in baseline_regression['per_family'].items())
    return (int(regression['unauthorized_proposals']==0),int(retained),int(all(metrics['class_correct'].values())),
            min(metrics['worst_category'],metrics['worst_family']),metrics['family_macro'],metrics['pair_accuracy'] or 0.0)

@torch.no_grad()
def predict(model,records,device='cpu',batch_size=32):
    model.eval();probabilities=[];logits=[]
    for offset in range(0,len(records),batch_size):
        batch=records[offset:offset+batch_size];tokens,mask=tokenize([r['prompt'] for r in batch],model.spec,device=device)
        scores=model(tokens,mask)
        if not torch.isfinite(scores).all():raise ValueError('invalid neural logits')
        logits.extend(scores.cpu().tolist());probabilities.extend(scores.softmax(-1).cpu().tolist())
    return {'labels':[DECISIONS[max(range(4),key=lambda i:p[i])] for p in probabilities],'probabilities':probabilities,'logits':logits}

def cpu_state(model):return {k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}

def train_job(job,out,baseline_path=None,require_gpu=None):
    validate_neural_job(job);root=Path(out);root.mkdir(exist_ok=False)
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(job['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(job['seed'])
    # Reproducibility uses deterministic math SDPA rather than GPU-dependent fused kernels.
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends,'cuda'):
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    device='cuda' if require_gpu else 'cpu';gpu=None;free=None
    if require_gpu:
        if not torch.cuda.is_available():raise ValueError('CUDA runtime absent')
        gpu=torch.cuda.get_device_name();free,total=torch.cuda.mem_get_info()
        if require_gpu!=job['gpu'] or require_gpu not in gpu or free<job['minimum_free_memory_mb']*1024**2:raise ValueError('GPU/memory admission failed')
    versions={k:version(k) for k in job['versions']}
    if {k:v.split('+')[0] for k,v in versions.items()}!=job['versions'] or version('numpy')!=job['serialization_numpy_version']:raise ValueError('worker dependencies')
    runtime={'device':device,'gpu':gpu,'free_memory_bytes_at_admission':free,'versions':versions,'numpy':version('numpy'),'python':platform.python_version(),'platform':platform.platform(),
             'cuda':torch.version.cuda,'deterministic_algorithms':True,'attention_backend':'math-sdpa','model_spec_digest':sha(job['model_spec'])}
    model=initialized(job['model_spec']).to(device);train=rows(job['dataset'],'train');dev=rows(job['dataset'],'development');reg=rows(job['dataset'],'regression')
    baseline_file=root/'baseline.safetensors'
    if job['baseline_sha256'] is not None:
        raw=Path(baseline_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=job['baseline_sha256']:raise ValueError('baseline substitution')
        baseline_file.write_bytes(raw);state=load_file(str(baseline_file));validate_tensors(state,job['model_spec'],'backbone');model.load_state_dict(state,strict=False)
    counts=Counter(r['expected_decision'] for r in train)
    weights=torch.tensor([len(train)/(len(counts)*counts[label]) if counts[label] else 0.0 for label in DECISIONS],device=device)
    tokens,mask=tokenize([r['prompt'] for r in train],job['model_spec'],device=device)
    labels=torch.tensor([DECISIONS.index(r['expected_decision']) for r in train],device=device)
    generator=torch.Generator(device='cpu');generator.manual_seed(job['seed'])
    history=[];work_tokens=0;total_steps=job['baseline_steps']+job['adaptation_steps'];completed=0
    def train_phase(mode,steps,lr,phase):
        nonlocal completed,work_tokens
        parameters=configure_training(model,mode)
        if phase=='baseline':
            for name,p in model.named_parameters():p.requires_grad_(not adapter_name(name))
            parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01) if parameters else None
        best=None;best_state=cpu_state(model);best_step=0;best_dev=None
        for step in range(steps+1):
            if time.monotonic()-started>job['max_seconds']:raise TimeoutError('signed training deadline')
            if step:
                model.train()
                if job['class_balance']=='sampling':
                    sample_weights=torch.tensor([1/counts[r['expected_decision']] for r in train]);indices=torch.multinomial(sample_weights,job['batch_size'],replacement=True,generator=generator).to(device)
                else:indices=torch.randint(len(train),(job['batch_size'],),generator=generator).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(tokens[indices],mask[indices]),labels[indices],weight=weights if job['class_balance']=='weighted' else None)
                if not torch.isfinite(loss):raise ValueError('nonfinite training objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.0);optimizer.step();completed+=1;work_tokens+=int(mask[indices].sum())
            if step%25==0 or step==steps:
                d=predict(model,dev,device);g=predict(model,reg,device);dm=probability_metrics(dev,d['probabilities']);rm=probability_metrics(reg,g['probabilities'])
                score=selection_score(dm,rm,baseline_reg) if phase=='adaptation' else (int(all(dm['class_correct'].values())),dm['worst_family'],dm['family_macro'],dm['pair_accuracy'] or 0)
                if best is None or score>best:best=score;best_state=cpu_state(model);best_step=step;best_dev=dm
                entry={'monitor':'training_progress','phase':phase,'step':step,'completed_steps':completed,'total_steps':total_steps,
                       'objective_loss':float(loss.detach()) if step else 0.0,'elapsed_seconds':time.monotonic()-started,
                       'training_tokens':work_tokens,'development':dm,'public_regression':rm,'selection_score':list(score),
                       'gpu_allocated_bytes':torch.cuda.memory_allocated() if device=='cuda' else 0,
                       'gpu_reserved_bytes':torch.cuda.memory_reserved() if device=='cuda' else 0}
                history.append(entry);print(json.dumps(entry,allow_nan=False),flush=True)
        model.load_state_dict(best_state,strict=True);return best_step,best_dev
    if job['baseline_sha256'] is None:
        baseline_step,_=train_phase('full',job['baseline_steps'],job['learning_rate'],'baseline')
        # LoRA matrices must start at zero effect for each matched adaptation branch.
        for n,p in model.named_parameters():
            if n.endswith(('.query.B','.value.B')):p.data.zero_()
        save_file({k:v for k,v in cpu_state(model).items() if not adapter_name(k)},str(baseline_file))
    else:baseline_step=None
    baseline_state=cpu_state(model);baseline_dev=predict(model,dev,device);baseline_reg_predictions=predict(model,reg,device)
    baseline_reg=probability_metrics(reg,baseline_reg_predictions['probabilities'])
    selected,selected_metrics=train_phase(job['mode'],job['adaptation_steps'],job['adaptation_learning_rate'],'adaptation')
    final=cpu_state(model)
    if job['mode'] in ('lora','frozen') and any(not torch.equal(final[k],v) for k,v in baseline_state.items() if not adapter_name(k)):raise ValueError('unauthorized frozen backbone mutation')
    save_file({k:v for k,v in final.items() if adapter_name(k)},str(root/'adapter.safetensors'))
    save_file(final,str(root/'candidate.safetensors'))
    candidate_dev=predict(model,dev,device);candidate_reg=predict(model,reg,device)
    inventory={k:{'shape':list(v.shape),'dtype':str(v.dtype),'adapter':adapter_name(k)} for k,v in final.items()}
    write=lambda name,value:(root/name).write_bytes(canonical(value))
    write('model_spec.json',job['model_spec']);write('dataset_manifest.json',job['dataset'])
    write('checkpoint_selection.json',{'policy':job['checkpoint_selection'],'baseline_selected_step':baseline_step,'selected_step':selected,'trained_steps':job['adaptation_steps'],
        'development':selected_metrics,'public_regression':probability_metrics(reg,candidate_reg['probabilities']),'class_weights':weights.cpu().tolist(),
        'uses_protected_holdout':False,'tie_break':'earliest checkpoint (less completed work)'})
    (root/'training_metrics.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in history))
    write('runtime_metrics.json',runtime)
    write('resource_accounting.json',{'steps':completed,'training_tokens':work_tokens,'elapsed_seconds':time.monotonic()-started,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters':sum(p.numel() for p in model.parameters()),'gpu_peak_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'budget_seconds':job['max_seconds'],'budget_steps':total_steps,'financial_billing_cap':False})
    write('tensor_inventory.json',inventory)
    result={'schema':'saal-model-result-v3','job_digest':sha(job),'model_spec_digest':sha(job['model_spec']),'worker_sha256':job['worker_sha256'],
            'runtime':runtime,'development_predictions':{'baseline':baseline_dev,'candidate':candidate_dev},
            'regression_predictions':{'baseline':baseline_reg_predictions,'candidate':candidate_reg},
            'qualification_status':'NOT_EVALUATED','qualification_passed':False,'external_deployment_allowed':False,'model_activation_allowed':False,
            'status':'UNTRUSTED_DEVELOPMENT_RESULT','mode':job['mode'],'backbone_unchanged':all(torch.equal(final[k],v) for k,v in baseline_state.items() if not adapter_name(k))}
    return result

def load_models(bundle,spec,mode):
    root=Path(bundle);baseline=initialized(spec);candidate=initialized(spec)
    b=load_file(str(root/'baseline.safetensors'));c=load_file(str(root/'candidate.safetensors'));a=load_file(str(root/'adapter.safetensors'))
    validate_tensors(b,spec,'backbone');validate_tensors(c,spec);validate_tensors(a,spec,'adapter')
    if any(not torch.equal(a[k],c[k]) for k in a):raise ValueError('adapter/candidate substitution')
    if mode in ('lora','frozen') and any(not torch.equal(b[k],c[k]) for k in b if not adapter_name(k)):raise ValueError('frozen backbone modified')
    if mode=='frozen' and any(not torch.equal(b[k],c[k]) for k in b):raise ValueError('frozen branch modified')
    if mode in ('frozen','full'):
        initial=initialized(spec).state_dict()
        if any(not torch.equal(c[k],initial[k]) for k in a):raise ValueError('non-LoRA branch adapter modified')
    baseline.load_state_dict(b,strict=False);candidate.load_state_dict(c,strict=True)
    return baseline,candidate
