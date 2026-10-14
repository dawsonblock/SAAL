"""Versioned optimizer, full coverage and resubstitution diagnostics.

The historical native selector stays safety-first. Training measurements never
select a checkpoint; public regression never selects one either.
"""
from pathlib import Path
import json,time
from .epoch_sampler import manifests
from .epoch_metrics import diagnostics
from .checkpoint_selection import selection_key,paired_family_gain
from .pretrained_contract import DECISIONS
from .spec import canonical

def head_parameter(name):
    return name.startswith(('head.','classifier.')) or '.head.modules_to_save.default.' in name or '.classifier.modules_to_save.default.' in name

def train(backend,job,out,*,deadline=None):
    import torch
    out=Path(out);out.mkdir(exist_ok=False);started=time.monotonic();deadline=deadline if deadline is not None else started+job['max_seconds']
    p=job['training_policy'];sample,diag=manifests(job);by_id={r['record_id']:r for r in job['dataset']['records']}
    parts={s:[r for r in by_id.values() if r['split']==s] for s in ('train','development','regression')};evaluated=parts|{'train':[by_id[i] for i in diag['record_ids']]}
    device=next(backend.model.parameters()).device;tokens=0;actual_ids=[]
    def check():
        if time.monotonic()>deadline:raise TimeoutError('admitted cumulative optimizer/evaluation budget')
    def emit(value):
        print(json.dumps(value,allow_nan=False),flush=True)
        with (out/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    def evaluate(step):
        check();pred={s:backend.evaluate(rows) for s,rows in evaluated.items()};check();m={s:diagnostics(evaluated[s],v) for s,v in pred.items()}
        emit({'monitor':'checkpoint_metrics','step':step,'train':m['train'],'development':m['development'],'regression':m['regression']});return pred,m
    def all_train():
        check();pred=backend.evaluate(parts['train']);check();return {'predictions':pred,'metrics':diagnostics(parts['train'],pred)}
    initial_predictions,initial_metrics=evaluate(0);full_initial=all_train()
    backend.export_checkpoint(out/'initial.safetensors');backend.export_checkpoint(out/'best.safetensors')
    best_step=0;best_metrics=initial_metrics;best_predictions=initial_predictions;history=[{'step':0,'metrics':initial_metrics}]
    head=[v for n,v in backend.trainable.items() if head_parameter(n)];encoder=[v for n,v in backend.trainable.items() if not head_parameter(n)]
    groups=[]
    if head:groups.append({'params':head,'lr':p['head_learning_rate'],'name':'head'})
    if encoder:groups.append({'params':encoder,'lr':p['encoder_learning_rate'],'name':'encoder'})
    optimizer=torch.optim.AdamW(groups)
    weights=None
    if job['class_balance']:
        counts=[sum(r['expected_decision']==d for r in parts['train']) for d in DECISIONS]
        weights=torch.tensor([len(parts['train'])/(4*n) for n in counts],device=device)
    warm_initial=backend.tensor_hash((n,v) for n,v in backend.trainable.items() if not head_parameter(n)) if p['head_warmup_steps'] else None
    warm_report={'steps':p['head_warmup_steps'],'encoder_unchanged':None,'initial_encoder_hash':warm_initial,'final_encoder_hash':None}
    offset=0
    for step in range(1,job['steps']+1):
        check();backend.model.train();optimizer.zero_grad(set_to_none=True);objective=0.0
        for _ in range(job['accumulation']):
            ids=sample['record_ids'][offset:offset+job['batch_size']];offset+=job['batch_size'];batch=[by_id[i] for i in ids];actual_ids.extend(ids)
            loss,work=backend.batch_loss(batch,weights)
            if not torch.isfinite(loss):raise ValueError('nonfinite objective; no automatic retry')
            objective+=float(loss.detach());tokens+=work;(loss/job['accumulation']).backward();check()
        if step<=p['head_warmup_steps']:
            for parameter in encoder:parameter.grad=None
        norm=torch.nn.utils.clip_grad_norm_(list(backend.trainable.values()),1.0)
        if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
        optimizer.step()
        if step==p['head_warmup_steps']:
            final=backend.tensor_hash((n,v) for n,v in backend.trainable.items() if not head_parameter(n));warm_report.update(final_encoder_hash=final,encoder_unchanged=final==warm_initial)
            if final!=warm_initial:raise ValueError('encoder changed during head-only warmup')
        emit({'monitor':'training_progress','completed_steps':step,'total_steps':job['steps'],'objective_loss':objective/job['accumulation'],'gradient_norm':float(norm),'optimizer_groups':{g['name']:g['lr'] for g in optimizer.param_groups},'tokens':tokens,'elapsed_seconds':time.monotonic()-started,'gpu_allocated_bytes':torch.cuda.memory_allocated() if device.type=='cuda' else 0})
        if step%job['checkpoint_interval']==0 or step==job['steps']:
            predictions,measures=evaluate(step);history.append({'step':step,'metrics':measures})
            if selection_key(measures['development'])>selection_key(best_metrics['development']):
                best_step=step;best_metrics=measures;best_predictions=predictions;backend.export_checkpoint(out/'best.safetensors')
    if actual_ids!=sample['record_ids']:raise ValueError('actual sample stream deviates from admission')
    full_terminal=all_train();backend.export_checkpoint(out/'terminal.safetensors');updated=backend.trainable_hash()!=backend.initial_trainable_hash;frozen=backend.verify_frozen_weights()
    backend.restore_checkpoint(out/'best.safetensors');fresh=backend.evaluate(parts['development'])
    from .epoch_metrics import prediction_equal
    if not prediction_equal(fresh,best_predictions['development'],'modernbert-seqcls-v1',1e-4):raise ValueError('same-runtime reload parity')
    (out/'sampler_manifest.json').write_bytes(canonical(sample));(out/'diagnostic_manifest.json').write_bytes(canonical(diag))
    return {'snapshots':{'initial':{'step':0,'predictions':initial_predictions,'metrics':initial_metrics},'best':{'step':best_step,'predictions':best_predictions,'metrics':best_metrics},'terminal':{'step':job['steps'],'predictions':predictions,'metrics':measures}},'full_training_evaluation':{'initial':full_initial,'terminal':full_terminal},'selection_history':history,'selection_policy':job['selection_policy'],'real_parameter_updates':updated,'frozen_integrity':frozen,'reload_parity':True,'warmup_evidence':warm_report,'adaptation_gain_best':paired_family_gain(initial_metrics['development'],best_metrics['development']),'adaptation_gain_terminal':paired_family_gain(initial_metrics['development'],measures['development']),'resource_accounting':{'steps':job['steps'],'microbatches':job['steps']*job['accumulation'],'sample_exposures':len(actual_ids),'unique_training_records':len(set(actual_ids)),'tokens':tokens,'elapsed_seconds':time.monotonic()-started,**backend.report_resources()},'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
