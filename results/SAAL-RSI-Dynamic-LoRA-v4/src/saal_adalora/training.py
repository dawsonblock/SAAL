"""Bounded native training with before/best/terminal evidence and no protected data."""
from pathlib import Path
import copy,json,time,random
from saal_rsi.checkpoint_selection import metrics,selection_key,paired_family_gain
from saal_rsi.pretrained_contract import DECISIONS


def train_backend(backend,dataset,budget):
    import torch
    job=budget['job'];out=Path(budget['out']);out.mkdir(exist_ok=False);started=time.monotonic();deadline=started+job['max_seconds'];rng=random.Random(job['seed'])
    parts={k:[r for r in dataset['records'] if r['split']==k] for k in ('train','development','regression')};kind=backend.spec.model_kind
    device=next(backend.model.parameters()).device;cuda=device.type=='cuda';amp=cuda and backend.spec.precision=='float16'
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=job['initial_loss_scale'],growth_interval=2000);weights=None
    if job['class_balance']:
        counts=[sum(r['expected_decision']==d for r in parts['train']) for d in DECISIONS]
        if not all(counts):raise ValueError('class coverage before updates')
        weights=torch.tensor([len(parts['train'])/(4*n) for n in counts],device=device)
    def check():
        if time.monotonic()>deadline:raise TimeoutError('cumulative admitted training/evaluation deadline')
    def emit(v):
        print(json.dumps(v,allow_nan=False),flush=True)
        with (out/'training_metrics.jsonl').open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
    def evaluate(step):
        check();preds={k:backend.evaluate(parts[k],job['max_new_tokens']) for k in ('development','regression')};check()
        ms={k:metrics(parts[k],preds[k],kind) for k in preds}
        emit({'monitor':'checkpoint_metrics','step':step,'development':ms['development'],'regression':ms['regression']})
        return preds,ms
    initial_predictions,initial_metrics=evaluate(0);backend.export_checkpoint(out/'initial.safetensors');backend.export_checkpoint(out/'best.safetensors')
    snapshot_allocations={'initial':copy.deepcopy(backend.allocation()),'best':copy.deepcopy(backend.allocation())}
    selected_step=0;best_metrics=initial_metrics;best_predictions=initial_predictions;history=[{'step':0,'metrics':initial_metrics}];tokens=0
    optimizer=torch.optim.AdamW(list(backend.trainable.values()),lr=job['learning_rate']) if backend.trainable else None
    # Bounded forward/backward preflight is the first admitted update, not an unaccounted extra step.
    for step in range(1,job['steps']+1):
        check();backend.model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0
        for micro in range(job['accumulation']):
            batch=[rng.choice(parts['train']) for _ in range(job['batch_size'])]
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):loss,work=backend.batch_loss(batch,weights)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss; no implicit retry')
            loss_sum+=float(loss.detach());tokens+=work;scaler.scale(loss/job['accumulation']).backward();check()
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(list(backend.trainable.values()),1.0)
        if not torch.isfinite(norm):raise ValueError('nonfinite gradients; interrupted attempt')
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<old_scale:raise ValueError('gradient overflow; no silently skipped update')
        allocation=backend.allocate(step)
        if step%10==0 or step==1:emit({'monitor':'dynamic_rank','step':step,**allocation})
        emit({'monitor':'training_progress','completed_steps':step,'total_steps':job['steps'],'objective_loss':loss_sum/job['accumulation'],'tokens':tokens,'elapsed_seconds':time.monotonic()-started,'gpu_allocated_bytes':torch.cuda.memory_allocated() if cuda else 0})
        if step%job['checkpoint_interval']==0 or step==job['steps']:
            preds,ms=evaluate(step);history.append({'step':step,'metrics':ms})
            if selection_key(ms['development'])>selection_key(best_metrics['development']):
                selected_step=step;best_metrics=ms;best_predictions=preds;backend.export_checkpoint(out/'best.safetensors');snapshot_allocations['best']=copy.deepcopy(backend.allocation())
    backend.export_checkpoint(out/'terminal.safetensors');snapshot_allocations['terminal']=copy.deepcopy(backend.allocation());terminal_predictions,terminal_metrics=(preds,ms) if job['steps'] else (initial_predictions,initial_metrics)
    updated=backend.trainable_hash()!=backend.initial_trainable_hash;freeze=backend.verify_frozen_weights();backend.restore_checkpoint(out/'best.safetensors');reloaded=backend.evaluate(parts['development'],job['max_new_tokens'])
    if kind=='modernbert-seqcls-v1':
        for a,b in zip(reloaded,best_predictions['development']):
            if a['decision']!=b['decision'] or max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))>1e-4:raise ValueError('checkpoint restore prediction parity')
    elif reloaded!=best_predictions['development']:raise ValueError('generation changed after restore')
    return {'snapshots':{'initial':{'step':0,'metrics':initial_metrics,'predictions':initial_predictions},'best':{'step':selected_step,'metrics':best_metrics,'predictions':best_predictions},'terminal':{'step':job['steps'],'metrics':terminal_metrics,'predictions':terminal_predictions}},
      'selection_history':history,'selection_policy':job['selection_policy'],'real_parameter_updates':updated,'frozen_integrity':freeze,'reload_parity':True,
      'adaptation_gain_best':paired_family_gain(initial_metrics['development'],best_metrics['development']),'adaptation_gain_terminal':paired_family_gain(initial_metrics['development'],terminal_metrics['development']),
      'snapshot_allocations':snapshot_allocations,'resource_accounting':{'steps':job['steps'],'microbatches':job['steps']*job['accumulation'],'tokens':tokens,'elapsed_seconds':time.monotonic()-started,**backend.report_resources()},'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
