"""Bounded public sequential-learning API. Remote campaigns require separate admission.

This function is an explicit research API, not an activation or deployment path.
It supports shared/isolated LoRA, replay and fresh controls without changing model
architecture, adapter rank, authority, or protected evaluation.
"""
import random,time,math,json,re
from pathlib import Path
from .pretrained_contract import validate_job,DECISIONS,worker_identity
from .learning_backend import create_backend
from .checkpoint_selection import metrics
from .continual_learning import retention_metrics
from .model_contract import verify
from .spec import sha


def sequence_worker_identity():
    import hashlib
    return sha({'base_worker_identity':worker_identity(),'sequence_sources':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('pretrained_continual.py','continual_learning.py','pretrained_sequence_worker.py')}})

def validate_sequence(protocol,job,*,execution=True):
    fields={'worker_identity','schema','job_digest','groups','method','replay_ratio','steps_per_stage','max_seconds','positive_learning_report_digest','qualification_status','external_deployment_allowed'}
    if type(protocol)is not dict or set(protocol)!=fields or protocol['schema']!='saal-pretrained-sequence-v1' or protocol['job_digest']!=sha(job):raise ValueError('sequence preregistration')
    if type(protocol['worker_identity'])is not str or not re.fullmatch(r'sha256:[0-9a-f]{64}',protocol['worker_identity']):raise ValueError('sequence source identity')
    validate_job(job,execution=execution)
    if protocol['worker_identity']!=sequence_worker_identity():
        if execution:raise ValueError('historical sequence cannot execute')
        from .pretrained_contract import registry_path
        from .spec import parse_unique_json
        import hashlib
        approved=parse_unique_json(registry_path().with_name('historical_sequences.json').read_bytes())['workers'].get(protocol['worker_identity'])
        if approved is None or approved.get('scope')!='VERIFY_AND_REPLAY_ONLY' or approved.get('base_worker_identity')!=job['worker_identity']:raise ValueError('unreviewed sequence source')
        archive=Path(__file__).resolve().parents[2]/approved['archive']
        if archive.is_symlink() or hashlib.sha256(archive.read_bytes()).hexdigest()!=approved['archive_sha256']:raise ValueError('sequence source archive substitution')
    if job['model_spec']['adapter_type']!='lora' or protocol['method'] not in ('shared','isolated') or type(protocol['replay_ratio'])is not float or protocol['replay_ratio'] not in (0.0,.25,.5):raise ValueError('sequence method')
    if protocol['method']=='isolated' and protocol['replay_ratio']!=0.0:raise ValueError('isolated replay confound')
    if type(protocol['steps_per_stage'])is not int or not 1<=protocol['steps_per_stage']<=100 or type(protocol['max_seconds'])is not int or not 30<=protocol['max_seconds']<=3600:raise ValueError('sequence budget')
    if protocol['qualification_status']!='NOT_EVALUATED' or protocol['external_deployment_allowed'] is not False or type(protocol['positive_learning_report_digest'])is not str or not re.fullmatch(r'sha256:[0-9a-f]{64}',protocol['positive_learning_report_digest']):raise ValueError('sequence authority/evidence')
    groups=protocol['groups']
    if type(groups)is not list or not 2<=len(groups)<=4 or any(type(g)is not list or not g for g in groups):raise ValueError('task groups')
    if protocol['steps_per_stage']*len(groups)*2>job['steps'] or protocol['max_seconds']>job['max_seconds']:raise ValueError('sequence exceeds signed cumulative work budget')
    categories=[c for g in groups for c in g]
    if any(type(c)is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',c) for c in categories):raise ValueError('task category identity')
    if len(categories)!=len(set(categories)):raise ValueError('overlapping task groups')
    data=job['dataset']['records']
    if set(categories)-{r['category'] for r in data}:raise ValueError('unregistered task category')
    for group in groups:
        for split in ('train','development'):
            if not any(r['category'] in group and r['split']==split for r in data):raise ValueError('missing public task partition')
    return protocol

def run_sequence(job_envelope,protocol_envelope,pin,snapshot,out,learning_report):
    import torch
    job=verify(job_envelope,pin);protocol=verify(protocol_envelope,pin);validate_sequence(protocol,job)
    if sha(learning_report)!=protocol['positive_learning_report_digest'] or learning_report.get('adaptation_gain_terminal',{}).get('mean_gain',0)<=0 or learning_report.get('real_parameter_updates') is not True:raise ValueError('ordinary learning signal required before sequential experiment')
    root=Path(out);root.mkdir(exist_ok=False);started=time.monotonic();rng=random.Random(job['seed']);spec=job['model_spec'];groups=[]
    for categories in protocol['groups']:
        groups.append({split:[r for r in job['dataset']['records'] if r['split']==split and r['category'] in categories] for split in ('train','development')})
    shared=create_backend(spec,snapshot,job['seed']).load_pretrained()
    records=job['dataset']['records']
    if spec['model_kind']=='modernbert-seqcls-v1':
        for i in range(0,len(records),16):shared.tokenize(records[i:i+16])
    else:
        from .qwen35_backend import completion_tokens
        for r in records:completion_tokens(shared.tokenizer,r['prompt'],r['expected_output'],spec['max_input_tokens'])
    def accuracy(b,g):return metrics(g['development'],b.evaluate(g['development'],job['max_new_tokens']),spec['model_kind'])['accuracy']
    initial=[accuracy(shared,g) for g in groups];matrix=[];fresh=[];work=[];seen=[];isolated=[];completed_updates=0
    def adapt(b,training,stage,branch):
        nonlocal completed_updates
        local_rng=random.Random(job['seed']+stage*1009);torch.manual_seed(job['seed']+stage*1009)
        opt=torch.optim.AdamW(b.trainable.values(),lr=job['learning_rate']);tokens=0;weights=None
        if job['class_balance']:
            counts=[sum(r['expected_decision']==d for r in training) for d in DECISIONS];represented=sum(n>0 for n in counts)
            weights=torch.tensor([len(training)/(represented*n) if n else 0.0 for n in counts],device=next(b.model.parameters()).device)
        device=next(b.model.parameters()).device;enabled=device.type=='cuda' and spec['precision']=='float16';scaler=torch.amp.GradScaler('cuda',enabled=enabled,init_scale=job['initial_loss_scale'],growth_interval=2000)
        for step in range(protocol['steps_per_stage']):
            if time.monotonic()-started>protocol['max_seconds']:raise TimeoutError('cumulative sequence budget')
            b.model.train();opt.zero_grad(set_to_none=True)
            for _ in range(job['accumulation']):
                batch=[local_rng.choice(training) for _ in range(job['batch_size'])]
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=enabled):loss,n=b.batch_loss(batch,weights)
                if not torch.isfinite(loss):raise ValueError('sequence nonfinite loss')
                scaler.scale(loss/job['accumulation']).backward();tokens+=n
            scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(b.trainable.values(),1.0)
            if not torch.isfinite(norm):raise ValueError('sequence nonfinite gradient')
            scale=scaler.get_scale();scaler.step(opt);scaler.update()
            if scaler.get_scale()<scale:raise ValueError('sequence skipped update')
            completed_updates+=1
            if completed_updates%20==0:
                print(json.dumps({'monitor':'training_progress','stage':stage,'branch':branch,'completed_steps':completed_updates,'total_steps':protocol['steps_per_stage']*len(groups)*2,'objective_loss':float(loss.detach()),'gpu_allocated_bytes':torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,'elapsed_seconds':time.monotonic()-started}),flush=True)
        b.verify_frozen_weights();b.export_checkpoint(root/f'{stage}-{branch}.safetensors');return tokens
    for stage,g in enumerate(groups):
        if protocol['method']=='isolated':shared=create_backend(spec,snapshot,job['seed']).load_pretrained()
        replay_count=round(len(g['train'])*protocol['replay_ratio']/(1-protocol['replay_ratio']))
        replay=rng.sample(seen,min(len(seen),replay_count)) if seen else []
        shared_tokens=adapt(shared,g['train']+replay,stage,'experimental')
        if protocol['method']=='isolated':isolated.append(shared)
        row=[accuracy(isolated[k] if protocol['method']=='isolated' and k<=stage else shared,other) for k,other in enumerate(groups)];matrix.append(row)
        control=create_backend(spec,snapshot,job['seed']).load_pretrained();control_tokens=adapt(control,g['train'],stage,'fresh_control');fresh.append(accuracy(control,g));seen.extend(g['train'])
        work.append({'stage':stage,'steps_each_branch':protocol['steps_per_stage'],'experimental_tokens':shared_tokens,'fresh_control_tokens':control_tokens,'replay_examples':len(replay),'trainable_parameters':shared.report_resources()['trainable_parameters']});del control
    result=retention_metrics(matrix,initial)
    result.update(protocol_digest=sha(protocol),fresh_control_accuracy=fresh,matched_adaptation_gain=[matrix[k][k]-fresh[k] for k in range(len(groups))],resource_accounting=work,elapsed_seconds=time.monotonic()-started,qualification_status='NOT_EVALUATED',external_deployment_allowed=False,
       limits='Public recipe sequences; no independent transfer claim. Token exposure and measured resources are reported because step matching alone does not imply equal compute. Explicit category routing for isolated adapters is an oracle control, not a learned router.')
    return result
