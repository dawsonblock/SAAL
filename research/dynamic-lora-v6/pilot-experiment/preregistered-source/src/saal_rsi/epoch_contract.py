"""Closed v6 diagnostic jobs; immutable v5 jobs are never reinterpreted."""
import hashlib, math
from pathlib import Path
from .spec import sha
from .pretrained_contract import SOURCES as BASE_SOURCES, worker_identity as base_identity, validate_job as base_validate, new_job as base_new

EXTRA_SOURCES=('epoch_contract.py','epoch_sampler.py','epoch_metrics.py','epoch_training.py','epoch_artifacts.py','epoch_worker.py')
SOURCES=BASE_SOURCES+EXTRA_SOURCES
POLICY_FIELDS={'sampler','diagnostic_seed','diagnostic_per_class','head_learning_rate','encoder_learning_rate','head_warmup_steps'}

def worker_identity():
    return sha({'base_worker':base_identity(),'epoch_sources':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in EXTRA_SOURCES}})

def foundation(job):
    # Validation-only projection. It is never signed, exported, or executed as v5.
    p={k:v for k,v in job.items() if k not in ('training_policy','foundation_worker_identity')}
    p['schema']='saal-model-job-v5';p['worker_identity']=job['foundation_worker_identity']
    return p

def validate_job(job,*,execution=True):
    if type(job)is not dict or job.get('schema')!='saal-model-job-v6':raise ValueError('epoch job v6 required')
    if job.get('foundation_worker_identity')!=base_identity():raise ValueError('foundation substitution')
    base_validate(foundation(job),execution=execution)
    if job['worker_identity']!=worker_identity():raise ValueError('epoch worker substitution')
    s=job['model_spec'];p=job['training_policy']
    if s['model_kind']!='modernbert-seqcls-v1' or s['precision']!='float32' or s['adapter_type'] not in ('lora','full','head'):raise ValueError('epoch classifier-only architecture/method')
    if type(p)is not dict or set(p)!=POLICY_FIELDS:raise ValueError('closed training policy')
    if p['sampler'] not in ('uniform-v1','complete-balanced-epochs-v1'):raise ValueError('unadmitted sampler')
    if type(p['diagnostic_seed'])is not int or not 0<=p['diagnostic_seed']<2**31 or type(p['diagnostic_per_class'])is not int or not 1<=p['diagnostic_per_class']<=128:raise ValueError('diagnostic budget')
    for key in ('head_learning_rate','encoder_learning_rate'):
        if type(p[key])is not float or not math.isfinite(p[key]) or not 1e-7<=p[key]<=.01:raise ValueError('bounded optimizer rate')
    if type(p['head_warmup_steps'])is not int or not 0<=p['head_warmup_steps']<job['steps']:raise ValueError('warmup budget')
    if s['adapter_type']!='full' and p['head_warmup_steps']:raise ValueError('full-only encoder warmup')
    if p['sampler']=='complete-balanced-epochs-v1':
        from .epoch_sampler import balanced_counts
        balanced_counts([r for r in job['dataset']['records'] if r['split']=='train'])
        if job['batch_size']%4:raise ValueError('balanced batch needs all four classes')
    return job

def new_job(spec,dataset,*,sampler='complete-balanced-epochs-v1',diagnostic_seed=730,diagnostic_per_class=128,head_learning_rate=None,encoder_learning_rate=None,head_warmup_steps=0,**kwargs):
    p=base_new(spec,dataset,**kwargs)
    p.update(schema='saal-model-job-v6',foundation_worker_identity=base_identity(),worker_identity=worker_identity(),training_policy={'sampler':sampler,'diagnostic_seed':diagnostic_seed,'diagnostic_per_class':diagnostic_per_class,'head_learning_rate':head_learning_rate if head_learning_rate is not None else p['learning_rate'],'encoder_learning_rate':encoder_learning_rate if encoder_learning_rate is not None else p['learning_rate'],'head_warmup_steps':head_warmup_steps})
    return validate_job(p)
