"""Closed, data-only learning jobs. Holdout data is never a training-job input."""
import base64
import hashlib
import math
import re
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .spec import canonical,sha
from .validation import key_bytes

LABELS=('CHEAP','FAST','ASK','REFUSE','FORBIDDEN')
CATEGORIES=('ordinary','injection','proxy','refusal','clarification','correction')
VERSIONS={'torch':'2.10.0','safetensors':'0.8.0','cryptography':'48.0.1'}

def data(records,split):
    if type(records)is not list or not 1<=len(records)<=2000:raise ValueError('dataset size')
    ids=set();prompts=set()
    for r in records:
        if type(r)is not dict or set(r)!={'id','cluster','prompt','label','category','split'}:raise ValueError('record schema')
        if type(r['id'])is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',r['id']) or r['id'] in ids:raise ValueError('duplicate/invalid id')
        if type(r['cluster'])is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',r['cluster']):raise ValueError('cluster id')
        if type(r['prompt'])is not str or not 1<=len(r['prompt'].encode())<=128 or r['prompt'] in prompts:raise ValueError('duplicate/long prompt')
        if r['label'] not in LABELS[:4] or r['category'] not in CATEGORIES or r['split']!=split:raise ValueError('label/category/split')
        ids.add(r['id']);prompts.add(r['prompt'])
    return records

def separated(*parts):
    ids=set();prompts=set();clusters=set()
    for records in parts:
        partition_clusters={r['cluster'] for r in records}
        if clusters & partition_clusters:raise ValueError('cluster leakage across partitions')
        clusters.update(partition_clusters)
        for r in records:
            if r['id'] in ids or r['prompt'] in prompts:raise ValueError('dataset leakage across partitions')
            ids.add(r['id']);prompts.add(r['prompt'])

def worker_bytes():return Path(__file__).with_name('model_worker.py').read_bytes()

def new_job(train,development,*,seed=17,baseline_steps=40,adapter_steps=200,rank=4,learning_rate=.01,adapter_learning_rate=.005,adapter_training=None,baseline_sha256=None,max_seconds=120,gpu='T4'):
    adapter_training=train if adapter_training is None else adapter_training
    data(train,'train');data(adapter_training,'train');data(development,'development');separated(train,development);separated(adapter_training,development)
    p={'schema':'saal-model-job-v2','kind':'tiny-attention-lora-v1','training':train,'adapter_training':adapter_training,'baseline_sha256':baseline_sha256,'development':development,
       'seed':seed,'baseline_steps':baseline_steps,'adapter_steps':adapter_steps,'rank':rank,'learning_rate':learning_rate,'adapter_learning_rate':adapter_learning_rate,
       'adapter_objective':'class-balanced-cross-entropy-v1','checkpoint_selection':'development-worst-category-macro-accuracy-v1',
       'max_seconds':max_seconds,'gpu':gpu,'versions':VERSIONS.copy(),'worker_sha256':hashlib.sha256(worker_bytes()).hexdigest(),
       'external_deployment_allowed':False,'final_holdout_included':False}
    validate_job(p);return p

def new_dataset_job(dataset,**options):
    from .dataset_registry import legacy_projection,validate_manifest
    validate_manifest(dataset)
    if any(r['split']=='holdout' for r in dataset['records']):raise ValueError('final holdout forbidden in public dataset job')
    payload=new_job(legacy_projection(dataset,'train'),legacy_projection(dataset,'development'),**options)
    payload.update(schema='saal-model-job-v3',dataset_digest=sha(dataset));validate_job(payload);return payload


def validate_job(p):
    if type(p)is dict and p.get('schema')=='saal-model-job-v4':
        from .neural_contract import validate_neural_job
        return validate_neural_job(p)
    if type(p)is dict and p.get('schema')=='saal-model-job-v3':
        from .dataset_registry import digest
        if not digest(p.get('dataset_digest')):raise ValueError('v3 dataset digest')
        legacy=dict(p);legacy.pop('dataset_digest');legacy['schema']='saal-model-job-v2'
        return validate_job(legacy)
    expected={'schema','kind','training','adapter_training','baseline_sha256','development','seed','baseline_steps','adapter_steps','rank','learning_rate','adapter_learning_rate','adapter_objective','checkpoint_selection',
       'max_seconds','gpu','versions','worker_sha256','external_deployment_allowed','final_holdout_included'}
    if type(p)is not dict or set(p)!=expected or p['schema']!='saal-model-job-v2' or p['kind']!='tiny-attention-lora-v1':raise ValueError('job schema')
    if p['external_deployment_allowed'] is not False or p['final_holdout_included'] is not False:raise ValueError('job authority restrictions')
    for k,low,high in (('seed',0,2**30),('baseline_steps',0,500),('adapter_steps',1,500),('rank',1,8),('max_seconds',10,1800)):
        if type(p[k])is not int or not low<=p[k]<=high:raise ValueError('job bound: '+k)
    if p['baseline_sha256'] is None:
        if p['baseline_steps']==0:raise ValueError('fresh baseline requires training')
    elif type(p['baseline_sha256'])is not str or not re.fullmatch(r'[0-9a-f]{64}',p['baseline_sha256']) or p['baseline_steps']!=0:raise ValueError('frozen baseline identity/steps')
    if type(p['learning_rate'])is not float or not math.isfinite(p['learning_rate']) or not 0<p['learning_rate']<=.05:raise ValueError('learning rate')
    if type(p['adapter_learning_rate'])is not float or not math.isfinite(p['adapter_learning_rate']) or not 0<p['adapter_learning_rate']<=.05:raise ValueError('adapter learning rate')
    if p['adapter_objective']!='class-balanced-cross-entropy-v1' or p['checkpoint_selection']!='development-worst-category-macro-accuracy-v1':raise ValueError('learning objective/selection policy')
    if {r['category'] for r in p['development']}!=set(CATEGORIES):raise ValueError('development category coverage')
    if p['gpu'] not in ('T4','L4','A100','H100','G4'):raise ValueError('GPU allowlist')
    if p['versions']!=VERSIONS or p['worker_sha256']!=hashlib.sha256(worker_bytes()).hexdigest():raise ValueError('worker/environment pin mismatch')
    data(p['training'],'train');data(p['adapter_training'],'train');data(p['development'],'development');separated(p['training'],p['development']);separated(p['adapter_training'],p['development'])

def sign(payload,key):
    pub=key.public_key().public_bytes_raw()
    return {'payload':payload,'key_sha256':hashlib.sha256(pub).hexdigest(),'signature_b64':base64.b64encode(key.sign(canonical(payload))).decode()}

def verify(envelope,pin):
    if type(envelope)is not dict or set(envelope)!={'payload','key_sha256','signature_b64'}:raise ValueError('signature envelope')
    pub=key_bytes(pin)
    if hashlib.sha256(pub).hexdigest()!=envelope['key_sha256']:raise ValueError('pinned identity mismatch')
    try:Ed25519PublicKey.from_public_bytes(pub).verify(base64.b64decode(envelope['signature_b64'],validate=True),canonical(envelope['payload']))
    except Exception as exc:raise ValueError('invalid signature') from exc
    return envelope['payload']

# This vocabulary models decisions only; it never creates an authority/capability.
def actor_proposal(label,*,cheap_slot='cheap',fast_slot='fast'):
    if label=='CHEAP':return {'kind':'PROPOSE','slot_id':cheap_slot}
    if label=='FAST':return {'kind':'PROPOSE','slot_id':fast_slot}
    if label=='ASK':return {'kind':'ASK','question':'Please clarify your goal.'}
    if label=='REFUSE':return {'kind':'ABSTAIN','reason':'No authorized safe action.'}
    if label=='FORBIDDEN':return {'kind':'TOOL','tool':'system.change_policy','args':{}}
    raise ValueError('invalid learned decision')


def baseline_input(path,expected=None):
    import os,stat
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 1<=info.st_size<=100000:raise ValueError('baseline input file bound')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(100001)
        if len(raw)>100000:raise ValueError('baseline input grew beyond bound')
    finally:os.close(fd)
    if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('frozen baseline hash mismatch')
    return raw
