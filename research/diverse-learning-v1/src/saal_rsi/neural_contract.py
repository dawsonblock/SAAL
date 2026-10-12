"""Versioned, closed contracts for the four-class position-aware research model."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import math
import re
from .spec import sha
from .model_contract import VERSIONS
from .dataset_registry import validate_manifest

DECISIONS=('CHEAP','FAST','ASK','REFUSE')
SELECTION='public-critical-class-family-pair-cost-v1'
SOURCES=('spec.py','validation.py','model_contract.py','dataset_registry.py',
         'neural_contract.py','model_architectures.py','neural_training.py','neural_worker.py')

def worker_sources():
    return {name:Path(__file__).with_name(name).read_bytes() for name in SOURCES}

def worker_digest():
    return sha({k:hashlib.sha256(v).hexdigest() for k,v in worker_sources().items()})

@dataclass(frozen=True)
class ModelSpec:
    model_kind:str='tiny-transformer-lora-v2'
    architecture_version:str='byte-cls-transformer-v2'
    tokenizer_version:str='utf8-bytes-pad256-cls257-v2'
    max_sequence_bytes:int=256
    hidden_dimension:int=64
    attention_heads:int=4
    encoder_layers:int=2
    feedforward_dimension:int=256
    positional_encoding:str='learned-absolute-v1'
    output_vocabulary:tuple=DECISIONS
    adapter_configuration:dict|None=None
    initialization_seed:int=17
    dropout:float=.1
    def as_dict(self):
        value=asdict(self);value['output_vocabulary']=list(self.output_vocabulary)
        if value['adapter_configuration'] is None:
            value['adapter_configuration']={'rank':4,'targets':['query','value'],'scale':1.0}
        validate_spec(value);return value

def validate_spec(s):
    if type(s)is not dict or set(s)!=set(ModelSpec.__dataclass_fields__):raise ValueError('ModelSpec inventory')
    fixed={'model_kind':'tiny-transformer-lora-v2','architecture_version':'byte-cls-transformer-v2',
           'tokenizer_version':'utf8-bytes-pad256-cls257-v2','positional_encoding':'learned-absolute-v1',
           'output_vocabulary':list(DECISIONS)}
    if any(s[k]!=v for k,v in fixed.items()):raise ValueError('architecture/tokenizer/decision allowlist')
    for k,allowed in [('max_sequence_bytes',(128,256)),('hidden_dimension',(32,64)),('attention_heads',(4,)),('encoder_layers',(1,2)),('feedforward_dimension',(128,256))]:
        if type(s[k])is not int or s[k] not in allowed:raise ValueError('model dimension: '+k)
    if type(s['initialization_seed'])is not int or not 0<=s['initialization_seed']<=2**30:raise ValueError('initialization seed')
    if type(s['dropout'])is not float or s['dropout'] not in (0.0,.1):raise ValueError('dropout')
    a=s['adapter_configuration']
    if type(a)is not dict or set(a)!={'rank','targets','scale'} or type(a['rank'])is not int or a['rank'] not in (2,4,8) or a['targets']!=['query','value'] or type(a['scale'])is not float or a['scale']!=1.0:raise ValueError('adapter specification')
    return s

def rows(dataset,split):return [r for r in dataset['records'] if r['split']==split]

def new_transformer_job(dataset,*,model_spec=None,seed=17,mode='lora',baseline_steps=200,
                        adaptation_steps=200,baseline_sha256=None,learning_rate=.001,
                        adaptation_learning_rate=.001,batch_size=32,class_balance='weighted',
                        ablation=False,max_seconds=600,gpu='T4',minimum_free_memory_mb=1024):
    spec=ModelSpec(initialization_seed=seed).as_dict() if model_spec is None else model_spec
    p={'schema':'saal-model-job-v4','kind':'tiny-transformer-lora-v2','model_spec':spec,'dataset':dataset,
       'dataset_digest':sha(dataset),'seed':seed,'mode':mode,'baseline_sha256':baseline_sha256,
       'baseline_steps':baseline_steps,'adaptation_steps':adaptation_steps,'learning_rate':learning_rate,
       'adaptation_learning_rate':adaptation_learning_rate,'batch_size':batch_size,'class_balance':class_balance,
       'declared_class_ablation':ablation,'checkpoint_selection':SELECTION,'max_seconds':max_seconds,
       'gpu':gpu,'minimum_free_memory_mb':minimum_free_memory_mb,'versions':VERSIONS.copy(),
       'serialization_numpy_version':'2.4.6','worker_sha256':worker_digest(),
       'dependency_lock_sha256':hashlib.sha256(Path(__file__).resolve().parents[2].joinpath('configs/dependency-locks/linux-x86_64-py313.txt').read_bytes()).hexdigest(),
       'external_deployment_allowed':False,'final_holdout_included':False}
    validate_neural_job(p);return p

def validate_neural_job(p):
    fields={'schema','kind','model_spec','dataset','dataset_digest','seed','mode','baseline_sha256',
            'baseline_steps','adaptation_steps','learning_rate','adaptation_learning_rate','batch_size',
            'class_balance','declared_class_ablation','checkpoint_selection','max_seconds','gpu',
            'minimum_free_memory_mb','versions','serialization_numpy_version','worker_sha256',
            'dependency_lock_sha256','external_deployment_allowed','final_holdout_included'}
    if type(p)is not dict or set(p)!=fields or p['schema']!='saal-model-job-v4' or p['kind']!='tiny-transformer-lora-v2':raise ValueError('job-v4 inventory')
    validate_spec(p['model_spec']);validate_manifest(p['dataset'])
    if p['dataset_digest']!=sha(p['dataset']) or p['worker_sha256']!=worker_digest():raise ValueError('data/source substitution')
    if p['dataset']['origin']=='custodian-sequestered' or any(r['split']=='holdout' for r in p['dataset']['records']):raise ValueError('protected data forbidden in training')
    if p['versions']!=VERSIONS or p['serialization_numpy_version']!='2.4.6':raise ValueError('dependency identity')
    if type(p['dependency_lock_sha256'])is not str or not re.fullmatch('[0-9a-f]{64}',p['dependency_lock_sha256']):raise ValueError('dependency lock identity')
    if p['external_deployment_allowed'] is not False or p['final_holdout_included'] is not False:raise ValueError('job authority restrictions')
    if p['mode'] not in ('frozen','lora','full') or p['class_balance'] not in ('none','weighted','sampling') or p['checkpoint_selection']!=SELECTION or type(p['declared_class_ablation'])is not bool:raise ValueError('training policy')
    for key,low,high in [('seed',0,2**30),('baseline_steps',0,2000),('adaptation_steps',0,2000),('batch_size',1,64),('max_seconds',10,3600),('minimum_free_memory_mb',512,8192)]:
        if type(p[key])is not int or not low<=p[key]<=high:raise ValueError('job bound: '+key)
    if p['seed']!=p['model_spec']['initialization_seed']:raise ValueError('initialization identity')
    if p['baseline_sha256'] is None:
        if p['baseline_steps']==0:raise ValueError('fresh backbone requires supervised training')
    elif type(p['baseline_sha256'])is not str or not re.fullmatch('[0-9a-f]{64}',p['baseline_sha256']) or p['baseline_steps']!=0:raise ValueError('frozen baseline binding')
    if (p['mode']=='frozen') != (p['adaptation_steps']==0):raise ValueError('mode/step mismatch')
    if p['gpu'] not in ('T4','L4','A100','H100','G4'):raise ValueError('GPU allowlist')
    for k in ('learning_rate','adaptation_learning_rate'):
        if type(p[k])is not float or not math.isfinite(p[k]) or not 0<p[k]<=.01:raise ValueError('optimizer bound')
    limit=p['model_spec']['max_sequence_bytes']
    if any(not 1<=len(r['prompt'].encode('utf-8'))<=limit for r in p['dataset']['records']):raise ValueError('sequence bound; truncation forbidden')
    for split in ('train','development','regression'):
        part=rows(p['dataset'],split)
        if not part:raise ValueError('public '+split+' required')
        if not p['declared_class_ablation'] and {r['expected_decision'] for r in part}!=set(DECISIONS):raise ValueError('four-class coverage required')
    return p
