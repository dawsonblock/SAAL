"""Additive v5 admission. Legacy v2/v3/v4 workers and identities stay unchanged."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,math,re
from .spec import canonical,sha,parse_unique_json

DECISIONS=('CHEAP','FAST','ASK','REFUSE')
VERSIONS={'torch':'2.10.0','transformers':'5.19.0','peft':'0.21.2','safetensors':'0.8.0','numpy':'2.4.6','huggingface-hub':'1.33.0','cryptography':'48.0.1'}
SOURCES=('spec.py','validation.py','model_contract.py','pretrained_contract.py','pretrained_registry.py','learning_dataset.py','task_oracles.py','learning_backend.py','modernbert_backend.py','qwen35_backend.py','peft_training.py','checkpoint_selection.py','pretrained_artifacts.py','pretrained_worker.py')

def registry_path():return Path(__file__).resolve().parents[2]/'configs/pretrained/registry.json'
def registry():return parse_unique_json(registry_path().read_bytes())
def entry(repository,revision):
    found=[m for m in registry()['models'] if (m['repository'],m['revision'])==(repository,revision)]
    if len(found)!=1:raise ValueError('unregistered official snapshot/revision')
    return found[0]
def worker_identity():return sha({n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES}|{'registry':hashlib.sha256(registry_path().read_bytes()).hexdigest(),'tensor_inventory':hashlib.sha256(registry_path().with_name('tensor_inventories.json').read_bytes()).hexdigest()})
def targets(kind):
    if kind=='modernbert-seqcls-v1':return [f'model.layers.{i}.attn.Wqkv' for i in range(22)]
    if kind=='qwen35-generative-v1':return [f'model.language_model.layers.{i}.'+suffix for i in range(24) for suffix in (('self_attn.q_proj','self_attn.v_proj') if i%4==3 else ('linear_attn.in_proj_qkv',))]
    raise ValueError('unsupported architecture')

@dataclass(frozen=True)
class PretrainedModelSpec:
    model_kind:str
    model_repository:str
    model_revision:str
    model_files_digest:str
    tokenizer_revision:str
    tokenizer_digest:str
    processor_digest:str|None
    architecture:str
    max_input_tokens:int
    output_schema_version:str
    adapter_type:str
    adapter_rank:int
    adapter_targets:list[str]
    precision:str
    compute_backend:str
    resource_budget_id:str
    chat_template_digest:str
    preprocessing:str
    dependency_identity:dict
    attention_implementation:str='eager'
    def __post_init__(self):
        e=entry(self.model_repository,self.model_revision)
        for field,key in [('model_kind','model_kind'),('architecture','architecture'),('model_files_digest','model_files_digest'),('tokenizer_digest','tokenizer_digest'),('processor_digest','processor_digest'),('chat_template_digest','chat_template_digest')]:
            if getattr(self,field)!=e[key]:raise ValueError('snapshot identity mismatch: '+field)
        if self.tokenizer_revision!=self.model_revision:raise ValueError('tokenizer revision substitution')
        if type(self.max_input_tokens)is not int or self.max_input_tokens not in (128,256,512):raise ValueError('token limit')
        if self.adapter_type not in ('head','full','lora','none') or self.model_kind=='qwen35-generative-v1' and self.adapter_type not in ('lora','none'):raise ValueError('unadmitted training method')
        if type(self.adapter_rank)is not int or self.adapter_rank not in ((4,8,16) if self.adapter_type=='lora' else (0,)):raise ValueError('rank')
        if self.adapter_targets!=(targets(self.model_kind) if self.adapter_type=='lora' else []):raise ValueError('adapter target inventory')
        if self.output_schema_version!=('decision-logits-v1' if self.model_kind=='modernbert-seqcls-v1' else 'synthetic-json-v1'):raise ValueError('output schema')
        if self.preprocessing!=('native-tokenizer-no-truncation-v1' if self.model_kind=='modernbert-seqcls-v1' else 'native-chat-text-only-no-thinking-v1'):raise ValueError('preprocessing')
        if self.precision not in ('float32','float16') or self.compute_backend not in ('local','colab'):raise ValueError('precision/backend')
        if self.attention_implementation not in ('eager','sdpa'):raise ValueError('unreviewed attention implementation')
        if self.dependency_identity!=VERSIONS:raise ValueError('dependency identity')
        if type(self.resource_budget_id)is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',self.resource_budget_id):raise ValueError('budget identity')
    def as_dict(self):return asdict(self)
    @classmethod
    def parse(cls,d):
        if type(d)is not dict or set(d)!=set(cls.__dataclass_fields__):raise ValueError('ModelContractV5 fields')
        return cls(**d)

def model_spec(repository,method='lora',rank=8,backend='colab',precision='float16',budget='pretrained_trial',max_tokens=256):
    e=next(m for m in registry()['models'] if m['repository']==repository)
    return PretrainedModelSpec(e['model_kind'],repository,e['revision'],e['model_files_digest'],e['revision'],e['tokenizer_digest'],e['processor_digest'],e['architecture'],max_tokens,
        'decision-logits-v1' if e['model_kind']=='modernbert-seqcls-v1' else 'synthetic-json-v1',method,rank if method=='lora' else 0,targets(e['model_kind']) if method=='lora' else [],precision,backend,budget,e['chat_template_digest'],
        'native-tokenizer-no-truncation-v1' if e['model_kind']=='modernbert-seqcls-v1' else 'native-chat-text-only-no-thinking-v1',VERSIONS.copy()).as_dict()

def new_job(spec,dataset,*,seed=17,steps=50,batch_size=4,accumulation=1,learning_rate=.0001,max_seconds=900,class_balance=False,checkpoint_interval=25,max_new_tokens=80,initial_loss_scale=256):
    lock=Path(__file__).resolve().parents[2]/'configs/dependency-locks'/('pretrained-linux-x86_64-py313.txt' if spec['compute_backend']=='colab' else 'pretrained-macos-arm64-py312.txt')
    p={'schema':'saal-model-job-v5','model_spec':spec,'dataset':dataset,'dataset_digest':sha(dataset),'seed':seed,'steps':steps,'batch_size':batch_size,'accumulation':accumulation,'learning_rate':learning_rate,'max_seconds':max_seconds,'class_balance':class_balance,'checkpoint_interval':checkpoint_interval,'max_new_tokens':max_new_tokens,
       'initial_loss_scale':initial_loss_scale,'selection_policy':'public-family-class-pair-v2','worker_identity':worker_identity(),'dependency_lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),'external_deployment_allowed':False,'final_holdout_included':False,'qualification_status':'NOT_EVALUATED','resume_policy':'NO_RESUME'}
    validate_job(p);return p

def validate_job(p,*,execution=True):
    fields={'initial_loss_scale','schema','model_spec','dataset','dataset_digest','seed','steps','batch_size','accumulation','learning_rate','max_seconds','class_balance','checkpoint_interval','max_new_tokens','selection_policy','worker_identity','dependency_lock_sha256','external_deployment_allowed','final_holdout_included','qualification_status','resume_policy'}
    if type(p)is not dict or set(p)!=fields or p['schema']!='saal-model-job-v5':raise ValueError('v5 job schema')
    spec=PretrainedModelSpec.parse(p['model_spec'])
    from .learning_dataset import validate_dataset
    validate_dataset(p['dataset'],kind=spec.model_kind)
    if p['dataset_digest']!=sha(p['dataset']):raise ValueError('data substitution')
    if p['worker_identity']!=worker_identity():
        if execution:raise ValueError('worker substitution / historical worker cannot execute')
        approved=parse_unique_json(registry_path().with_name('historical_workers.json').read_bytes())['workers'].get(p['worker_identity'])
        if approved is None or approved.get('scope')!='VERIFY_AND_REPLAY_ONLY':raise ValueError('unreviewed historical worker')
        archive=Path(__file__).resolve().parents[2]/approved['archive']
        if archive.is_symlink() or hashlib.sha256(archive.read_bytes()).hexdigest()!=approved['archive_sha256']:raise ValueError('historical worker archive substitution')
    for k,lo,hi in [('initial_loss_scale',1,65536),('seed',0,2**31-1),('steps',0,1000),('batch_size',1,32),('accumulation',1,16),('max_seconds',30,3600),('checkpoint_interval',1,200),('max_new_tokens',8,128)]:
        if type(p[k])is not int or not lo<=p[k]<=hi:raise ValueError('bounded '+k)
    if type(p['learning_rate'])is not float or not math.isfinite(p['learning_rate']) or not 1e-7<=p['learning_rate']<=.01:raise ValueError('learning rate')
    if type(p['class_balance'])is not bool:raise ValueError('balanced loss Boolean')
    if spec.adapter_type=='none' and p['steps']!=0 or spec.adapter_type!='none' and p['steps']==0:raise ValueError('training budget/method mismatch')
    if spec.model_kind=='qwen35-generative-v1' and (p['class_balance'] or p['batch_size']!=1):raise ValueError('generative microbatch/loss contract')
    if p['selection_policy']!='public-family-class-pair-v2' or p['resume_policy']!='NO_RESUME' or p['qualification_status']!='NOT_EVALUATED' or p['external_deployment_allowed'] is not False or p['final_holdout_included'] is not False:raise ValueError('authority escalation')
    if type(p['dependency_lock_sha256'])is not str or not re.fullmatch('[0-9a-f]{64}',p['dependency_lock_sha256']):raise ValueError('lock hash')
    return p


def job_bytes(path):
    """Bounded regular-file job/config input, before JSON and signature parsing."""
    import os,stat
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 1<=info.st_size<=50_000_000:raise ValueError('signed job file bound')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(50_000_001)
        if len(raw)>50_000_000:raise ValueError('signed job grew beyond admitted bound')
        return raw
    finally:os.close(fd)
