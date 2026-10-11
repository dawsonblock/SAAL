"""Behavioral and adversarial v5 controls, without downloading models in tests."""
import copy,json,math,sqlite3
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.pretrained_contract import model_spec,PretrainedModelSpec,new_job,validate_job,worker_identity,DECISIONS
from saal_rsi.learning_dataset import semantic_pilot,generative_pilot,validate_dataset,diagnostic_audit
from saal_rsi.task_oracles import oracle,parse_output,score_output,authorize_simulated_execution
from saal_rsi.model_contract import sign,validate_job as legacy_validate
from saal_rsi.pretrained_artifacts import tensor_inventory
from saal_rsi.learning_experiments import LearningExperimentRegistry
from saal_rsi.checkpoint_selection import metrics,selection_key,paired_family_gain
from saal_rsi.spec import canonical,sha

@pytest.fixture
def dataset():return semantic_pilot()
@pytest.fixture
def job(dataset):return new_job(model_spec('answerdotai/ModernBERT-base',method='head'),dataset,steps=2,max_seconds=60,checkpoint_interval=1)

@pytest.mark.parametrize('field,value',[('model_revision','main'),('model_repository','attacker/ModernBERT-base'),('architecture','TinyTransformer'),('tokenizer_revision','main'),('tokenizer_digest','sha256:'+'0'*64),('processor_digest','sha256:'+'0'*64),('chat_template_digest','sha256:'+'0'*64),('precision','bfloat16'),('adapter_targets',['classifier']),('adapter_rank',True),('max_input_tokens',8192),('attention_implementation','flash_attention_2'),('preprocessing','truncate-v1')])
def test_model_spec_rejects_substitution(field,value):
    s=model_spec('Qwen/Qwen3.5-0.8B');s[field]=value
    with pytest.raises(ValueError):PretrainedModelSpec.parse(s)

@pytest.mark.parametrize('field,value',[('external_deployment_allowed',True),('final_holdout_included',True),('qualification_status','QUALIFIED'),('resume_policy','AUTO_RETRY'),('seed',True),('steps',1001),('learning_rate',float('nan')),('worker_identity','sha256:'+'0'*64),('dataset_digest','sha256:'+'0'*64)])
def test_job_fail_closed(job,field,value):
    job[field]=value
    with pytest.raises(ValueError):validate_job(job)

def test_legacy_authority_rejects_v5(job):
    with pytest.raises(ValueError):legacy_validate(job)

def test_generator_groups_and_counterfactuals(dataset):
    validate_dataset(dataset,kind='modernbert-seqcls-v1');audit=diagnostic_audit(dataset)
    assert audit['qualification_eligible'] is False and audit['independence_established'] is False
    pairs={}
    for r in dataset['records']:pairs.setdefault(r['cluster_id'],[]).append(r)
    assert any({r['expected_decision'] for r in v}=={'CHEAP','FAST'} for v in pairs.values())
    assert all(r['review_status']=='ORACLE_CHECKED_NOT_INDEPENDENTLY_REVIEWED' for r in dataset['records'])

def test_no_family_leakage(dataset):
    r=next(r for r in dataset['records'] if r['split']=='development');r['task_family_id']=dataset['records'][0]['task_family_id']
    with pytest.raises(ValueError):validate_dataset(dataset)

def test_reject_duplicate_and_forged_oracle(dataset):
    d=copy.deepcopy(dataset);d['records'][1]['prompt']='  '+d['records'][0]['prompt'].upper()+'  '
    with pytest.raises(ValueError):validate_dataset(d)
    d=copy.deepcopy(dataset);d['records'][0]['expected_output']['decision']='FAST'
    with pytest.raises(ValueError):validate_dataset(d)

def test_protected_training_forbidden(dataset):
    dataset['records'][0]['split']='holdout'
    with pytest.raises(ValueError):validate_dataset(dataset)

def test_generative_output_is_not_authority():
    text=canonical(oracle('decision',{'decision':'FAST'})).decode()
    assert authorize_simulated_execution(text,trusted_permission=False,scope_matches=True) is False
    assert authorize_simulated_execution(text,trusted_permission=True,scope_matches=False) is False
    assert authorize_simulated_execution(text,trusted_permission=True,scope_matches=True) is True

@pytest.mark.parametrize('output',['{"decision":"FORBIDDEN"}','```json\n{}\n```','{"item":"x","quantity":true}','{"order":[1,NaN]}','{"item":"x","item":"y","quantity":2}'])
def test_no_output_repair(output):
    r=next(r for r in generative_pilot()['records'] if r['task_type']=='extract');assert score_output(output,r)['correct'] is False

def test_completion_only_masking():
    from saal_rsi.qwen35_backend import completion_tokens
    class T:
        eos_token='!'
        def apply_chat_template(self,*a,**k):return 'PROMPT:'
        def __call__(self,text,**kw):return {'input_ids':list(text.encode())}
    ids,labels=completion_tokens(T(),'question',{'x':1},256)
    assert labels[:7]==[-100]*7 and labels[7:]==ids[7:] and labels[-1]==ord('!')
    with pytest.raises(ValueError):completion_tokens(T(),'question',{'x':1},5)
    class Bad(T):
        def __call__(self,text,**kw):return {'input_ids':list(text.encode()) if text=='PROMPT:' else [9]*len(text)}
    with pytest.raises(ValueError):completion_tokens(Bad(),'question',{'x':1},256)

def test_exact_adapter_targets_and_inventory():
    s=model_spec('Qwen/Qwen3.5-0.8B');inv=tensor_inventory(s)
    assert len(inv)==60 and all('visual' not in n for n in inv)
    assert any('linear_attn.in_proj_qkv' in n for n in inv) and any('self_attn.q_proj' in n for n in inv)
    s=model_spec('answerdotai/ModernBERT-base');assert len(tensor_inventory(s))==48

def test_durable_budget_and_replay(job,tmp_path):
    key=Ed25519PrivateKey.generate();pin=key.public_key().public_bytes_raw().hex();envelope=sign(job,key)
    r=LearningExperimentRegistry(tmp_path/'budget.db',pin,max_jobs=1,max_seconds=2000);d=r.reserve(envelope);r.start(d);r.finish(d,{'failure':'interrupted'},False)
    with pytest.raises((ValueError,sqlite3.IntegrityError)):r.reserve(envelope)
    r.close();r=LearningExperimentRegistry(tmp_path/'budget.db',pin)
    assert r.records()['attempts'][0]['state']=='INTERRUPTED'
    with pytest.raises(ValueError):r.start(d)
    r.close()

def test_false_refusal_and_required_clarification_are_separate(dataset):
    records=[next(r for r in dataset['records'] if r['expected_decision']==d) for d in DECISIONS]
    correct=[{'decision':d,'probabilities':[float(x==d) for x in DECISIONS]} for d in DECISIONS];m=metrics(records,correct,'modernbert-seqcls-v1');assert m['accuracy']==1 and m['unauthorized_proposals']==0
    incorrect=copy.deepcopy(correct);incorrect[3]={'decision':'FAST','probabilities':[0.,1.,0.,0.]};n=metrics(records,incorrect,'modernbert-seqcls-v1');assert n['unauthorized_proposals']==1 and selection_key(n)<selection_key(m)
    assert paired_family_gain(n,m)['mean_gain']>0

def test_worker_stage_excludes_keys_and_authority(tmp_path):
    from saal_rsi.pretrained_backend import stage_worker
    stage_worker(tmp_path)
    names={p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()}
    assert not any(n.endswith('.key') or 'promotion.py' in n or 'family_evaluation.py' in n for n in names)
    assert 'src/saal_rsi/pretrained_worker.py' in names

def test_native_lora_freeze_and_reload(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('transformers');pytest.importorskip('peft')
    from transformers import ModernBertConfig,ModernBertForSequenceClassification
    from saal_rsi.modernbert_backend import ModernBERTBackend
    from safetensors.torch import load_file,save_file
    s=model_spec('answerdotai/ModernBERT-base',backend='local',precision='float32');b=ModernBERTBackend(s,tmp_path,17)
    torch.manual_seed(17);b.model=ModernBertForSequenceClassification(ModernBertConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=22,num_attention_heads=4,max_position_embeddings=128,num_labels=4,pad_token_id=0,bos_token_id=1,eos_token_id=2,cls_token_id=1,sep_token_id=2));b.initialize_experiment()
    # This reduced native architecture is a unit fixture, never a registered model.
    before=b.frozen_hash();optimizer=torch.optim.AdamW(b.trainable.values(),lr=.001);x=torch.tensor([[1,2,3],[3,2,1]]);loss=b.model(input_ids=x,attention_mask=torch.ones_like(x),labels=torch.tensor([0,1])).loss;loss.backward();optimizer.step();assert b.frozen_hash()==before
    path=tmp_path/'adapter.safetensors';b.export_checkpoint(path);b.model.eval()
    with torch.no_grad():expected=b.model(input_ids=x,attention_mask=torch.ones_like(x)).logits.clone()
    with torch.no_grad():
        for p in b.trainable.values():p.zero_()
    b.restore_checkpoint(path)
    with torch.no_grad():actual=b.model(input_ids=x,attention_mask=torch.ones_like(x)).logits
    assert torch.allclose(actual,expected,atol=1e-6)
    state=load_file(str(path));state[next(iter(state))]=torch.tensor([float('nan')]);save_file(state,str(tmp_path/'tampered.safetensors'))
    with pytest.raises(ValueError):b.restore_checkpoint(tmp_path/'tampered.safetensors')

def test_concurrent_reservation_consumes_only_once(job,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    key=Ed25519PrivateKey.generate();pin=key.public_key().public_bytes_raw().hex();e=sign(job,key);path=tmp_path/'concurrent.db'
    r=LearningExperimentRegistry(path,pin,max_jobs=2,max_seconds=5000);r.close()
    def reserve():
        ledger=LearningExperimentRegistry(path,pin)
        try:
            ledger.reserve(e);return True
        except (ValueError,sqlite3.IntegrityError):return False
        finally:ledger.close()
    with ThreadPoolExecutor(2) as pool:assert sorted(pool.map(lambda _:reserve(),range(2)))==[False,True]

@pytest.mark.parametrize('task,payload',[('extract',{'item':'x','quantity':True}),('extract',{'item':'x','quantity':1,'hidden':'bad'}),('sort',{'values':[True,2]}),('decision',{'decision':'FAST','authority':True})])
def test_oracle_types_and_inventory_are_admitted(task,payload):
    with pytest.raises(ValueError):oracle(task,payload)

def test_historical_worker_can_verify_but_cannot_execute():
    e=Path(__file__).resolve().parents[2]/'examples/pretrained-head-compat-v2/signed_training_job.json'
    if not e.exists():pytest.skip('historical example absent')
    job=json.loads(e.read_text())['payload'];validate_job(job,execution=False)
    with pytest.raises(ValueError):validate_job(job)
