import copy,json,collections,hashlib
from pathlib import Path
import pytest
from saal_rsi.epoch_contract import new_job,validate_job,worker_identity
from saal_rsi.pretrained_contract import model_spec,validate_job as validate_v5,worker_identity as v5_identity
from saal_rsi.epoch_sampler import sample_ids,manifests

@pytest.fixture(scope='module')
def data():
    return json.loads((Path(__file__).resolve().parents[2]/'research/learning-reliability-v2/corpus/dataset.json').read_text())

def job(data,**kwargs):
    return new_job(model_spec('answerdotai/ModernBERT-base',backend='local',precision='float32',method='lora',rank=4),data,steps=10,batch_size=8,learning_rate=.0003,**kwargs)

def test_complete_epochs_and_four_class_batches(data):
    train=[r for r in data['records'] if r['split']=='train'];ids=sample_ids(train,'complete-balanced-epochs-v1',47,8000);by_id={r['record_id']:r for r in train}
    assert len(set(ids[:3840]))==3840 and len(set(ids[3840:7680]))==3840
    assert all(set(x)==set(by_id) for x in (ids[:3840],ids[3840:7680]))
    for i in range(0,len(ids),8):assert collections.Counter(by_id[r]['expected_decision'] for r in ids[i:i+8])=={d:2 for d in ('CHEAP','FAST','ASK','REFUSE')}
    assert sample_ids(train,'complete-balanced-epochs-v1',47,8000)==ids
    assert sample_ids(train,'complete-balanced-epochs-v1',83,8000)!=ids

def test_uniform_control_is_historical_seeded_sampler(data):
    import random
    train=[r for r in data['records'] if r['split']=='train'];rng=random.Random(47)
    assert sample_ids(train,'uniform-v1',47,80)==[rng.choice(train)['record_id'] for _ in range(80)]

def test_v5_cannot_process_v6(data):
    j=job(data)
    assert j['worker_identity']!=v5_identity()
    assert j['foundation_worker_identity']=='sha256:e3cea596f0a1ab795c105b7ba00b1a9a6a8188da1dfd59bcfd0d2601b0176f14'
    with pytest.raises(ValueError):validate_v5(j)
    assert validate_job(j)==j

@pytest.mark.parametrize('mutation',['sampler','worker','foundation','unknown','rank','protected','promotion','rate','warmup','diagnostic','imbalance','batch'])
def test_closed_admission_negative_cases(data,mutation):
    j=job(data)
    if mutation=='sampler':j['training_policy']['sampler']='use_holdout'
    elif mutation=='worker':j['worker_identity']='sha256:'+'0'*64
    elif mutation=='foundation':j['foundation_worker_identity']='sha256:'+'0'*64
    elif mutation=='unknown':j['training_policy']['python']='print(1)'
    elif mutation=='rank':j['model_spec']['adapter_rank']=3
    elif mutation=='protected':j['final_holdout_included']=True
    elif mutation=='promotion':j['external_deployment_allowed']=True
    elif mutation=='rate':j['training_policy']['head_learning_rate']=float('nan')
    elif mutation=='warmup':j['training_policy']['head_warmup_steps']=2
    elif mutation=='diagnostic':j['training_policy']['diagnostic_per_class']=50000
    elif mutation=='imbalance':
        from saal_rsi.spec import sha
        j['dataset']=copy.deepcopy(j['dataset']);j['dataset']['records'].pop(0);j['dataset_digest']=sha(j['dataset'])
    else:j['batch_size']=7
    with pytest.raises(ValueError):validate_job(j)

def test_sampler_rejects_evaluation_input(data):
    with pytest.raises(ValueError):sample_ids(data['records'],'uniform-v1',47,8)

def test_diagnostic_population_fixed_across_models_and_seeds(data):
    a=job(data);b=copy.deepcopy(a);b['seed']=83
    _,da=manifests(a);_,db=manifests(b)
    assert da['record_ids']==db['record_ids'] and len(da['record_ids'])==512
    train={r['record_id'] for r in data['records'] if r['split']=='train'}
    assert set(da['record_ids'])<=train

def test_staged_worker_dependency_closure(tmp_path):
    import subprocess,sys
    from saal_rsi.epoch_backend import stage_worker
    stage_worker(tmp_path/'worker')
    subprocess.run([sys.executable,'-I',str(tmp_path/'worker/worker.py'),'--help'],check=True,capture_output=True,timeout=30)

def test_optimizer_warmup_preserves_encoder_and_records_fit(tmp_path,data):
    import torch
    from saal_rsi.learning_backend import LearningBackend
    from saal_rsi.epoch_training import train
    from saal_rsi.sequence_replay_v2 import prediction_equal
    # Explicit tiny engineering fixture, never a pretrained research result.
    class FixtureBackend(LearningBackend):
        def load_pretrained(self):return self
        def batch_loss(self,records,class_weights=None):
            labels=torch.tensor([('CHEAP','FAST','ASK','REFUSE').index(r['expected_decision']) for r in records]);x=torch.nn.functional.one_hot(labels,4).float();logits=self.model(x)
            return torch.nn.functional.cross_entropy(logits,labels,weight=class_weights),len(records)
        def evaluate(self,records,max_new_tokens=80):
            self.model.eval();out=[]
            with torch.no_grad():
                for r in records:
                    label=('CHEAP','FAST','ASK','REFUSE').index(r['expected_decision']);p=self.model(torch.nn.functional.one_hot(torch.tensor(label),4).float()).softmax(-1).tolist();out.append({'decision':('CHEAP','FAST','ASK','REFUSE')[max(range(4),key=p.__getitem__)],'probabilities':p})
            return out
    j=new_job(model_spec('answerdotai/ModernBERT-base',backend='local',precision='float32',method='full'),data,steps=4,batch_size=8,learning_rate=.001,head_learning_rate=.001,encoder_learning_rate=.0001,head_warmup_steps=2)
    b=FixtureBackend(j['model_spec'],tmp_path,47)
    b.model=torch.nn.Sequential(collections.OrderedDict([('encoder',torch.nn.Linear(4,4)),('classifier',torch.nn.Linear(4,4))]));b.trainable=dict(b.model.named_parameters());b.initial_frozen_hash=b.frozen_hash();b.initial_trainable_hash=b.trainable_hash()
    before={n:p.detach().clone() for n,p in b.trainable.items()}
    result=train(b,j,tmp_path/'run')
    assert result['warmup_evidence']['encoder_unchanged']
    assert result['real_parameter_updates'] and result['resource_accounting']['sample_exposures']==32
    assert 'negative_log_likelihood_by_class' in result['snapshots']['terminal']['metrics']['train']
    assert len(result['full_training_evaluation']['terminal']['predictions'])==3840
    assert set(p.name for p in (tmp_path/'run').glob('*.safetensors'))=={'initial.safetensors','best.safetensors','terminal.safetensors'}
    from safetensors.torch import load_file
    terminal=load_file(str(tmp_path/'run/terminal.safetensors'))
    assert any(not torch.equal(before[n],p) for n,p in terminal.items() if n.startswith('classifier.'))
