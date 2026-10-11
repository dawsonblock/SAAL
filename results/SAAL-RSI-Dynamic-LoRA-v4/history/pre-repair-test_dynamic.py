from pathlib import Path
import copy,json
import pytest
from saal_rsi.spec import sha
from saal_adalora.contract import ADAPTER,validate,source_identity,FIXED_PIN,FIXED_DIGEST,SCREEN
from saal_adalora.controller import reserve,consume,finish
from saal_rsi.pretrained_contract import worker_identity

def job():
    # Exact frozen reference fixture is copied at release preparation, before any dynamic run.
    e=json.loads((Path(__file__).parent/'fixed-job.json').read_text());r=e['payload']
    return {'schema':'saal-adalora-research-job-v1','native_configuration_reference':r,'dynamic_adapter':copy.deepcopy(ADAPTER),'dynamic_source_identity':source_identity(),'base_worker_identity':worker_identity(),'fixed_comparison_digest':FIXED_DIGEST,'fixed_comparison_envelope':e,'fixed_comparison_pin':FIXED_PIN,'screen':copy.deepcopy(SCREEN),'resource_budget_id':'dynamic4_'+source_identity()[7:23],'reserved_seconds':2300,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}

def test_exact_new_contract_admits_fixed_reference():assert validate(job())['steps']==600
@pytest.mark.parametrize('change',['rank','steps','source','promotion','comparison','screen','budget'])
def test_unadmitted_change_fails(change):
    j=job()
    if change=='rank':j['dynamic_adapter']['target_r']=64
    elif change=='steps':j['native_configuration_reference']['steps']=999
    elif change=='source':j['dynamic_source_identity']='sha256:'+'0'*64
    elif change=='promotion':j['external_deployment_allowed']=True
    elif change=='screen':j['screen']['complete_pair_accuracy']=.1
    elif change=='budget':j['reserved_seconds']=10000
    else:j['fixed_comparison_digest']='sha256:'+'0'*64
    with pytest.raises(ValueError):validate(j)
def test_native_v5_cannot_be_reinterpreted():
    with pytest.raises(ValueError):validate(job()['native_configuration_reference'])
def test_attempt_is_one_use_and_failure_spends_it(tmp_path):
    p=tmp_path/'ledger';reserve(p,'candidate');consume(p,'candidate');finish(p,'candidate',{'state':'INTERRUPTED'})
    with pytest.raises(ValueError):consume(p,'candidate')
    with pytest.raises(ValueError):reserve(p,'candidate')
def test_receipt_does_not_match_substituted_candidate(tmp_path):
    p=tmp_path/'ledger';reserve(p,'a')
    with pytest.raises(ValueError):consume(p,'b')

def test_real_adalora_pruning_freeze_and_safe_reload(tmp_path):
    import torch
    from torch import nn
    from peft import AdaLoraConfig,get_peft_model
    from saal_rsi.learning_backend import LearningBackend
    torch.manual_seed(47)
    class Net(nn.Module):
        def __init__(self):super().__init__();self.query=nn.Linear(6,6);self.value=nn.Linear(6,3)
        def forward(self,x):return self.value(torch.tanh(self.query(x)))
    class Adapter(LearningBackend):
        def load_pretrained(self):pass
        def batch_loss(self,*a):pass
        def evaluate(self,*a):pass
    m=get_peft_model(Net(),AdaLoraConfig(init_r=4,target_r=1,total_step=12,tinit=2,tfinal=2,deltaT=1,target_modules=['query','value'],lora_alpha=8,lora_dropout=0))
    b=object.__new__(Adapter);b.model=m;b.trainable={n:p for n,p in m.named_parameters() if p.requires_grad};b.initial_frozen_hash=b.frozen_hash();b.initial_trainable_hash=b.trainable_hash();optimizer=torch.optim.AdamW(b.trainable.values(),lr=.01)
    for step in range(1,13):
        optimizer.zero_grad();x=torch.randn(8,6);loss=m(x).square().mean();loss.backward();optimizer.step();m.base_model.update_and_allocate(step)
    pattern=m.peft_config['default'].rank_pattern
    assert sum(sum(mask) for mask in pattern.values())<=2 and len(pattern)==2
    assert b.trainable_hash()!=b.initial_trainable_hash and b.frozen_hash()==b.initial_frozen_hash
    m.eval();x=torch.randn(2,6);before=m(x).detach();path=tmp_path/'adapter.safetensors';b.export_checkpoint(path)
    with torch.no_grad():
        for p in b.trainable.values():p.add_(.1)
    b.restore_checkpoint(path);assert torch.equal(m(x).detach(),before)
    from safetensors.torch import load_file,save_file
    tensors=load_file(path);k=next(iter(tensors));tensors[k].view(-1)[0]=float('nan');save_file(tensors,path)
    with pytest.raises(ValueError,match='finite'):b.restore_checkpoint(path)
