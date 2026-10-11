import copy
import pytest
from saal_rsi.pretrained_continual import validate_sequence,sequence_worker_identity
from saal_rsi.pretrained_contract import model_spec,new_job
from saal_rsi.learning_dataset import semantic_pilot
from saal_rsi.spec import sha
from saal_rsi.continual_learning import retention_metrics

def make():
    job=new_job(model_spec('answerdotai/ModernBERT-base'),semantic_pilot(),steps=40,max_seconds=600)
    protocol={'worker_identity':sequence_worker_identity(),'schema':'saal-pretrained-sequence-v1','job_digest':sha(job),'groups':[['ordinary','correction'],['refusal','clarification']],'method':'shared','replay_ratio':.25,'steps_per_stage':10,'max_seconds':600,'positive_learning_report_digest':'sha256:'+'0'*64,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    return protocol,job

def test_registered_sequence():
    p,j=make();assert validate_sequence(p,j)==p

@pytest.mark.parametrize('field,value',[('groups',[['ordinary'],['ordinary']]),('replay_ratio',.99),('method','dynamic-rank'),('external_deployment_allowed',True),('max_seconds',100000),('job_digest','sha256:'+'0'*64)])
def test_sequence_fail_closed(field,value):
    p,j=make();p[field]=value
    with pytest.raises(ValueError):validate_sequence(p,j)

def test_retention_reports_actual_losses():
    r=retention_metrics([[.9,.2],[.6,.8]],[.3,.2]);assert r['backward_transfer']==pytest.approx(-.3);assert r['forgetting_by_task'][0]==pytest.approx(.3)

def test_sequential_optimizer_executes_both_branches(monkeypatch,tmp_path):
    # A reduced supervised learner exercises the actual sequence update loop,
    # including autocast, optimizer, replay, fresh controls and checkpoint export.
    import torch
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from saal_rsi.model_contract import sign
    from saal_rsi.pretrained_continual import run_sequence
    from saal_rsi.pretrained_contract import DECISIONS
    class Fixture:
        def __init__(self):
            torch.manual_seed(17);self.model=torch.nn.Linear(4,4);self.trainable=dict(self.model.named_parameters())
        def load_pretrained(self):return self
        def tokenize(self,records):pass
        def batch_loss(self,records,weights):
            y=torch.tensor([DECISIONS.index(r['expected_decision']) for r in records]);x=torch.nn.functional.one_hot(y,4).float()
            return torch.nn.functional.cross_entropy(self.model(x),y,weight=weights),len(records)
        def evaluate(self,records,max_tokens):
            y=torch.tensor([DECISIONS.index(r['expected_decision']) for r in records])
            with torch.no_grad():p=self.model(torch.nn.functional.one_hot(y,4).float()).softmax(-1).tolist()
            return [{'decision':DECISIONS[max(range(4),key=lambda i:v[i])],'probabilities':v} for v in p]
        def verify_frozen_weights(self):return True
        def export_checkpoint(self,path):torch.save(self.model.state_dict(),path)
        def report_resources(self):return {'trainable_parameters':20}
    monkeypatch.setattr('saal_rsi.pretrained_continual.create_backend',lambda *args:Fixture())
    p,j=make();report={'adaptation_gain_terminal':{'mean_gain':.1},'real_parameter_updates':True};p['positive_learning_report_digest']=sha(report)
    key=Ed25519PrivateKey.generate();result=run_sequence(sign(j,key),sign(p,key),key.public_key().public_bytes_raw().hex(),tmp_path,tmp_path/'out',report)
    assert len(result['accuracy_matrix'])==2 and len(result['fresh_control_accuracy'])==2
    assert len(list((tmp_path/'out').glob('*.safetensors')))==4
    assert result['resource_accounting'][1]['replay_examples']>0
    assert result['qualification_status']=='NOT_EVALUATED' and result['external_deployment_allowed'] is False


def test_historical_sequence_verifies_but_cannot_execute():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/'examples/pretrained_research/v094-sequence-no-replay'
    job=json.loads((root/'signed_job.json').read_text())['payload'];protocol=json.loads((root/'signed_protocol.json').read_text())['payload']
    validate_sequence(protocol,job,execution=False)
    with pytest.raises(ValueError):validate_sequence(protocol,job)
