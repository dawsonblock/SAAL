import copy
import hashlib
from pathlib import Path
import shutil
import pytest
from saal_rsi.pretrained_contract import model_spec,new_job
from saal_rsi.public_mechanisms import public_mechanism_dataset
from saal_rsi.sequence_v2 import identity,validate_protocol,run_sequence_v2
from saal_rsi.sequence_replay_v2 import validate_bundle,prediction_equal
from saal_rsi.spec import sha,canonical,parse_unique_json


def make():
    report={'adaptation_gain_terminal':{'mean_gain':.1},'real_parameter_updates':True}
    job=new_job(model_spec('answerdotai/ModernBERT-base','lora',4,'local','float32'),public_mechanism_dataset(1),steps=8,batch_size=4,max_seconds=300,learning_rate=.01)
    protocol={'worker_identity':identity(),'schema':'saal-pretrained-sequence-v2','job_digest':sha(job),'groups':[['ordinary','correction','proxy'],['refusal','clarification','injection']],'method':'shared','replay_ratio':.25,'steps_per_stage':2,'max_seconds':300,'positive_learning_report_digest':sha(report),'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'replay_policy':'stratified-family-v1','replay_seed':71,'reservoir_capacity':512,'dropout_control':'seeded-per-update-v1'}
    return job,protocol,report


@pytest.mark.parametrize('field,value',[('worker_identity',sha('other')),('replay_policy','fixed-prefix'),('replay_seed',-1),('reservoir_capacity',25000),('dropout_control','uncontrolled'),('external_deployment_allowed',True),('schema','saal-pretrained-sequence-v1')])
def test_protocol_v2_no_silent_changes(field,value):
    job,p,report=make();p[field]=value
    with pytest.raises(ValueError):validate_protocol(p,job)


def test_actual_optimizer_loop_and_identifiable_evidence(monkeypatch,tmp_path):
    import torch
    from safetensors.torch import save_file
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from saal_rsi.model_contract import sign
    from saal_rsi.pretrained_contract import DECISIONS
    class Learner:
        def __init__(self):
            torch.manual_seed(17);self.model=torch.nn.Linear(4,4);self.trainable=dict(self.model.named_parameters())
        def load_pretrained(self):return self
        def tokenize(self,records):pass
        def batch_loss(self,records,weights):
            labels=torch.tensor([DECISIONS.index(r['expected_decision']) for r in records]);inputs=torch.nn.functional.one_hot(labels,4).float()
            return torch.nn.functional.cross_entropy(self.model(inputs),labels,weight=weights),len(records)
        def evaluate(self,records,max_tokens):
            labels=torch.tensor([DECISIONS.index(r['expected_decision']) for r in records])
            with torch.no_grad():probs=self.model(torch.nn.functional.one_hot(labels,4).float()).softmax(-1).tolist()
            return [{'decision':DECISIONS[max(range(4),key=lambda i:v[i])],'probabilities':v} for v in probs]
        def verify_frozen_weights(self):pass
        def trainable_hash(self):return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in self.trainable.values())).hexdigest()
        def export_checkpoint(self,path):save_file({n:p.detach().contiguous() for n,p in self.trainable.items()},str(path))
        def report_resources(self):return {'trainable_parameters':20}
    monkeypatch.setattr('saal_rsi.sequence_v2.create_backend',lambda *args:Learner())
    job,p,report=make();key=Ed25519PrivateKey.generate();pin=key.public_key().public_bytes_raw().hex()
    result=run_sequence_v2(sign(job,key),sign(p,key),pin,tmp_path,tmp_path/'bundle',report)
    assert result['completed_updates']==8 and len(result['stage_predictions'])==2
    a=result['resource_accounting'][1]
    experimental=a['experimental']['used_batches'];control=a['fresh_control']['used_batches']
    plan=parse_unique_json((tmp_path/'bundle/1-replay-plan.json').read_bytes())
    for index,(e,c) in enumerate(zip(experimental,control)):
        assert c['record_ids']==plan['batches'][index]['current_ids']
        assert e['record_ids']==c['record_ids']+plan['batches'][index]['replay_ids']
    validate_bundle(tmp_path/'bundle',pin)
    assert a['experimental']['checkpoint_sha256']!=a['fresh_control']['checkpoint_sha256']


def test_packaged_native_sequence_and_replay_evidence():
    root=Path(__file__).resolve().parents[2]/'examples/corrective_research'
    job,p,r=validate_bundle(root/'sequence-v2-smoke',(root/'controller.pub').read_text().strip())
    replay=parse_unique_json((root/'sequence-replay.json').read_bytes())
    assert replay['per_record_agreement']is True and r['completed_updates']==8
    assert r['resource_accounting'][1]['experimental']['used_batches']
    assert r['qualification_status']=='NOT_EVALUATED'


def test_tampered_replay_plan_even_with_rewritten_worker_inventory(tmp_path):
    source=Path(__file__).resolve().parents[2]/'examples/corrective_research'
    bundle=tmp_path/'bundle';shutil.copytree(source/'sequence-v2-smoke',bundle)
    plan=parse_unique_json((bundle/'1-replay-plan.json').read_bytes());plan['batches'][0]['replay_ids']=['protected-example']
    (bundle/'1-replay-plan.json').write_bytes(canonical(plan))
    inv=parse_unique_json((bundle/'inventory.json').read_bytes());q=bundle/'1-replay-plan.json';inv['files'][q.name]={'size':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()};(bundle/'inventory.json').write_bytes(canonical(inv))
    with pytest.raises(ValueError):validate_bundle(bundle,(source/'controller.pub').read_text().strip())


def test_prediction_parity_is_stricter_than_accuracy():
    a=[{'decision':'CHEAP','probabilities':[.7,.1,.1,.1]}]
    b=[{'decision':'CHEAP','probabilities':[.4,.2,.2,.2]}]
    assert not prediction_equal(a,b,'modernbert-seqcls-v1',1e-4)
    assert prediction_equal(a,a,'modernbert-seqcls-v1',1e-4)
