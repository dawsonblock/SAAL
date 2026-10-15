import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from saal_rsi import adaptive_training_v4 as training
from saal_rsi.pretrained_contract import DECISIONS


class TinyBackend:
    def __init__(self):
        torch.manual_seed(61)
        self.model=torch.nn.Linear(4,4)
        self.trainable=dict(self.model.named_parameters())
        self.initial_trainable_hash=self.trainable_hash()
        self.unadapted_model_hash='test-only'
    def trainable_hash(self):
        return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in self.trainable.values())).hexdigest()
    def inputs(self,rows): return torch.eye(4)[[DECISIONS.index(r['expected_decision']) for r in rows]]
    def batch_objective(self,rows,weights=None):
        target=torch.tensor([DECISIONS.index(r['expected_decision']) for r in rows])
        loss=torch.nn.functional.cross_entropy(self.model(self.inputs(rows)),target,weight=weights)
        return loss,len(rows)*4,{'cross_entropy':float(loss.detach()),'orthogonal_penalty':0.}
    def evaluate(self,rows):
        self.model.eval()
        with torch.no_grad(): values=torch.softmax(self.model(self.inputs(rows)),dim=-1).tolist()
        return [{'decision':DECISIONS[max(range(4),key=p.__getitem__)],'probabilities':p} for p in values]
    def export_checkpoint(self,path): save_file(self.model.state_dict(),str(path))
    def restore_checkpoint(self,path,state): self.model.load_state_dict(load_file(str(path)))
    def verify_frozen_weights(self): return {'unchanged':True,'scope':'test fixture has no backbone'}
    def report_resources(self): return {'trainable_parameters':20}


class TinyController:
    def __init__(self,model,adapter): self.state={'step':0}
    def complete_update(self,step,**kwargs):
        assert kwargs['optimizer_updated'] is True
        self.state['step']=step
        return {'aggregate_active_rank':0,'completed_optimizer_step':step}
    def require_completion(self,steps):
        assert self.state['step']==steps
        return {'completed_optimizer_updates':steps}


def job():
    records=[]
    for split in ('train','development'):
        for i,decision in enumerate(DECISIONS):
            records.append({'record_id':split+str(i),'task_family_id':split+str(i//2),
                'cluster_id':split+'pair'+str(i//2),'category':'category',
                'expected_decision':decision,'split':split})
    return {'dataset':{'records':records},'max_seconds':30,'steps':4,'batch_size':4,
        'accumulation':2,'checkpoint_interval':2,'sampler_seed':61,
        'adapter':{'method':'fixed-r8'},'model_spec':{'precision':'float32'},
        'head_learning_rate':.01,'learning_rate':.02,'class_balance':False,
        'selection_policy':'public-guardrails-pair-family-v1','worker_identity':'test-only',
        'replay_tolerance':1e-4}


def test_v4_loop_records_effective_learning_rates_each_completed_update(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'RankController',TinyController)
    monkeypatch.setattr(training,'sample_ids',lambda records,policy,seed,count:
        [records[i%len(records)]['record_id'] for i in range(count)])
    out=tmp_path/'output'
    result=training.train(TinyBackend(),job(),out)
    rows=[json.loads(line) for line in (out/'training_metrics.jsonl').read_text().splitlines()]
    progress=[r for r in rows if r.get('monitor')=='training_progress']
    assert result['schema']=='saal-adaptive-result-v4'
    assert [r['optimizer_step'] for r in progress]==[1,2,3,4]
    assert all(r['optimizer']=='AdamW' and r['scheduler']=='none' for r in progress)
    assert all(r['learning_rates']=={'head':.01,'adapter':.02} for r in progress)


def test_v4_loop_does_not_advance_rank_schedule_after_skipped_update(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'RankController',TinyController)
    monkeypatch.setattr(training,'sample_ids',lambda records,policy,seed,count:
        [records[i%len(records)]['record_id'] for i in range(count)])
    monkeypatch.setattr(training,'optimizer_update',lambda *args,**kwargs:
        (False,{'finite':False,'norm':0.,'gradient_tensors':0}))
    with pytest.raises(ValueError,match='optimizer did not complete'):
        training.train(TinyBackend(),job(),tmp_path/'failed')
    assert not (tmp_path/'failed/rank_history.jsonl').exists()
    assert not (tmp_path/'failed/terminal.safetensors').exists()
