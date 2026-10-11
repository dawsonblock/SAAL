import copy,pytest
from pathlib import Path
from saal_rsi.learning_dataset import semantic_pilot,validate_dataset
from saal_rsi.pretrained_contract import model_spec,new_job,validate_job,job_bytes

@pytest.mark.parametrize('field,value',[('source_digest',[]),('category',{}),('parent_record_id',[]),('label_rationale','x'*4097),('split',[]),('task_type',{}),('parent_record_id',True)])
def test_metadata_rejected_before_processing(field,value):
 d=semantic_pilot();d['records'][0][field]=value
 with pytest.raises(ValueError):validate_dataset(d)

def test_job_input_size_and_symlink(tmp_path):
 p=tmp_path/'job.json'
 with p.open('wb') as f:f.truncate(50_000_001)
 with pytest.raises(ValueError):job_bytes(p)
 p.write_bytes(b'{}');assert job_bytes(p)==b'{}'
 link=tmp_path/'link';link.symlink_to(p)
 with pytest.raises(OSError):job_bytes(link)

def test_malformed_lock_and_sampling_description():
 d=semantic_pilot();j=new_job(model_spec('answerdotai/ModernBERT-base'),d);j['dependency_lock_sha256']=[]
 with pytest.raises(ValueError):validate_job(j)
 d['sampling_description']='x'*8193
 with pytest.raises(ValueError):validate_dataset(d)

def test_invalid_late_tensor_does_not_partially_restore(tmp_path):
 import torch
 from safetensors.torch import save_file
 from saal_rsi.modernbert_backend import ModernBERTBackend
 b=ModernBERTBackend(model_spec('answerdotai/ModernBERT-base',method='head',backend='local',precision='float32'),tmp_path,17)
 b.model=torch.nn.Linear(2,2);b.trainable=dict(b.model.named_parameters());before={n:t.detach().clone() for n,t in b.trainable.items()};state={n:t+1 for n,t in before.items()};state['weight']=torch.full_like(state['weight'],float('nan'));p=tmp_path/'invalid.safetensors';save_file(state,str(p))
 with pytest.raises(ValueError):b.restore_checkpoint(p)
 assert all(torch.equal(before[n],t) for n,t in b.trainable.items())


def test_generative_job_missing_partitions_fails_admission():
    from saal_rsi.learning_dataset import generative_pilot
    d=generative_pilot();d['records']=[r for r in d['records'] if r['split']=='train']
    with pytest.raises(ValueError,match='partition'):new_job(model_spec('Qwen/Qwen3.5-0.8B'),d)

@pytest.mark.parametrize('prediction',[{'decision':[],'probabilities':[0.,1.,0.,0.]},{'decision':'FAST','probabilities':{}},{'decision':None,'probabilities':[0.,1.,0.,0.]}])
def test_malformed_scores_are_counted_as_failure(prediction):
    from saal_rsi.checkpoint_selection import metrics
    r=semantic_pilot()['records'][0];m=metrics([r],[prediction],'modernbert-seqcls-v1')
    assert m['invalid_outputs']==1 and m['accuracy']==0 and m['confusion']=={r['expected_decision']+'->INVALID':1}
