import copy
import pytest
from saal_rsi.neural_contract import ModelSpec,new_transformer_job,validate_neural_job
from saal_rsi.research_datasets import semantic_benchmark

@pytest.fixture
def job():return new_transformer_job(semantic_benchmark(2,1),baseline_steps=1,adaptation_steps=1)

def test_v4_version_and_long_input(job):
    assert job['schema']=='saal-model-job-v4'
    from saal_rsi.dataset_registry import manifest
    from saal_rsi.spec import sha
    records=copy.deepcopy(job['dataset']['records']);records[0]['prompt']+=' The final user correction must be retained.'
    job['dataset']=manifest(records);job['dataset_digest']=sha(job['dataset'])
    assert len(records[0]['prompt'].encode())>128
    from saal_rsi.model_contract import validate_job
    validate_job(job)

@pytest.mark.parametrize('field,value',[('max_sequence_bytes',1024),('tokenizer_version','normalized'),('positional_encoding','none'),('output_vocabulary',['CHEAP','FAST','ASK','REFUSE','FORBIDDEN']),('initialization_seed',True),('dropout',float('nan'))])
def test_architecture_substitution(job,field,value):
    job['model_spec'][field]=value
    with pytest.raises(ValueError):validate_neural_job(job)

@pytest.mark.parametrize('rank',[0,1,3,16,True])
def test_invalid_rank(job,rank):
    job['model_spec']['adapter_configuration']['rank']=rank
    with pytest.raises(ValueError):validate_neural_job(job)

@pytest.mark.parametrize('key,value',[('max_seconds',True),('external_deployment_allowed',True),('mode','execute'),('baseline_steps',-1),('adaptation_steps',5000),('class_balance','magic')])
def test_malformed_job(job,key,value):
    job[key]=value
    with pytest.raises(ValueError):validate_neural_job(job)

def test_missing_class_and_protected_data(job):
    from saal_rsi.dataset_registry import manifest
    from saal_rsi.spec import sha
    records=[r for r in job['dataset']['records'] if r['expected_decision']!='FAST'];job['dataset']=manifest(records);job['dataset_digest']=sha(job['dataset'])
    with pytest.raises(ValueError,match='coverage'):validate_neural_job(job)
    job['declared_class_ablation']=True;validate_neural_job(job)
    job['dataset']=manifest(records,origin='custodian-sequestered');job['dataset_digest']=sha(job['dataset'])
    with pytest.raises(ValueError,match='protected'):validate_neural_job(job)
