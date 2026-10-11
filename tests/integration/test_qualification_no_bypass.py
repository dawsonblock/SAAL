import pytest
from saal_rsi.neural_contract import new_transformer_job
from saal_rsi.research_datasets import semantic_benchmark
from saal_rsi.model_evaluation import sealed_plan
from saal_rsi.model_contract import actor_proposal
from saal_rsi.family_statistics import compare_families

def test_v4_cannot_use_legacy_case_number_qualification():
    job=new_transformer_job(semantic_benchmark(2,1),baseline_steps=1,adaptation_steps=1)
    with pytest.raises(ValueError,match='family-level'):sealed_plan(job,'unused',[])

def test_perfect_public_outputs_do_not_qualify():
    records=[r for r in semantic_benchmark(2,1)['records'] if r['split']=='development']
    labels=[r['expected_decision'] for r in records]
    metrics=compare_families(records,['CHEAP']*len(records),labels,.0125)
    assert metrics['research_qualified'] is False
    assert actor_proposal('FAST')['kind']=='PROPOSE'
    assert 'token' not in actor_proposal('FAST')
