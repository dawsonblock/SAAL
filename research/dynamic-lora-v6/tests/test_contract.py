import copy

import pytest

from saal_rsi.adaptive_contract import validate_job, adapter_spec
from saal_rsi.adaptive_dataset import training_projection, validate_training_data
from saal_rsi.pretrained_contract import validate_job as old_validate
from saal_rsi.spec import sha


def test_projection_contains_only_public_training_and_development(population):
    d = training_projection(population)
    assert {r['split'] for r in d['records']} == {'train', 'development'}
    assert not set(r['record_id'] for r in d['records']) & set(r['record_id'] for r in population['records'] if r['split'] == 'regression')
    assert d['parent_population_digest'] == sha(population)
    assert d['independent_population'] is False


def test_new_jobs_cannot_be_reinterpreted_as_legacy(admitted_job):
    assert validate_job(admitted_job) == admitted_job
    with pytest.raises(ValueError):
        old_validate(admitted_job)
    assert admitted_job['model_spec']['adapter_type'] == 'none'
    assert admitted_job['adapter']['modules_to_save'] == ['head', 'classifier']


@pytest.mark.parametrize('fault', ['no_schedule', 'schedule_step', 'schedule_bool', 'rank', 'targets',
    'head', 'singular_format', 'worker', 'dependency', 'budget_missing', 'resume', 'promote',
    'holdout', 'seed_bool', 'steps_bool', 'rate_nan', 'precision', 'baseline_adapter', 'unknown',
    'protected_record', 'regression_record', 'tensor_source', 'alpha'])
def test_closed_admission_negative_cases(admitted_job, fault):
    j = copy.deepcopy(admitted_job)
    if fault == 'no_schedule': j['adapter']['schedule'] = None
    elif fault == 'schedule_step': j['adapter']['schedule']['total_step'] += 1
    elif fault == 'schedule_bool': j['adapter']['schedule']['tinit'] = True
    elif fault == 'rank': j['adapter']['init_r'] = 16
    elif fault == 'targets': j['adapter']['target_modules'].append('model.embeddings')
    elif fault == 'head': j['adapter']['modules_to_save'] = ['classifier']
    elif fault == 'singular_format': j['adapter']['serialization'] = 'fixed-lora-v1'
    elif fault == 'worker': j['worker_identity'] = 'sha256:'+'0'*64
    elif fault == 'dependency': j['peft_implementation']['layer.py'] = '0'*64
    elif fault == 'budget_missing': del j['budget']['total_lifecycle_seconds']
    elif fault == 'resume': j['resume_policy'] = 'RETRY'
    elif fault == 'promote': j['external_deployment_allowed'] = True
    elif fault == 'holdout': j['final_holdout_included'] = True
    elif fault == 'seed_bool': j['seed'] = True
    elif fault == 'steps_bool': j['steps'] = True
    elif fault == 'rate_nan': j['learning_rate'] = float('nan')
    elif fault == 'precision': j['model_spec']['precision'] = 'int4'
    elif fault == 'baseline_adapter': j['model_spec']['adapter_type'] = 'lora'
    elif fault == 'unknown': j['execute_python'] = 'print(1)'
    elif fault in ('protected_record', 'regression_record'):
        j['dataset']['records'][0]['split'] = 'holdout' if fault == 'protected_record' else 'regression'
        j['dataset_digest'] = sha(j['dataset'])
    elif fault == 'tensor_source': j['model_spec']['model_files_digest'] = 'sha256:'+'0'*64
    else: j['adapter']['lora_alpha'] = 48
    with pytest.raises(ValueError): validate_job(j)


@pytest.mark.parametrize('fault', ['parent', 'duplicate', 'family', 'label', 'independence'])
def test_training_data_lineage_leakage(admitted_job, fault):
    d = copy.deepcopy(admitted_job['dataset'])
    if fault == 'parent': d['records'][0]['parent_record_id'] = 'missing'
    elif fault == 'duplicate': d['records'][1]['prompt'] = d['records'][0]['prompt'].upper()
    elif fault == 'family':
        train = next(r for r in d['records'] if r['split'] == 'train')
        next(r for r in d['records'] if r['split'] == 'development')['task_family_id'] = train['task_family_id']
    elif fault == 'label': d['records'][0]['expected_decision'] = 'FORBIDDEN'
    else: d['independent_population'] = True
    with pytest.raises(ValueError): validate_training_data(d)


@pytest.mark.parametrize('method,rank,target', [('fixed-r8', 8, 8), ('fixed-r12', 12, 12), ('adalora-12-to-8', 12, 8)])
def test_registered_comparison_capacities(method, rank, target):
    a = adapter_spec(method)
    assert a['init_r'] == rank and a['target_r'] == target
    assert len(a['target_modules']) == 22
    assert a['modules_to_save'] == ['head', 'classifier']
