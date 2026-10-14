import pytest

from saal_rsi.adaptive_contract import adapter_spec, new_job
from saal_rsi.adaptive_dataset import training_projection
from saal_rsi.learning_dataset import generative_pilot, wrap
from saal_rsi.pretrained_contract import model_spec


@pytest.fixture(scope='session')
def population():
    return wrap([r for r in generative_pilot()['records'] if r['task_type'] == 'decision'])


@pytest.fixture
def admitted_job(population):
    spec = model_spec('answerdotai/ModernBERT-base', method='none', backend='local', precision='float32', budget='adaptive_test')
    budget = {'campaign_id': 'test_campaign', 'run_id': 'test_run', 'max_runs': 3,
              'total_lifecycle_seconds': 15000, 'reserved_seconds': 5000,
              'automatic_training_retries': 0, 'financial_cap': False}
    return new_job(spec, training_projection(population),
                   adapter_spec('adalora-12-to-8', total_steps=4, tinit=1, tfinal=1, delta=1),
                   seed=60606, budget=budget, steps=4, max_seconds=3600,
                   campaign_stage='ENGINEERING', checkpoint_interval=2)
