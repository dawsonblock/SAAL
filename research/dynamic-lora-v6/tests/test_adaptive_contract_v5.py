import copy
import hashlib
from pathlib import Path

import pytest

from saal_rsi.adaptive_contract_v5 import (
    ORTHOGONAL_COEFFICIENTS,
    build_fixed_r8_reference,
    build_static_adalora_pilot,
    scientific_source_digest,
    validate_pilot_config,
)
from saal_rsi.numerical_metrics_v2 import POLICY_DIGEST
from saal_rsi.pretrained_contract import model_spec, targets


def identity():
    digest = 'sha256:' + 'a' * 64
    spec = model_spec('answerdotai/ModernBERT-base', method='none', backend='colab',
                      precision='float16', budget='dynamic-lora-v5-pilot')
    lock = Path('configs/dependency-locks/pretrained-linux-x86_64-py313.txt').read_bytes()
    return {
        'model_spec': spec,
        'model_digest': spec['model_files_digest'],
        'tokenizer_digest': spec['tokenizer_digest'],
        'dataset_digest': digest,
        'source_digest': scientific_source_digest(),
        'dependency_digest': 'sha256:' + hashlib.sha256(lock).hexdigest(),
        'metric_policy_digest': POLICY_DIGEST,
        'sample_order_digest': digest,
        'seed': 29,
        'sampler_seed': 31,
        'dropout_seed': 37,
        'batch_size': 8,
        'accumulation': 2,
        'adapter_learning_rate': 0.0003,
        'head_learning_rate': 0.0003,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'target_modules': targets('modernbert-seqcls-v1'),
        'precision': 'float16',
        'class_balance': False,
    }


@pytest.mark.parametrize('coefficient', ORTHOGONAL_COEFFICIENTS)
def test_four_static_ablation_arms_change_only_regularization(coefficient):
    config = build_static_adalora_pilot(identity(), coefficient)
    assert config['method'] == 'adalora-static-r12'
    assert config['init_rank'] == config['target_rank'] == 12
    assert config['rank_allocation'] == 'disabled-static-control-v1'
    assert config['steps'] == 200
    assert config['evaluation_interval'] == 20
    assert config['sample_exposures'] == 200 * 8 * 2
    assert config['orthogonal_coefficient'] == coefficient
    assert config['authorization_status'] == 'FRESH_SIGNED_ADMISSION_REQUIRED'
    assert config['execution_status'] == 'NOT_EXECUTABLE'
    assert config['qualification_status'] == 'NOT_EVALUATED'
    assert config['external_deployment_allowed'] is False
    assert validate_pilot_config(config) is config


def test_fixed_r8_reference_matches_shared_conditions_without_regularizer():
    baseline = build_fixed_r8_reference(identity())
    candidate = build_static_adalora_pilot(identity(), 0.1)
    assert baseline['method'] == 'fixed-r8'
    assert baseline['init_rank'] == baseline['target_rank'] == 8
    assert baseline['orthogonal_coefficient'] == 0.0
    assert baseline['identity'] == candidate['identity']
    for key in ('steps', 'evaluation_interval', 'sample_exposures', 'optimizer',
                'modules_to_save', 'head_policy', 'checkpoint_policy', 'peft_version'):
        assert baseline[key] == candidate[key]


def test_candidate_is_not_a_v4_worker_job_and_rejects_policy_substitution():
    config = build_static_adalora_pilot(identity(), 0.1)
    assert config['schema'] == 'saal-adaptive-scientific-config-v5'
    assert 'budget' not in config
    changed = copy.deepcopy(config)
    changed['rank_allocation'] = 'dynamic'
    with pytest.raises(ValueError, match='substitution'):
        validate_pilot_config(changed)


def test_scientific_config_rejects_unregistered_or_invalid_input():
    with pytest.raises(ValueError, match='preregistered'):
        build_static_adalora_pilot(identity(), 0.05)
    invalid = identity()
    invalid['dataset_digest'] = 'not-a-digest'
    with pytest.raises(ValueError, match='digest'):
        build_static_adalora_pilot(invalid, 0.1)
    invalid_source = identity()
    invalid_source['source_digest'] = 'sha256:' + 'b' * 64
    with pytest.raises(ValueError, match='source identity'):
        build_static_adalora_pilot(invalid_source, 0.1)
    invalid_metric = identity()
    invalid_metric['metric_policy_digest'] = 'sha256:' + 'c' * 64
    with pytest.raises(ValueError, match='numerical metric policy'):
        build_static_adalora_pilot(invalid_metric, 0.1)
    with pytest.raises(ValueError, match='closed'):
        validate_pilot_config({'schema': 'saal-adaptive-job-v4'})
    changed = build_fixed_r8_reference(identity())
    changed['orthogonal_coefficient'] = 0.01
    with pytest.raises(ValueError, match='orthogonal coefficient'):
        validate_pilot_config(changed)
