"""Unadmitted v5 scientific configuration for static AdaLoRA diagnosis.

This is deliberately not a worker/job contract.  A returned configuration is
only a frozen candidate description and cannot authorize training.  v4 jobs
and their signatures keep their original interpretation.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path

SCHEMA = 'saal-adaptive-scientific-config-v5'
PILOT_ID = 'static-adalora-orthogonal-regularization-screen-v1'
ORTHOGONAL_COEFFICIENTS = (0.0, 0.01, 0.1, 0.5)
PILOT_STEPS = 200
EVALUATION_INTERVAL = 20
PEFT_VERSION = '0.21.2'
_DIGEST = re.compile(r'^sha256:[0-9a-f]{64}$')
_IDENTITY_FIELDS = {
    'model_spec', 'model_digest', 'tokenizer_digest', 'dataset_digest', 'source_digest',
    'dependency_digest', 'sample_order_digest', 'seed', 'sampler_seed',
    'dropout_seed', 'batch_size', 'accumulation', 'adapter_learning_rate',
    'head_learning_rate', 'weight_decay', 'gradient_clip', 'target_modules',
    'precision', 'class_balance',
}
_CONFIG_FIELDS = {
    'schema', 'protocol_id', 'variant_id', 'identity', 'method', 'init_rank',
    'target_rank', 'lora_alpha', 'lora_dropout', 'modules_to_save',
    'head_policy', 'rank_allocation', 'orthogonal_coefficient', 'steps',
    'evaluation_interval', 'sample_exposures', 'optimizer', 'initialization',
    'scaling', 'checkpoint_policy', 'peft_version', 'qualification_status',
    'external_deployment_allowed', 'final_holdout_included', 'resume_policy',
    'automatic_retries', 'execution_status', 'authorization_status',
}
SCIENTIFIC_SOURCE_FILES = (
    'adaptive_contract_v5.py',
    'dynamic_lora_v5/__init__.py',
    'dynamic_lora_v5/svd_reference.py',
    'pretrained_contract.py',
    'spec.py',
)


def _positive_float(value, field, *, allow_zero=False):
    lower_ok = value >= 0 if allow_zero else value > 0
    if type(value) is not float or not math.isfinite(value) or not lower_ok:
        raise ValueError(f'invalid finite float: {field}')


def _digest(value, field):
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f'invalid content digest: {field}')


def scientific_source_digest():
    """Hash the diagnostic/configuration code; this is not a worker closure."""
    base = Path(__file__).parent
    files = {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
             for name in SCIENTIFIC_SOURCE_FILES}
    registry = base.parents[1] / 'configs/pretrained/registry.json'
    files['configs/pretrained/registry.json'] = hashlib.sha256(registry.read_bytes()).hexdigest()
    from .spec import sha
    return sha(files)


def _validate_identity(identity):
    if type(identity) is not dict or set(identity) != _IDENTITY_FIELDS:
        raise ValueError('closed scientific input identity required')
    for field in ('model_digest', 'tokenizer_digest', 'dataset_digest',
                  'source_digest', 'dependency_digest', 'sample_order_digest'):
        _digest(identity[field], field)
    if identity['source_digest'] != scientific_source_digest():
        raise ValueError('scientific configuration source identity mismatch')
    from .pretrained_contract import PretrainedModelSpec, targets
    spec = PretrainedModelSpec.parse(identity['model_spec'])
    if spec.model_kind != 'modernbert-seqcls-v1' or spec.adapter_type != 'none':
        raise ValueError('frozen ModernBERT base snapshot required')
    if identity['model_digest'] != spec.model_files_digest:
        raise ValueError('model digest differs from immutable model revision')
    if identity['tokenizer_digest'] != spec.tokenizer_digest:
        raise ValueError('tokenizer digest differs from immutable model revision')
    if identity['target_modules'] != targets(spec.model_kind):
        raise ValueError('registered ModernBERT AdaLoRA target inventory required')
    if identity['precision'] != spec.precision:
        raise ValueError('precision differs from immutable model specification')
    lock_name = ('pretrained-linux-x86_64-py313.txt' if spec.compute_backend == 'colab'
                 else 'pretrained-macos-arm64-py312.txt')
    lock_path = Path(__file__).resolve().parents[2] / 'configs/dependency-locks' / lock_name
    actual_lock = 'sha256:' + hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if identity['dependency_digest'] != actual_lock:
        raise ValueError('dependency digest differs from platform hash lock')
    for field, low, high in (('seed', 0, 2**31-1), ('batch_size', 1, 4096),
                             ('accumulation', 1, 1024),
                             ('sampler_seed', 0, 2**31-1),
                             ('dropout_seed', 0, 2**31-1)):
        value = identity[field]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'invalid integer: {field}')
    for field in ('adapter_learning_rate', 'head_learning_rate', 'gradient_clip'):
        _positive_float(identity[field], field)
    _positive_float(identity['weight_decay'], 'weight_decay', allow_zero=True)
    if (type(identity['target_modules']) is not list or not identity['target_modules']
            or any(type(item) is not str or not item for item in identity['target_modules'])
            or len(set(identity['target_modules'])) != len(identity['target_modules'])):
        raise ValueError('unique target module list required')
    if identity['precision'] not in ('float32', 'bfloat16', 'float16'):
        raise ValueError('unsupported precision identity')
    if type(identity['class_balance']) is not bool:
        raise ValueError('Boolean class-balance policy required')
    return identity


def _make_config(identity, orthogonal_coefficient, method='adalora-static-r12'):
    if method not in ('fixed-r8', 'adalora-static-r12'):
        raise ValueError('pilot supports only fixed-r8 and static AdaLoRA r12')
    if method == 'fixed-r8' and orthogonal_coefficient != 0.0:
        raise ValueError('orthogonal coefficient applies only to AdaLoRA')
    rank = 8 if method == 'fixed-r8' else 12
    exposures = PILOT_STEPS * identity['batch_size'] * identity['accumulation']
    return {
        'schema': SCHEMA,
        'protocol_id': PILOT_ID,
        'variant_id': 'fixed-r8-reference' if method == 'fixed-r8' else f'orth-reg-{orthogonal_coefficient:g}',
        'identity': copy.deepcopy(identity),
        'method': method,
        'init_rank': rank,
        'target_rank': rank,
        'lora_alpha': 2 * rank,
        'lora_dropout': 0.05,
        'modules_to_save': ['head', 'classifier'],
        'head_policy': 'train-head-with-adapter-v1',
        'rank_allocation': 'disabled-static-control-v1',
        'orthogonal_coefficient': orthogonal_coefficient,
        'steps': PILOT_STEPS,
        'evaluation_interval': EVALUATION_INTERVAL,
        'sample_exposures': exposures,
        'optimizer': {
            'name': 'AdamW',
            'adapter_learning_rate': identity['adapter_learning_rate'],
            'head_learning_rate': identity['head_learning_rate'],
            'weight_decay': identity['weight_decay'],
            'gradient_clip': identity['gradient_clip'],
            'scheduler': 'none',
        },
        'initialization': ('peft-0.21.2-lora-default-zero-output-v1' if method == 'fixed-r8'
                           else 'peft-0.21.2-zero-singular-values-v1'),
        'scaling': ('peft-0.21.2-alpha-over-r-v1' if method == 'fixed-r8'
                    else 'peft-0.21.2-stored-ranknum-v1'),
        'checkpoint_policy': 'initial-every-20-best-terminal-v1',
        'peft_version': PEFT_VERSION,
        'qualification_status': 'NOT_EVALUATED',
        'external_deployment_allowed': False,
        'final_holdout_included': False,
        'resume_policy': 'NO_RESUME',
        'automatic_retries': 0,
        'execution_status': 'NOT_EXECUTABLE',
        'authorization_status': 'FRESH_SIGNED_ADMISSION_REQUIRED',
    }


def build_static_adalora_pilot(identity, orthogonal_coefficient):
    """Build one non-authorizing arm; all shared conditions come from identity."""
    _validate_identity(identity)
    if type(orthogonal_coefficient) is not float or orthogonal_coefficient not in ORTHOGONAL_COEFFICIENTS:
        raise ValueError('coefficient is outside the preregistered four-arm screen')
    return validate_pilot_config(_make_config(identity, orthogonal_coefficient))


def build_fixed_r8_reference(identity):
    """Build the matched fixed-r8 reference; this still does not authorize execution."""
    _validate_identity(identity)
    return validate_pilot_config(_make_config(identity, 0.0, method='fixed-r8'))


def validate_pilot_config(config):
    if type(config) is not dict or set(config) != _CONFIG_FIELDS:
        raise ValueError('closed static AdaLoRA scientific configuration required')
    identity = _validate_identity(config['identity'])
    coefficient = config['orthogonal_coefficient']
    method = config.get('method')
    if type(coefficient) is not float or (method == 'adalora-static-r12'
                                           and coefficient not in ORTHOGONAL_COEFFICIENTS):
        raise ValueError('unregistered orthogonal coefficient')
    if method == 'fixed-r8' and coefficient != 0.0:
        raise ValueError('fixed LoRA reference cannot have an orthogonal coefficient')
    expected = _make_config(identity, coefficient, method=method)
    # Compare canonical representations to reject bool/int aliases and extras.
    from .spec import canonical
    if canonical(config) != canonical(expected):
        raise ValueError('static AdaLoRA pilot identity or policy substitution')
    return config
