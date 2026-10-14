"""Additive ModernBERT adapter jobs; legacy contracts remain immutable.

The model specification describes an unadapted official snapshot. Adapter
admission is a separate, closed contract, including fixed rank 12 and AdaLoRA.
No validation projection is executed as a legacy job.
"""
import hashlib
import math
import re
from pathlib import Path

from .pretrained_contract import PretrainedModelSpec, VERSIONS, targets, registry_path
from .spec import sha

SCHEMA = 'saal-adaptive-job-v1'
SELECTION = 'public-guardrails-pair-family-v1'
PEFT_IMPLEMENTATION = {
    'config.py': 'f1b2eea2e36c7c69c5ab3a91cde4b851e6ddf40a4a806726c2ee89af23dcdc88',
    'layer.py': 'fea21a311da946eceb85d7d72e9bf36cf71dc090a1b2a452cae5002d8d7c8701',
    'model.py': '39d950eeacef3e37c30b62363edebd05408940205cff78d948f470e9b43c5c49',
}
SOURCES = (
    'spec.py', 'validation.py', 'model_contract.py', 'pretrained_contract.py',
    'pretrained_registry.py', 'learning_backend.py', 'modernbert_backend.py',
    'learning_dataset.py', 'task_oracles.py', 'checkpoint_selection.py',
    'epoch_metrics.py', 'epoch_sampler.py', 'adaptive_contract.py',
    'adaptive_dataset.py', 'adaptive_backend.py', 'adaptive_rank_controller.py',
    'adaptive_training.py', 'adaptive_artifacts.py', 'adaptive_evaluator.py',
    'adaptive_worker.py',
)


def worker_identity():
    base = Path(__file__).parent
    files = {n: hashlib.sha256((base/n).read_bytes()).hexdigest() for n in SOURCES}
    for n in ('registry.json', 'tensor_inventories.json'):
        files['configs/'+n] = hashlib.sha256(registry_path().with_name(n).read_bytes()).hexdigest()
    return sha(files)


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('bounded integer: '+name)


def rate(value, low, high, name):
    if type(value) is not float or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('bounded real: '+name)


def adapter_spec(method, *, total_steps=1000, tinit=100, tfinal=200, delta=10):
    if method not in ('fixed-r8', 'fixed-r12', 'adalora-12-to-8'):
        raise ValueError('unregistered adapter method')
    dynamic = method == 'adalora-12-to-8'
    rank = 8 if method == 'fixed-r8' else 12
    return {
        'schema': 'saal-adapter-spec-v1', 'method': method, 'init_r': rank,
        'target_r': 8 if dynamic else rank, 'lora_alpha': 2*rank,
        'dropout': 0.05, 'bias': 'none',
        'target_modules': targets('modernbert-seqcls-v1'),
        'modules_to_save': ['head', 'classifier'],
        'schedule': {'total_step': total_steps, 'tinit': tinit, 'tfinal': tfinal,
                     'deltaT': delta, 'beta1': .85, 'beta2': .85,
                     'orth_reg_weight': .5} if dynamic else None,
        'serialization': 'full-allocated-tensors-and-mask-v1',
        'mask_policy': 'persist-after-completed-update-v1' if dynamic else 'fixed-no-mask-v1',
    }


def validate_adapter(a, steps):
    fields = set(adapter_spec('fixed-r8'))
    if type(a) is not dict or set(a) != fields:
        raise ValueError('closed adapter specification')
    method = a['method']
    schedule = a['schedule']
    if method == 'adalora-12-to-8':
        if type(schedule) is not dict or set(schedule) != {'total_step', 'tinit', 'tfinal', 'deltaT', 'beta1', 'beta2', 'orth_reg_weight'}:
            raise ValueError('adaptive schedule required')
        integer(schedule['total_step'], 3, 1000, 'allocation total')
        integer(schedule['tinit'], 1, steps-2, 'allocation initialization')
        integer(schedule['tfinal'], 1, steps-2, 'allocation stabilization')
        integer(schedule['deltaT'], 1, steps, 'allocation frequency')
        if schedule['total_step'] != steps or schedule['tinit']+schedule['tfinal'] >= steps:
            raise ValueError('schedule / completed update budget mismatch')
        if steps-schedule['tfinal'] <= schedule['tinit']+schedule['deltaT']:
            raise ValueError('schedule has no redistribution interval')
        expected = adapter_spec(method, total_steps=steps, tinit=schedule['tinit'],
                                tfinal=schedule['tfinal'], delta=schedule['deltaT'])
    else:
        expected = adapter_spec(method)
    # Exact values and types prevent bool-as-rank and float-as-counter admission.
    from .spec import canonical
    if canonical(a) != canonical(expected):
        raise ValueError('adapter identity, targets, head policy or schedule substitution')
    return a


def validate_budget(budget, job):
    required = {'campaign_id', 'run_id', 'max_runs', 'total_lifecycle_seconds',
                'reserved_seconds', 'automatic_training_retries', 'financial_cap'}
    if type(budget) is not dict or set(budget) != required:
        raise ValueError('complete experiment budget required')
    for key in ('campaign_id', 'run_id'):
        if type(budget[key]) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,80}', budget[key]):
            raise ValueError('budget/run identity')
    integer(budget['max_runs'], 1, 25, 'run quota')
    integer(budget['reserved_seconds'], job['max_seconds']+1400, 15000, 'lifecycle reservation')
    integer(budget['total_lifecycle_seconds'], budget['reserved_seconds'], 100000, 'campaign seconds')
    if type(budget['automatic_training_retries']) is not int or budget['automatic_training_retries'] != 0 or budget['financial_cap'] is not False:
        raise ValueError('retry or billing authority escalation')


def validate_job(job):
    fields = {'schema', 'model_spec', 'adapter', 'dataset', 'dataset_digest', 'seed',
              'sampler_seed', 'dropout_seed', 'steps', 'batch_size', 'accumulation',
              'learning_rate', 'head_learning_rate', 'max_seconds', 'class_balance',
              'checkpoint_interval', 'selection_policy', 'worker_identity',
              'peft_implementation', 'dependency_lock_sha256', 'budget',
              'campaign_stage', 'qualification_status', 'external_deployment_allowed',
              'final_holdout_included', 'resume_policy', 'replay_tolerance'}
    if type(job) is not dict or set(job) != fields or job['schema'] != SCHEMA:
        raise ValueError('adaptive job schema')
    spec = PretrainedModelSpec.parse(job['model_spec'])
    if spec.model_kind != 'modernbert-seqcls-v1' or spec.adapter_type != 'none':
        raise ValueError('official unadapted ModernBERT specification required')
    for key in ('seed', 'sampler_seed', 'dropout_seed'):
        integer(job[key], 0, 2**31-1, key)
    for key, lo, hi in [('steps', 4, 1000), ('batch_size', 4, 32), ('accumulation', 1, 16),
                         ('max_seconds', 30, 7200), ('checkpoint_interval', 1, 200)]:
        integer(job[key], lo, hi, key)
    if job['batch_size'] % 4:
        raise ValueError('balanced four-class microbatch required')
    for key in ('learning_rate', 'head_learning_rate'):
        rate(job[key], 1e-7, .01, key)
    if type(job['class_balance']) is not bool or job['selection_policy'] != SELECTION:
        raise ValueError('loss/selection policy')
    validate_adapter(job['adapter'], job['steps'])
    from .adaptive_dataset import validate_training_data
    validate_training_data(job['dataset'])
    if job['dataset_digest'] != sha(job['dataset']):
        raise ValueError('training data substitution')
    if job['worker_identity'] != worker_identity() or job['peft_implementation'] != PEFT_IMPLEMENTATION:
        raise ValueError('worker / PEFT implementation substitution')
    lock = Path(__file__).resolve().parents[2]/'configs/dependency-locks'/(
        'pretrained-linux-x86_64-py313.txt' if spec.compute_backend == 'colab' else 'pretrained-macos-arm64-py312.txt')
    if job['dependency_lock_sha256'] != hashlib.sha256(lock.read_bytes()).hexdigest():
        raise ValueError('platform hash lock substitution')
    if job['campaign_stage'] not in ('ENGINEERING', 'PILOT', 'STUDY', 'DATA_CONTROL'):
        raise ValueError('campaign phase')
    if job['campaign_stage'] != 'ENGINEERING':
        if spec.compute_backend != 'colab' or job['steps'] != 1000 or job['batch_size']*job['accumulation'] != 16:
            raise ValueError('matched T4 campaign requires 1000 updates / 16000 exposures')
        if job['adapter']['method'] == 'adalora-12-to-8' and job['adapter']['schedule'] != adapter_spec('adalora-12-to-8')['schedule']:
            raise ValueError('registered full allocation schedule required')
    validate_budget(job['budget'], job)
    if (job['qualification_status'] != 'NOT_EVALUATED' or job['external_deployment_allowed'] is not False
            or job['final_holdout_included'] is not False or job['resume_policy'] != 'NO_RESUME'
            or type(job['replay_tolerance']) is not float or job['replay_tolerance'] != 1e-4):
        raise ValueError('authority / replay policy escalation')
    return job


def new_job(spec, dataset, adapter, *, seed, budget, steps=1000, batch_size=8,
            accumulation=2, max_seconds=3600, campaign_stage='PILOT',
            learning_rate=.0003, head_learning_rate=.0003, checkpoint_interval=100):
    lock = Path(__file__).resolve().parents[2]/'configs/dependency-locks'/(
        'pretrained-linux-x86_64-py313.txt' if spec['compute_backend'] == 'colab' else 'pretrained-macos-arm64-py312.txt')
    return validate_job({
        'schema': SCHEMA, 'model_spec': spec, 'adapter': adapter, 'dataset': dataset,
        'dataset_digest': sha(dataset), 'seed': seed, 'sampler_seed': seed,
        'dropout_seed': seed, 'steps': steps, 'batch_size': batch_size,
        'accumulation': accumulation, 'learning_rate': learning_rate,
        'head_learning_rate': head_learning_rate, 'max_seconds': max_seconds,
        'class_balance': False, 'checkpoint_interval': checkpoint_interval,
        'selection_policy': SELECTION, 'worker_identity': worker_identity(),
        'peft_implementation': PEFT_IMPLEMENTATION.copy(),
        'dependency_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest(),
        'budget': budget, 'campaign_stage': campaign_stage,
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False,
        'final_holdout_included': False, 'resume_policy': 'NO_RESUME', 'replay_tolerance': 1e-4,
    })
