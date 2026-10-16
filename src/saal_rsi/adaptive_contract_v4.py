"""Fresh v2-metric admission; never reinterpret v1-v3 jobs or receipts."""
import copy
import hashlib
from pathlib import Path

from . import adaptive_contract_v3 as v3
from .numerical_metrics_v2 import POLICY, POLICY_DIGEST, validate_policy
from .spec import canonical, sha

SCHEMA = 'saal-adaptive-job-v4'
RANKNUM_POLICY = v3.RANKNUM_POLICY
METRIC_POLICY = POLICY
SOURCES = v3.SOURCES + (
    'numerical_metrics_v2.py', 'adaptive_contract_v4.py',
    'adaptive_training_v4.py', 'adaptive_artifacts_v4.py', 'adaptive_worker_v4.py',
    'adaptive_backend_v4.py', 'adaptive_backend_v2_v4.py',
    'adaptive_rank_controller_v4.py', 'adaptive_replay_v4.py',
)


def worker_identity():
    base = Path(__file__).parent
    files = {name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in SOURCES}
    for name in ('registry.json', 'tensor_inventories.json'):
        files['configs/'+name] = hashlib.sha256(v3.v1.registry_path().with_name(name).read_bytes()).hexdigest()
    return sha(files)


def adapter_spec(method, *, total_steps=1000, tinit=100, tfinal=200, delta=10):
    """Return an exact v4 adapter contract, including a no-allocation AdaLoRA arm."""
    if method == 'adalora-static-r12':
        value = v3.v1.adapter_spec('adalora-12-to-8', total_steps=total_steps,
                                   tinit=tinit, tfinal=tfinal, delta=delta)
        value.update(method=method, target_r=12, mask_policy='fixed-no-reallocation-v1')
        return value
    return v3.v1.adapter_spec(method, total_steps=total_steps, tinit=tinit,
                              tfinal=tfinal, delta=delta)


def validate_adapter(adapter, steps):
    if type(adapter) is not dict:
        raise ValueError('v4 adapter specification required')
    if adapter.get('method') != 'adalora-static-r12':
        return v3.v1.validate_adapter(adapter, steps)
    schedule = adapter.get('schedule')
    if type(schedule) is not dict or set(schedule) != {
            'total_step', 'tinit', 'tfinal', 'deltaT', 'beta1', 'beta2', 'orth_reg_weight'}:
        raise ValueError('static AdaLoRA schedule identity required')
    dynamic = v3.v1.adapter_spec('adalora-12-to-8', total_steps=steps,
                                 tinit=schedule.get('tinit'), tfinal=schedule.get('tfinal'),
                                 delta=schedule.get('deltaT'))
    v3.v1.validate_adapter(dynamic, steps)
    expected = adapter_spec('adalora-static-r12', total_steps=steps,
                            tinit=schedule['tinit'], tfinal=schedule['tfinal'],
                            delta=schedule['deltaT'])
    if canonical(adapter) != canonical(expected):
        raise ValueError('static AdaLoRA tensor or no-reallocation policy substitution')
    return adapter


def validate_job(job):
    if (type(job) is not dict or job.get('schema') != SCHEMA
            or job.get('ranknum_policy') != RANKNUM_POLICY
            or job.get('worker_identity') != worker_identity()
            or job.get('numerical_metric_policy_digest') != POLICY_DIGEST):
        raise ValueError('fresh v2-metric v4 admission required')
    validate_policy(job.get('numerical_metric_policy'))
    validate_adapter(job.get('adapter'), job.get('steps'))
    constraints = dict(job)
    for name in ('numerical_metric_policy', 'numerical_metric_policy_digest'):
        constraints.pop(name)
    constraints.update(schema=v3.SCHEMA, worker_identity=v3.worker_identity())
    if job['adapter']['method'] == 'adalora-static-r12':
        constraints['adapter'] = v3.v1.adapter_spec('fixed-r12')
    constraints.update(numerical_metric_policy=copy.deepcopy(v3.POLICY),
                       numerical_metric_policy_digest=v3.POLICY_DIGEST)
    v3.validate_job(constraints)
    return job


def new_job(*args, **kwargs):
    if args:
        if len(args) != 3 or 'adapter' in kwargs:
            raise TypeError('v4 new_job expects spec, dataset, adapter')
        spec, dataset, adapter = args
    else:
        try:
            spec, dataset, adapter = kwargs.pop('spec'), kwargs.pop('dataset'), kwargs.pop('adapter')
        except KeyError as exc:
            raise TypeError('v4 new_job requires spec, dataset and adapter') from exc
    validate_adapter(adapter, kwargs.get('steps', 1000))
    legacy_adapter = v3.v1.adapter_spec('fixed-r12') if adapter['method'] == 'adalora-static-r12' else adapter
    job = v3.new_job(spec, dataset, legacy_adapter, **kwargs)
    job['adapter'] = copy.deepcopy(adapter)
    job.update(schema=SCHEMA, numerical_metric_policy=copy.deepcopy(POLICY),
               numerical_metric_policy_digest=POLICY_DIGEST)
    job['worker_identity'] = worker_identity()
    return validate_job(job)
