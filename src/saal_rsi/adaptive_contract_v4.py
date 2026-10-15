"""Fresh v2-metric admission; never reinterpret v1-v3 jobs or receipts."""
import copy
import hashlib
from pathlib import Path

from . import adaptive_contract_v3 as v3
from .numerical_metrics_v2 import POLICY, POLICY_DIGEST, validate_policy
from .spec import sha

SCHEMA = 'saal-adaptive-job-v4'
RANKNUM_POLICY = v3.RANKNUM_POLICY
METRIC_POLICY = POLICY
SOURCES = v3.SOURCES + (
    'numerical_metrics_v2.py', 'adaptive_contract_v4.py',
    'adaptive_training_v4.py', 'adaptive_artifacts_v4.py', 'adaptive_worker_v4.py',
)


def worker_identity():
    base = Path(__file__).parent
    files = {name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in SOURCES}
    for name in ('registry.json', 'tensor_inventories.json'):
        files['configs/'+name] = hashlib.sha256(v3.v1.registry_path().with_name(name).read_bytes()).hexdigest()
    return sha(files)


def validate_job(job):
    if (type(job) is not dict or job.get('schema') != SCHEMA
            or job.get('ranknum_policy') != RANKNUM_POLICY
            or job.get('worker_identity') != worker_identity()
            or job.get('numerical_metric_policy_digest') != POLICY_DIGEST):
        raise ValueError('fresh v2-metric v4 admission required')
    validate_policy(job.get('numerical_metric_policy'))
    constraints = dict(job)
    for name in ('numerical_metric_policy', 'numerical_metric_policy_digest'):
        constraints.pop(name)
    constraints.update(schema=v3.SCHEMA, worker_identity=v3.worker_identity())
    constraints.update(numerical_metric_policy=copy.deepcopy(v3.POLICY),
                       numerical_metric_policy_digest=v3.POLICY_DIGEST)
    v3.validate_job(constraints)
    return job


def new_job(*args, **kwargs):
    job = v3.new_job(*args, **kwargs)
    job.update(schema=SCHEMA, numerical_metric_policy=copy.deepcopy(POLICY),
               numerical_metric_policy_digest=POLICY_DIGEST)
    job['worker_identity'] = worker_identity()
    return validate_job(job)
