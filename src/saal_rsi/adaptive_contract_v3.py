"""Fresh numerical-policy admissions; never reinterpret consumed v1/v2 jobs."""
import copy
import hashlib
from pathlib import Path

from . import adaptive_contract as v1
from . import adaptive_contract_v2 as v2
from .numerical_metrics_v1 import POLICY, POLICY_DIGEST, validate_policy
from .spec import sha

SCHEMA = 'saal-adaptive-job-v3'
RANKNUM_POLICY = v2.RANKNUM_POLICY
METRIC_POLICY = POLICY
SOURCES = v2.SOURCES + (
    'numerical_metrics_v1.py', 'adaptive_contract_v3.py',
    'adaptive_training_v3.py', 'adaptive_artifacts_v3.py', 'adaptive_worker_v3.py',
)


def worker_identity():
    base = Path(__file__).parent
    files = {n: hashlib.sha256((base/n).read_bytes()).hexdigest() for n in SOURCES}
    for name in ('registry.json', 'tensor_inventories.json'):
        files['configs/'+name] = hashlib.sha256(v1.registry_path().with_name(name).read_bytes()).hexdigest()
    return sha(files)


def validate_job(job):
    if (type(job) is not dict or job.get('schema') != SCHEMA
            or job.get('ranknum_policy') != RANKNUM_POLICY
            or job.get('worker_identity') != worker_identity()
            or job.get('numerical_metric_policy_digest') != POLICY_DIGEST):
        raise ValueError('fresh numerical-policy v3 admission required')
    validate_policy(job.get('numerical_metric_policy'))
    # Only unsigned constraints are projected. No legacy signature or consumed
    # attempt can authorize this new worker, metrics schema or output inventory.
    constraints = dict(job)
    for name in ('ranknum_policy', 'numerical_metric_policy', 'numerical_metric_policy_digest'):
        constraints.pop(name)
    constraints.update(schema=v1.SCHEMA, worker_identity=v1.worker_identity())
    v1.validate_job(constraints)
    return job


def new_job(*args, **kwargs):
    job = v1.new_job(*args, **kwargs)
    job.update(schema=SCHEMA, ranknum_policy=RANKNUM_POLICY,
               numerical_metric_policy=copy.deepcopy(POLICY),
               numerical_metric_policy_digest=POLICY_DIGEST,
               worker_identity=worker_identity())
    return validate_job(job)
