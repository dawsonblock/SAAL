"""New signed worker identity for CUDA-safe AdaLoRA; v1 stays immutable.

Legacy validation is reused only for parameter constraints on an unsigned
copy. No v2 job or checkpoint is ever executed as a v1 admission.
"""
import hashlib
from pathlib import Path

from . import adaptive_contract as v1
from .spec import sha

SCHEMA = 'saal-adaptive-job-v2'
RANKNUM_POLICY = 'freeze-scaling-after-peft-device-placement-v1'
SOURCES = v1.SOURCES+('adaptive_contract_v2.py', 'adaptive_backend_v2.py',
                      'adaptive_artifacts_v2.py', 'adaptive_worker_v2.py')


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
            or type(job.get('adapter')) is not dict
            or job.get('adapter', {}).get('method') != 'adalora-12-to-8'):
        raise ValueError('closed CUDA-safe AdaLoRA v2 admission required')
    constraints = dict(job)
    constraints.pop('ranknum_policy')
    constraints.update(schema=v1.SCHEMA, worker_identity=v1.worker_identity())
    v1.validate_job(constraints)
    return job


def new_job(*args, **kwargs):
    job = v1.new_job(*args, **kwargs)
    job.update(schema=SCHEMA, ranknum_policy=RANKNUM_POLICY, worker_identity=worker_identity())
    return validate_job(job)
