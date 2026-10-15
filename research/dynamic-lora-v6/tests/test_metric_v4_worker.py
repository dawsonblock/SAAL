import copy
import subprocess
import sys

import pytest

from saal_rsi.adaptive_campaign_v4 import stage_worker
from saal_rsi.adaptive_contract_v4 import POLICY, POLICY_DIGEST, new_job, validate_job, worker_identity
from saal_rsi.adaptive_contract_v3 import POLICY as V3_POLICY, POLICY_DIGEST as V3_POLICY_DIGEST
from saal_rsi.adaptive_contract_v3 import validate_job as validate_v3
from saal_rsi.adaptive_dispatch import validate_job as dispatch_v3
from saal_rsi.model_contract import sign
from saal_rsi.spec import canonical


def v4_job(admitted_job):
    budget = copy.deepcopy(admitted_job['budget'])
    return new_job(admitted_job['model_spec'], admitted_job['dataset'], admitted_job['adapter'],
        seed=admitted_job['seed'], budget=budget, steps=admitted_job['steps'],
        batch_size=admitted_job['batch_size'], accumulation=admitted_job['accumulation'],
        max_seconds=admitted_job['max_seconds'], campaign_stage='ENGINEERING',
        learning_rate=admitted_job['learning_rate'], head_learning_rate=admitted_job['head_learning_rate'],
        checkpoint_interval=admitted_job['checkpoint_interval'])


def test_v4_binds_v2_metric_and_has_distinct_worker_identity(admitted_job):
    job = v4_job(admitted_job)
    assert job['schema'] == 'saal-adaptive-job-v4'
    assert job['numerical_metric_policy'] == POLICY
    assert job['numerical_metric_policy_digest'] == POLICY_DIGEST
    assert job['worker_identity'] == worker_identity()
    assert dispatch_v3(admitted_job) is admitted_job
    with pytest.raises(ValueError):
        validate_v3(job)
    with pytest.raises(ValueError):
        dispatch_v3(job)


@pytest.mark.parametrize('fault', ['policy', 'digest', 'worker', 'authority', 'metric downgrade'])
def test_v4_rejects_contract_tampering(admitted_job, fault):
    job = v4_job(admitted_job)
    if fault == 'policy': job['numerical_metric_policy']['policy_id'] = 'other'
    elif fault == 'digest': job['numerical_metric_policy_digest'] = 'sha256:'+'0'*64
    elif fault == 'worker': job['worker_identity'] = 'sha256:'+'0'*64
    elif fault == 'authority': job['external_deployment_allowed'] = True
    else:
        job['numerical_metric_policy'] = V3_POLICY
        job['numerical_metric_policy_digest'] = V3_POLICY_DIGEST
    with pytest.raises(ValueError):
        validate_job(job)


def test_v4_staging_contains_allowlisted_worker_and_no_private_key(tmp_path, admitted_job):
    job = v4_job(admitted_job)
    target = tmp_path/'staged'
    stage_worker(target)
    key = __import__('cryptography.hazmat.primitives.asymmetric.ed25519', fromlist=['Ed25519PrivateKey']).Ed25519PrivateKey.generate()
    job_file = tmp_path/'job.json'
    job_file.write_bytes(canonical(sign(job, key)))
    result = subprocess.run([sys.executable, '-I', str(target/'worker.py'), '--job', str(job_file), '--help'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (target/'src/saal_rsi/adaptive_worker_v4.py').is_file()
    assert not list(target.rglob('*.ed25519'))
    assert not list(target.rglob('*private*key*'))
