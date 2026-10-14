import copy
import subprocess
import sys

import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.adaptive_backend_v2 import freeze_ranknum
from saal_rsi.adaptive_campaign import stage_worker
from saal_rsi.adaptive_contract import validate_job as validate_v1
from saal_rsi.adaptive_contract_v2 import RANKNUM_POLICY, SCHEMA, worker_identity, validate_job
from saal_rsi.adaptive_dispatch import validate_job as dispatch
from saal_rsi.model_contract import sign
from saal_rsi.spec import canonical


def test_parameterdict_device_cast_registration_preserves_value_but_can_enable_gradient():
    # CPU dtype transfer reproduces the same Parameter -> Tensor -> ParameterDict
    # registration that happens during PEFT's CUDA transfer.
    original = torch.nn.Parameter(torch.tensor([12.], dtype=torch.float64), requires_grad=False)
    parameters = torch.nn.ParameterDict()
    parameters['default'] = original.float()
    assert parameters['default'].requires_grad is True
    before = parameters['default'].detach().clone()
    assert freeze_ranknum(dict(parameters.items())) == 1
    assert parameters['default'].requires_grad is False
    assert torch.equal(before, parameters['default']) and parameters['default'].grad is None
    assert freeze_ranknum(dict(parameters.items())) == 0


@pytest.mark.parametrize('bad', [torch.tensor([11.]), torch.tensor([12., 12.]),
    torch.tensor([12.], dtype=torch.float64), torch.tensor([float('nan')])])
def test_scaling_repair_rejects_changed_value_shape_dtype_or_nonfinite_before_mutation(bad):
    good = torch.nn.Parameter(torch.tensor([12.]), requires_grad=True)
    other = torch.nn.Parameter(bad, requires_grad=True)
    with pytest.raises(ValueError):
        freeze_ranknum({'good': good, 'bad': other})
    assert good.requires_grad and good.item() == 12


@pytest.fixture
def v2_job(admitted_job):
    job = copy.deepcopy(admitted_job)
    job.update(schema=SCHEMA, ranknum_policy=RANKNUM_POLICY, worker_identity=worker_identity())
    return job


def test_distinct_signed_schema_never_reinterprets_original_jobs(admitted_job, v2_job):
    assert dispatch(admitted_job) is admitted_job
    assert dispatch(v2_job) is v2_job
    with pytest.raises(ValueError): validate_v1(v2_job)
    with pytest.raises(ValueError): validate_job(admitted_job)


@pytest.mark.parametrize('fault', ['policy', 'worker', 'unknown', 'adapter', 'fixed_method'])
def test_v2_admission_is_closed_and_scaling_policy_is_signed(v2_job, fault):
    if fault == 'policy': v2_job['ranknum_policy'] = 'train-scaling'
    elif fault == 'worker': v2_job['worker_identity'] = 'sha256:'+'0'*64
    elif fault == 'unknown': v2_job['allow_external_activation'] = True
    elif fault == 'adapter': v2_job['adapter'] = []
    else: v2_job['adapter']['method'] = 'fixed-r8'
    with pytest.raises(ValueError): dispatch(v2_job)


def test_v2_staging_includes_only_allowlisted_workers_and_dispatches_version(tmp_path, v2_job):
    stage_worker(tmp_path/'worker', include_v2=True)
    key = Ed25519PrivateKey.generate()
    job_file = tmp_path/'job.json'; job_file.write_bytes(canonical(sign(v2_job, key)))
    result = subprocess.run([sys.executable, '-I', str(tmp_path/'worker/worker.py'),
        '--job', str(job_file), '--help'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (tmp_path/'worker/src/saal_rsi/adaptive_worker_v2.py').is_file()
    assert not list((tmp_path/'worker').rglob('*.ed25519'))
