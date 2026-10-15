"""Controller seams for new admissions; no cloud or historical ledger writes."""
import copy
import subprocess
import sys
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.adaptive_contract import adapter_spec, new_job as legacy_job
from saal_rsi.adaptive_contract_v2 import validate_job as validate_v2
from saal_rsi.adaptive_contract_v3 import SOURCES, new_job, worker_identity
from saal_rsi.adaptive_dataset import training_projection
from saal_rsi.adaptive_dispatch import validate_job
from saal_rsi.adaptive_registry import AdaptiveRegistry
from saal_rsi.learning_dataset import generative_pilot, wrap
from saal_rsi.model_contract import sign
from saal_rsi.numerical_metrics_v1 import POLICY, POLICY_DIGEST
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.spec import canonical, sha


@pytest.fixture(scope='module')
def job_arguments():
    population = wrap([r for r in generative_pilot()['records'] if r['task_type'] == 'decision'])
    spec = model_spec('answerdotai/ModernBERT-base', method='none', backend='local',
                      precision='float32', budget='numerical_contract_test')
    budget = {'campaign_id': 'numeric_v3_test', 'run_id': 'test_run', 'max_runs': 3,
              'total_lifecycle_seconds': 15000, 'reserved_seconds': 5000,
              'automatic_training_retries': 0, 'financial_cap': False}
    return (spec, training_projection(population)), dict(seed=60606, budget=budget,
        steps=4, max_seconds=3600, campaign_stage='ENGINEERING', checkpoint_interval=2)


@pytest.fixture
def v3_job(job_arguments):
    args, kwargs = job_arguments
    return new_job(*copy.deepcopy(args),
        adapter_spec('adalora-12-to-8', total_steps=4, tinit=1, tfinal=1, delta=1),
        **copy.deepcopy(kwargs))


def protocol_for(job, key):
    budget = {k: job['budget'][k] for k in ('campaign_id', 'max_runs',
        'total_lifecycle_seconds', 'automatic_training_retries', 'financial_cap')}
    budget['max_provisioning_attempts'] = 1
    return sign({'schema': 'saal-adaptive-campaign-v3', 'stage': job['campaign_stage'],
        'runs': [{'name': job['budget']['run_id'], 'method': job['adapter']['method'],
                  'seed': job['seed'], 'job_digest': sha(job)}],
        'budget': budget,
        'job_digests': [sha(job)], 'worker_identities': {'v3': worker_identity()},
        'colab_cli_version': '0.7.4',
        'numerical_metric_policy': copy.deepcopy(POLICY),
        'numerical_metric_policy_digest': POLICY_DIGEST,
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}, key)


@pytest.mark.parametrize('method', ['fixed-r8', 'fixed-r12', 'adalora-12-to-8'])
def test_controller_dispatches_all_three_v3_methods_without_legacy_reinterpretation(job_arguments, method):
    args, kwargs = job_arguments
    adapter = (adapter_spec(method, total_steps=4, tinit=1, tfinal=1, delta=1)
               if method == 'adalora-12-to-8' else adapter_spec(method))
    job = new_job(*copy.deepcopy(args), adapter, **copy.deepcopy(kwargs))
    assert validate_job(job) is job
    from saal_rsi.adaptive_contract import validate_job as validate_v1
    with pytest.raises(ValueError): validate_v1(job)
    with pytest.raises(ValueError): validate_v2(job)


def test_v3_staging_is_explicit_and_contains_only_registered_sources(tmp_path, v3_job):
    from saal_rsi.adaptive_campaign import stage_worker
    destination = tmp_path/'worker'
    stage_worker(destination, version='v3')
    names = {p.name for p in (destination/'src/saal_rsi').iterdir()}
    assert names == set(SOURCES) | {'__init__.py'}
    key = Ed25519PrivateKey.generate()
    admitted = tmp_path/'job.json'
    admitted.write_bytes(canonical(sign(v3_job, key)))
    result = subprocess.run([sys.executable, '-B', '-I', str(destination/'worker.py'),
        '--job', str(admitted), '--help'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not list(destination.rglob('*.ed25519'))
    assert set(p.name for p in destination.rglob('*.json')) == {'registry.json', 'tensor_inventories.json'}


@pytest.mark.parametrize('version', ['v4', '', True])
def test_unknown_staging_version_fails_before_filesystem_mutation(tmp_path, version):
    from saal_rsi.adaptive_campaign import stage_worker
    destination = tmp_path/'worker'
    with pytest.raises(ValueError): stage_worker(destination, version=version)
    assert not destination.exists()


@pytest.mark.parametrize('fault', ['policy', 'digest', 'worker', 'missing_policy', 'missing_cli', 'invalid_cli'])
def test_v3_protocol_rejects_changed_policy_or_worker_before_budget_creation(tmp_path, v3_job, fault):
    key = Ed25519PrivateKey.generate()
    envelope = protocol_for(v3_job, key)
    payload = envelope['payload']
    if fault == 'policy': payload['numerical_metric_policy'] = {'unregistered': True}
    elif fault == 'digest': payload['numerical_metric_policy_digest'] = 'sha256:'+'0'*64
    elif fault == 'worker': payload['worker_identities'] = {'v3': 'sha256:'+'0'*64}
    elif fault == 'missing_policy': del payload['numerical_metric_policy']
    elif fault == 'missing_cli': del payload['colab_cli_version']
    else: payload['colab_cli_version'] = 'latest'
    path = tmp_path/'budget.sqlite3'
    with pytest.raises(ValueError):
        AdaptiveRegistry(path, key.public_key().public_bytes_raw().hex(), sign(payload, key), create=True)
    assert not path.exists()


@pytest.mark.parametrize(('stdout', 'expected', 'valid'), [
    ('Version: 0.7.4\n', '0.7.4', True),
    ('Version: 0.7.5\n', '0.7.4', False),
    ('unknown\n', '0.7.4', False),
])
def test_colab_cli_version_must_match_signed_preprovision_value(monkeypatch, stdout, expected, valid):
    from saal_rsi import adaptive_campaign
    monkeypatch.setattr(adaptive_campaign.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=stdout))
    if valid:
        assert adaptive_campaign.verify_colab_cli_version(expected) == expected
    else:
        with pytest.raises(ValueError, match='Colab CLI version'):
            adaptive_campaign.verify_colab_cli_version(expected)


def test_fresh_v3_budget_cannot_reserve_legacy_jobs_or_reset_interruption(tmp_path, v3_job, job_arguments):
    key = Ed25519PrivateKey.generate()
    pin = key.public_key().public_bytes_raw().hex()
    protocol = protocol_for(v3_job, key)
    args, kwargs = job_arguments
    old = legacy_job(*copy.deepcopy(args),
        adapter_spec('adalora-12-to-8', total_steps=4, tinit=1, tfinal=1, delta=1),
        **copy.deepcopy(kwargs))
    path = tmp_path/'budget.sqlite3'
    registry = AdaptiveRegistry(path, pin, protocol, create=True)
    try:
        with pytest.raises(ValueError, match='separate v3 campaign admission'):
            registry.reserve(sign(old, key))
        registry.reserve(sign(v3_job, key))
        registry.begin_provisioning()
        registry.start(sha(v3_job))
        registry.finish(sha(v3_job), {'error': 'test interruption'}, False)
        registry.end(False)
        with pytest.raises(ValueError): registry.begin_provisioning()
        with pytest.raises(ValueError): registry.end(True)
        value = registry.records()
        assert value['campaign_state'] == 'FAILED'
        assert value['provisioning_attempts'] == 1
        assert value['attempts'][0]['state'] == 'INTERRUPTED'
    finally:
        registry.close()
    with pytest.raises(FileExistsError): AdaptiveRegistry(path, pin, protocol, create=True)


@pytest.mark.parametrize('schema', ['saal-adaptive-campaign-v1', 'saal-adaptive-campaign-v2'])
def test_legacy_campaign_cannot_reserve_v3_job_without_new_policy_admission(tmp_path, v3_job, schema):
    """A newly signed job alone cannot bypass the campaign's policy binding."""
    key = Ed25519PrivateKey.generate()
    pin = key.public_key().public_bytes_raw().hex()
    payload = protocol_for(v3_job, key)['payload']
    payload['schema'] = schema
    for field in ('worker_identities', 'numerical_metric_policy', 'numerical_metric_policy_digest'):
        payload.pop(field)
    registry = AdaptiveRegistry(tmp_path/'legacy-budget.sqlite3', pin, sign(payload, key), create=True)
    try:
        before = registry.records()
        with pytest.raises(ValueError, match='separate v3 campaign admission'):
            registry.reserve(sign(v3_job, key))
        assert registry.records() == before
        assert registry.records()['attempts'] == []
        assert registry.records()['provisioning_attempts'] == 0
    finally:
        registry.close()


@pytest.mark.parametrize('probability_drift, agrees', [(0., True), (.0002, False)])
def test_controller_replay_receipt_binds_v3_policy_without_widening_prediction_tolerance(
        tmp_path, monkeypatch, v3_job, probability_drift, agrees):
    """Exercise controller routing with a stub backend, not model replay evidence."""
    import torch
    from saal_rsi import adaptive_backend, adaptive_campaign, adaptive_replay_v3
    from saal_rsi import learning_backend, pretrained_registry
    roles = ('initial', 'best', 'terminal')
    rows = [r for r in v3_job['dataset']['records'] if r['split'] == 'development']
    archived = [{'decision': 'CHEAP', 'probabilities': [1., 0., 0., 0.]} for _ in rows]
    native = {'unadapted_model_hash': 'same-initial-identity',
              'frozen_integrity': {'initial': 'same-backbone'},
              'snapshots': {role: {'predictions': archived} for role in roles}}
    restored = []

    class StubReplayBackend:
        def __init__(self, job, snapshot, initial_checkpoint, initial_state):
            assert job is v3_job
            self.unadapted_model_hash = 'same-initial-identity'
            self.initial_frozen_hash = 'same-backbone'
            self.model = SimpleNamespace(parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]))

        def load_pretrained(self): return self

        def restore_checkpoint(self, checkpoint, state): restored.append(checkpoint.stem)

        def evaluate(self, records):
            return [{'decision': 'CHEAP', 'probabilities':
                     [1.-probability_drift, probability_drift, 0., 0.]} for _ in records]

    monkeypatch.setattr(adaptive_replay_v3, 'PortableReplayBackendV3', StubReplayBackend)
    monkeypatch.setattr(adaptive_campaign, 'validate_bundle', lambda *a, **kw: native)
    monkeypatch.setattr(adaptive_backend, 'verify_peft_implementation', lambda: {})
    monkeypatch.setattr(learning_backend, 'verify_dependencies', lambda: {})
    monkeypatch.setattr(pretrained_registry, 'validate_snapshot', lambda *a: {})
    for role in roles:
        (tmp_path/(role+'_rank_pattern.json')).write_text('{}')
    (tmp_path/'artifact_inventory.json').write_text('{}')
    receipt = adaptive_campaign.fresh_local_replay(tmp_path, v3_job, 'test-pin', tmp_path/'cache')
    assert receipt['verified'] is agrees
    assert restored == list(roles)
    assert receipt['schema'] == 'saal-adaptive-controller-replay-v3'
    assert receipt['numerical_metric_policy'] == POLICY
    assert receipt['numerical_metric_policy_digest'] == POLICY_DIGEST
    assert receipt['worker_identity'] == worker_identity()
    assert receipt['tolerance'] == 1e-4
    assert receipt['independent_administration'] is False
    assert receipt['qualification_status'] == 'NOT_EVALUATED'
