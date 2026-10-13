import copy
import json
from pathlib import Path
import pytest
from saal_rsi.epoch_artifacts_v3 import validate_evidence, validate_bundle
from saal_rsi.model_contract import verify
from saal_rsi import epoch_backend_v6 as backend

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT/'research/behavior-closure-v4/experiment/artifacts/run-02-1ead0e5752ba'
PIN = (ROOT/'research/behavior-closure-v4/experiment/controller.pub').read_text().strip()


def fixture():
    job = verify(json.loads((BUNDLE/'signed_training_job.json').read_text()), PIN)
    result = json.loads((BUNDLE/'result.json').read_text())
    inventory = json.loads((BUNDLE/'artifact_inventory.json').read_text())
    hashes = {n: inventory['files'][n+'.safetensors']['sha256'] for n in ('initial', 'best', 'terminal')}
    return job, result, hashes


def test_real_native_tensors_pass_stronger_validation():
    job, result, _ = fixture()
    assert validate_bundle(BUNDLE, job, PIN) == result


@pytest.mark.parametrize('fault', ['attention', 'missing_checkpoint'])
def test_previous_validator_accepts_reproduction_of_defect_new_validator_rejects(tmp_path, fault):
    import os
    from saal_rsi.epoch_artifacts_v2 import validate_bundle as previous, seal
    from saal_rsi.spec import canonical
    job, result, _ = fixture()
    bundle = tmp_path/'bundle'; bundle.mkdir()
    for source in BUNDLE.iterdir():
        if source.suffix == '.safetensors': os.link(source, bundle/source.name)
        else: (bundle/source.name).write_bytes(source.read_bytes())
    if fault == 'attention': result['runtime']['attention'] = 'sdpa'
    else:
        result['selection_history'] = [r for r in result['selection_history'] if r['step'] != 100]
        log = [json.loads(line) for line in (bundle/'training_metrics.jsonl').read_text().splitlines()]
        (bundle/'training_metrics.jsonl').write_bytes(b'\n'.join(canonical(r) for r in log
            if not (r.get('monitor') == 'checkpoint_metrics' and r['step'] == 100))+b'\n')
    (bundle/'result.json').write_bytes(canonical(result)); seal(bundle)
    assert previous(bundle, job, PIN) == result
    with pytest.raises(ValueError): validate_bundle(bundle, job, PIN)


@pytest.mark.parametrize('fault', ['missing_step', 'extra_step', 'boolean_step', 'terminal_metrics',
    'attention', 'platform', 'microbatches', 'parameter_count', 'negative_time', 'infinite_time',
    'bool_memory', 'memory_bounds', 'gain', 'selected_tensor'])
def test_evidence_substitution_is_rejected(fault):
    job, original, hashes = fixture()
    value = copy.deepcopy(original)
    if fault == 'missing_step': del value['selection_history'][1]
    elif fault == 'extra_step': value['selection_history'].insert(1, dict(value['selection_history'][0], step=1))
    elif fault == 'boolean_step': value['selection_history'][0]['step'] = False
    elif fault == 'terminal_metrics': value['selection_history'][-1]['metrics']['development']['accuracy'] = 0.
    elif fault == 'attention': value['runtime']['attention'] = 'sdpa'
    elif fault == 'platform': value['runtime']['python'] = '3.12.0'
    elif fault == 'microbatches': value['resource_accounting']['microbatches'] += 1
    elif fault == 'parameter_count': value['resource_accounting']['trainable_parameters'] += 1
    elif fault == 'negative_time': value['total_worker_seconds'] = -1.
    elif fault == 'infinite_time': value['total_worker_seconds'] = float('inf')
    elif fault == 'bool_memory': value['runtime']['total_gpu_bytes'] = True
    elif fault == 'memory_bounds': value['runtime']['free_gpu_bytes_preflight'] = value['runtime']['total_gpu_bytes']+1
    elif fault == 'gain': value['adaptation_gain_terminal']['mean_gain'] = 1.
    else:
        value['snapshots']['best'] = copy.deepcopy(value['snapshots']['terminal'])
        value['adaptation_gain_best'] = copy.deepcopy(value['adaptation_gain_terminal'])
        hashes['best'] = '0'*64
    with pytest.raises(ValueError): validate_evidence(value, job, hashes)


def test_disk_admission_accounts_for_archive_and_uncompressed_tensors(monkeypatch, tmp_path):
    job, _, _ = fixture()
    from collections import namedtuple
    usage = namedtuple('usage', 'total used free')
    import shutil
    monkeypatch.setattr(shutil, 'disk_usage', lambda p: usage(100, 99, 1))
    with pytest.raises(OSError) as error: backend.require_storage(tmp_path, job, 20_000_000)
    assert error.value.errno == 28


def test_coordinator_keeps_original_failure_if_json_export_fails(monkeypatch, tmp_path):
    job, _, _ = fixture()
    envelope = json.loads((BUNDLE/'signed_training_job.json').read_text())
    path = tmp_path/'job.json'; path.write_text(json.dumps(envelope))
    ledger_path = tmp_path/'budget.sqlite3'
    from saal_rsi.epoch_registry import EpochExperimentRegistry
    ledger = EpochExperimentRegistry(ledger_path, PIN, max_jobs=1, max_seconds=3800)
    ledger.reserve(envelope); ledger.close()
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'workspace': str(tmp_path), 'session': 'no-remote', 'pin': PIN,
        'jobs': [str(path)], 'artifacts': str(tmp_path/'artifacts'), 'ledger': str(ledger_path)}))
    monkeypatch.setattr(backend, 'require_storage', lambda *args: {})
    def write(path, value):
        if path.name == 'outcomes.json': raise OSError(28, 'full disk')
    monkeypatch.setattr(backend, 'write_json', write)
    # Missing lock fails before a cloud call; export subsequently also fails.
    with pytest.raises(RuntimeError, match='FileNotFoundError.*JSON export failed'):
        backend.coordinator(config)
    ledger = EpochExperimentRegistry(ledger_path, PIN)
    records = ledger.records(); ledger.close()
    row = records['attempts'][0]
    assert row['state'] == 'INTERRUPTED' and row['outcome']['error']['type'] == 'FileNotFoundError'
