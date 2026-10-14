import importlib.util
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from saal_rsi import adaptive_artifacts
from saal_rsi.model_contract import sign
from saal_rsi.spec import canonical, sha

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('portable_adaptive_gate', ROOT/'research/dynamic-lora-v6/prepare.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def extracted_pilot(tmp_path, monkeypatch):
    key = Ed25519PrivateKey.generate()
    pin = key.public_key().public_bytes_raw().hex()
    base = tmp_path/'extracted-pilot'; base.mkdir()
    (base/'colab').mkdir(); (base/'artifacts').mkdir()
    (base/'controller.pub').write_text(pin)
    predictions = [{'decision': 'FAST', 'probabilities': [.1, .7, .1, .1]}]
    rows, descriptors = [], []
    for index, method in enumerate(gate.METHODS):
        job = {'steps': 1000, 'seed': gate.ENGINEERING_SEED, 'adapter': {'method': method}}
        digest = sha(job); name = 'method_'+str(index); run = f'run-{index:02d}-'+digest[7:19]
        (base/(name+'-job.json')).write_bytes(canonical(sign(job, key)))
        artifact = base/'artifacts'/run; artifact.mkdir()
        (artifact/'artifact_inventory.json').write_text('{}')
        (artifact/'sampler_manifest.json').write_bytes(canonical({'record_ids': ['matched-stream']}))
        receipt = {'inventory_sha256': adaptive_artifacts.file_hash(artifact/'artifact_inventory.json'),
            'job_digest': digest, 'verified': True,
            'predictions': {role: predictions for role in ('initial', 'best', 'terminal')}}
        (base/'colab'/(run+'-controller-replay.json')).write_bytes(canonical(receipt))
        descriptors.append({'name': name, 'job_digest': digest})
        rows.append({'state': 'COMPLETED', 'digest': digest, 'outcome': {'run': run,
            'job_digest': digest, 'artifacts': '/unavailable/original-checkout/artifacts/'+run,
            'controller_replay': '/unavailable/original-checkout/colab/'+run+'-controller-replay.json'}})
    protocol = {'runs': descriptors}
    (base/'signed_protocol.json').write_bytes(canonical(sign(protocol, key)))
    export = {'campaign_state': 'COMPLETED', 'attempts': rows}
    (base/'colab/registry-export.json').write_bytes(canonical(export))
    (base/'colab/monitor-state.json').write_bytes(canonical({'phase': 'COMPLETED', 'cleanup_verified': True}))
    seen = []
    def validate_local(path, job, public, require_replay):
        assert public == pin and require_replay
        assert Path(path).parent == base/'artifacts'
        seen.append(path)
        return {'unadapted_model_hash': 'same', 'snapshots': {role: {'predictions': predictions}
            for role in ('initial', 'best', 'terminal')}}
    monkeypatch.setattr(adaptive_artifacts, 'validate_bundle', validate_local)
    return base, export, seen


def test_clean_extraction_gate_reads_its_own_artifacts_not_original_absolute_paths(extracted_pilot):
    base, export, seen = extracted_pilot
    assert gate.pilot_accepted(base) == sha(export)
    assert len(seen) == 3


@pytest.mark.parametrize('fault', ['substituted_job', 'path_escape', 'wrong_run'])
def test_portable_gate_rejects_outcome_identity_substitution(extracted_pilot, fault):
    base, export, _ = extracted_pilot
    outcome = export['attempts'][0]['outcome']
    if fault == 'substituted_job':
        outcome['job_digest'] = 'sha256:'+'0'*64
    elif fault == 'path_escape':
        outcome['run'] = '../outside'
    else:
        outcome['run'] = 'run-00-'+'0'*12
    (base/'colab/registry-export.json').write_bytes(canonical(export))
    with pytest.raises(ValueError):
        gate.pilot_accepted(base)
