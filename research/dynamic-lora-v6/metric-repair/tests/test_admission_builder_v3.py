"""The fresh pilot admission builder signs/reserves locally and never starts cloud work."""
import json
from pathlib import Path
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.model_contract import verify

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import admit_pilot_v3


def _key_files(tmp_path, key=None):
    key = key or Ed25519PrivateKey.generate()
    secret = tmp_path/'controller.ed25519'
    secret.write_bytes(key.private_bytes_raw())
    secret.chmod(0o600)
    pin = tmp_path/'controller.pub'
    pin.write_text(key.public_key().public_bytes_raw().hex()+'\n')
    return key, secret, pin


def test_builder_makes_new_signed_pilot_budget_without_provisioning(tmp_path):
    _, secret, pinfile = _key_files(tmp_path)
    out = admit_pilot_v3.prepare(tmp_path/'pilot-v3', secret, pinfile)
    pin = pinfile.read_text().strip()
    protocol_envelope = json.loads((out/'signed_protocol.json').read_text())
    protocol = verify(protocol_envelope, pin)
    assert protocol['schema'] == 'saal-adaptive-campaign-v3'
    assert protocol['stage'] == 'PILOT'
    assert protocol['colab_cli_version'] == '0.7.4'
    assert protocol['qualification_status'] == 'NOT_EVALUATED'
    assert protocol['external_deployment_allowed'] is False
    assert len(protocol['runs']) == 3
    assert {row['method'] for row in protocol['runs']} == {'fixed-r8', 'fixed-r12', 'adalora-12-to-8'}
    assert len(list((out/'jobs').glob('*-job.json'))) == 3
    reservation = json.loads((out/'preprovision-registry.json').read_text())
    assert reservation['campaign_state'] == 'PREPARED'
    assert reservation['provisioning_attempts'] == 0
    assert len(reservation['attempts']) == 3
    assert not list(out.rglob('*.ed25519'))
    assert not (out/'controller.key').exists()
    authorization = json.loads((out/'authorization-record.json').read_text())
    assert authorization['approval_source'] == 'direct user instruction in this conversation'
    assert (out/'budget.sqlite3').exists()


def test_builder_rejects_wrong_controller_key_before_creating_output(tmp_path):
    _, secret, _ = _key_files(tmp_path)
    pin = tmp_path/'different.pub'
    pin.write_text(Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex())
    out = tmp_path/'pilot-v3'
    with pytest.raises(ValueError, match='does not match'):
        admit_pilot_v3.prepare(out, secret, pin)
    assert not out.exists()


def test_builder_preserves_existing_output_directory(tmp_path):
    _, secret, pin = _key_files(tmp_path)
    out = tmp_path/'pilot-v3'
    out.mkdir()
    marker = out/'keep.txt'; marker.write_text('existing evidence')
    with pytest.raises(FileExistsError):
        admit_pilot_v3.prepare(out, secret, pin)
    assert marker.read_text() == 'existing evidence'
