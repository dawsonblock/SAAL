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


def test_v3_builder_rejects_stale_source_binding_after_v4_worker_changes(tmp_path):
    _, secret, pinfile = _key_files(tmp_path)
    out = tmp_path/'pilot-v3'
    with pytest.raises(ValueError, match='controller / analysis source changed'):
        admit_pilot_v3.prepare(out, secret, pinfile)
    assert not out.exists()


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
