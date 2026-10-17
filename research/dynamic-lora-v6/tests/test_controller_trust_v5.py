import hashlib
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# Reuse the repository's fully valid v4 admission fixture. The test exercises
# the new external trust boundary without starting the Colab runner.
from test_metric_v4_admission import _signed_pair

from saal_rsi.adaptive_colab_trusted_v5 import (
    TRUST_STORE_ENV,
    TRUST_STORE_SCHEMA,
    execute,
    stage_trusted_admission,
)
from saal_rsi.spec import canonical


def write_trust_store(path, public_hex):
    key_id = hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()
    path.write_bytes(canonical({'schema': TRUST_STORE_SCHEMA, 'keys': {
        key_id: {'public_key': public_hex, 'roles': ['campaign-admission']}
    }}))
    os.chmod(path, 0o600)
    return key_id


def make_admission(tmp_path, admitted_job):
    _, pin, _, job_envelope, protocol_envelope, protocol = _signed_pair(admitted_job, tmp_path)
    admission = tmp_path / 'submitted-admission'
    (admission / 'jobs').mkdir(parents=True)
    (admission / 'controller.pub').write_text('0' * 64)
    (admission / 'signed_protocol.json').write_bytes(canonical(protocol_envelope))
    name = protocol['runs'][0]['name'] + '-job.json'
    (admission / 'jobs' / name).write_bytes(canonical(job_envelope))
    return admission, pin, name


def test_trusted_admission_stages_exact_signed_bytes_and_replaces_submitted_pin(tmp_path, admitted_job):
    admission, pin, job_name = make_admission(tmp_path, admitted_job)
    trust_path = tmp_path / 'controller-trust.json'
    write_trust_store(trust_path, pin)
    original_protocol = (admission / 'signed_protocol.json').read_bytes()
    original_job = (admission / 'jobs' / job_name).read_bytes()
    with stage_trusted_admission(admission, trust_path) as staged:
        assert (staged / 'controller.pub').read_text() == pin
        assert (staged / 'signed_protocol.json').read_bytes() == original_protocol
        assert (staged / 'jobs' / job_name).read_bytes() == original_job
        assert (staged / 'controller.pub').read_text() != (admission / 'controller.pub').read_text()
    assert (admission / 'signed_protocol.json').read_bytes() == original_protocol
    assert (admission / 'jobs' / job_name).read_bytes() == original_job


def test_untrusted_or_substituted_signing_key_rejects_before_staging(tmp_path, admitted_job):
    admission, _pin, *_ = make_admission(tmp_path, admitted_job)
    other_public = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    trust_path = tmp_path / 'controller-trust.json'
    write_trust_store(trust_path, other_public)
    with pytest.raises(PermissionError, match='absent from controller trust store'), \
            stage_trusted_admission(admission, trust_path):
        pass


def test_trust_store_cannot_be_supplied_from_admission_or_symlink(tmp_path, admitted_job):
    admission, pin, *_ = make_admission(tmp_path, admitted_job)
    inside = admission / 'trust.json'
    write_trust_store(inside, pin)
    with pytest.raises(ValueError, match='outside'), stage_trusted_admission(admission, inside):
        pass
    symlink = tmp_path / 'trust-link.json'
    symlink.symlink_to(inside)
    with pytest.raises(ValueError, match='symlink'), stage_trusted_admission(admission, symlink):
        pass


def test_controller_execute_fails_closed_without_external_trust_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv(TRUST_STORE_ENV, raising=False)
    with pytest.raises(PermissionError, match=TRUST_STORE_ENV):
        execute(tmp_path / 'workspace', tmp_path / 'admission',
                tmp_path / 'artifacts', tmp_path / 'cache')
    assert not (tmp_path / 'workspace').exists()


def test_trusted_stage_rejects_extra_job_and_untrusted_envelope(tmp_path, admitted_job):
    admission, pin, *_ = make_admission(tmp_path, admitted_job)
    trust_path = tmp_path / 'controller-trust.json'
    write_trust_store(trust_path, pin)
    (admission / 'jobs' / 'unsolicited.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory'), stage_trusted_admission(admission, trust_path):
        pass
