"""Controller-owned trust gate for v5 Colab campaign admissions.

The legacy runner reads ``controller.pub`` from the submitted admission
directory. This wrapper resolves that identity against an operator-provisioned
trust store first, verifies every signed object, then stages an immutable
admission copy whose public key comes from the trust store. It does not make a
self-signed development key independently administered or qualified.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .adaptive_evaluator_v4 import validate_jobs, validate_protocol
from .model_contract import verify
from .spec import parse_unique_json
from .validation import key_bytes

TRUST_STORE_ENV = 'SAAL_CONTROLLER_TRUST_STORE'
TRUST_STORE_SCHEMA = 'saal-controller-trust-store-v1'
MAX_TRUST_STORE_BYTES = 1_000_000
MAX_ADMISSION_OBJECT_BYTES = 50_000_000
MAX_ADMISSION_TOTAL_BYTES = 250_000_000


def _read_regular_file(path, *, maximum):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('symlink input rejected')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= maximum:
            raise ValueError('regular bounded file required')
        chunks, remaining = [], maximum + 1
        while remaining:
            block = os.read(fd, min(1_048_576, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b''.join(chunks)
        if len(raw) > maximum:
            raise ValueError('input grew beyond configured size bound')
        return raw, info
    finally:
        os.close(fd)


def load_controller_trust_store(path, *, admission_dir):
    """Read a controller-owned, non-group/world-writable public-key registry."""
    trust_path = Path(path)
    admission = Path(admission_dir)
    if not trust_path.is_absolute():
        raise ValueError('controller trust-store path must be absolute')
    if trust_path.is_symlink() or admission.is_symlink():
        raise ValueError('trust store and admission directory must not be symlinks')
    trust_path = trust_path.resolve(strict=True)
    admission = admission.resolve(strict=True)
    if trust_path.is_relative_to(admission):
        raise ValueError('controller trust store must be outside the submitted admission')
    raw, info = _read_regular_file(trust_path, maximum=MAX_TRUST_STORE_BYTES)
    if info.st_uid not in (os.geteuid(), 0) or stat.S_IMODE(info.st_mode) & 0o022:
        raise PermissionError('trust store owner or write permissions are not controller-safe')
    store = parse_unique_json(raw)
    if type(store) is not dict or set(store) != {'schema', 'keys'} or store['schema'] != TRUST_STORE_SCHEMA:
        raise ValueError('controller trust-store schema')
    keys = store['keys']
    if type(keys) is not dict or not keys:
        raise ValueError('nonempty controller public-key registry required')
    normalized = {}
    for key_id, entry in keys.items():
        if (type(key_id) is not str or not re.fullmatch(r'[0-9a-f]{64}', key_id)
                or type(entry) is not dict or set(entry) != {'public_key', 'roles'}
                or type(entry['public_key']) is not str
                or entry['roles'] != ['campaign-admission']):
            raise ValueError('malformed controller public-key entry')
        public = key_bytes(entry['public_key'])
        if hashlib.sha256(public).hexdigest() != key_id:
            raise ValueError('controller trust-store key identity mismatch')
        normalized[key_id] = entry['public_key']
    return normalized


def _trusted_envelope(envelope, trusted_keys):
    if type(envelope) is not dict or type(envelope.get('key_sha256')) is not str:
        raise ValueError('signed envelope identity is missing')
    key_id = envelope['key_sha256']
    pin = trusted_keys.get(key_id)
    if pin is None:
        raise PermissionError('signing key is absent from controller trust store')
    verify(envelope, pin)
    return pin


@contextmanager
def stage_trusted_admission(admission_dir, trust_store_path):
    """Verify and copy the exact protocol/jobs under a controller-owned pin."""
    admission = Path(admission_dir)
    if admission.is_symlink():
        raise ValueError('admission directory symlink rejected')
    admission = admission.resolve(strict=True)
    jobs_dir = admission / 'jobs'
    if jobs_dir.is_symlink() or not jobs_dir.is_dir():
        raise ValueError('regular signed jobs directory required')
    trusted_keys = load_controller_trust_store(trust_store_path, admission_dir=admission)
    protocol_bytes, _ = _read_regular_file(admission / 'signed_protocol.json',
                                           maximum=MAX_ADMISSION_OBJECT_BYTES)
    total_bytes = len(protocol_bytes)
    protocol_envelope = parse_unique_json(protocol_bytes)
    pin = _trusted_envelope(protocol_envelope, trusted_keys)
    protocol = validate_protocol(verify(protocol_envelope, pin))
    run_names = [run['name'] + '-job.json' for run in protocol['runs']]
    if len(set(run_names)) != len(run_names):
        raise ValueError('duplicate admitted job filenames')
    entries = list(jobs_dir.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ValueError('unexpected non-regular entry in signed jobs directory')
    if {item.name for item in entries} != set(run_names):
        raise ValueError('signed job inventory differs from controller-verified protocol')
    job_bytes = []
    envelopes = []
    for name in run_names:
        raw, _ = _read_regular_file(jobs_dir / name, maximum=MAX_ADMISSION_OBJECT_BYTES)
        total_bytes += len(raw)
        if total_bytes > MAX_ADMISSION_TOTAL_BYTES:
            raise ValueError('signed admission exceeds aggregate byte bound')
        envelope = parse_unique_json(raw)
        if _trusted_envelope(envelope, trusted_keys) != pin:
            raise PermissionError('protocol and job signers differ')
        envelopes.append(envelope)
        job_bytes.append((name, raw))
    validate_jobs(protocol, envelopes, pin)

    # The delegated legacy runner reads its key from the admission directory.
    # Stage only verified bytes and the independently pinned key, avoiding a
    # time-of-check/time-of-use read from the caller-supplied directory.
    with tempfile.TemporaryDirectory(prefix='saal-controller-admission-') as temporary:
        staged = Path(temporary)
        (staged / 'jobs').mkdir(mode=0o700)
        (staged / 'controller.pub').write_text(pin, encoding='ascii')
        (staged / 'signed_protocol.json').write_bytes(protocol_bytes)
        for name, raw in job_bytes:
            (staged / 'jobs' / name).write_bytes(raw)
        os.chmod(staged / 'controller.pub', 0o600)
        os.chmod(staged / 'signed_protocol.json', 0o600)
        for name, _ in job_bytes:
            os.chmod(staged / 'jobs' / name, 0o600)
        yield staged


def execute(workspace, admission, artifacts, cache):
    """Run only after resolving a trusted key from controller process config."""
    trust_store_path = os.environ.get(TRUST_STORE_ENV)
    if not trust_store_path:
        raise PermissionError(f'{TRUST_STORE_ENV} must point to a controller-managed trust store')
    from .adaptive_colab_v5 import execute as execute_v5
    with stage_trusted_admission(admission, trust_store_path) as staged:
        return execute_v5(workspace, staged, artifacts, cache)
