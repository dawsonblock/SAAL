"""Signed, one-use controller path for the public static-adapter v5 pilot.

The execution function is a research controller, not an isolation boundary.
Its admission key and attempt authority must be configured outside the input
directory. A completed run remains unqualified and does not authorize runtime
promotion.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import os
import platform
import re
import secrets
import stat
import time
from pathlib import Path

from .adaptive_contract_v5 import validate_pilot_config
from .adaptive_dataset import validate_training_data
from .model_contract import verify
from .pretrained_registry import fetch_snapshot, validate_snapshot
from .spec import canonical, parse_unique_json, sha

TRUST_STORE_SCHEMA = 'saal-static-pilot-trust-store-v1'
ADMISSION_SCHEMA = 'saal-static-pilot-admission-v5'
ATTEMPT_SCHEMA = 'saal-static-pilot-attempt-receipt-v1'
OUTCOME_SCHEMA = 'saal-static-pilot-outcome-receipt-v1'
TRUST_STORE_ENV = 'SAAL_STATIC_PILOT_TRUST_STORE'
GIB = 1024 ** 3
_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$')
_LOCK_REQUIREMENT = re.compile(r'^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\\\s]+)')
_ISSUER = object()


class PilotPreflightError(PermissionError):
    """Sanitized rejection carrying the exact failed preflight stage."""

    def __init__(self, stage, failure_type):
        self.stage = stage
        self.failure_type = failure_type
        super().__init__(f'preflight rejected at {stage} ({failure_type})')


def _run_preflight_stage(stage, operation):
    try:
        return operation()
    except Exception as exc:  # noqa: BLE001 - sanitize every failed preflight operation
        raise PilotPreflightError(stage, type(exc).__name__) from None


def _parse_dependency_lock(raw):
    """Read exact pinned package versions from a hash-bearing requirements lock."""
    packages = {}
    for line in raw.decode('utf-8').splitlines():
        match = _LOCK_REQUIREMENT.match(line.strip())
        if match is None:
            continue
        name = re.sub(r'[-_.]+', '-', match.group(1)).lower()
        version = match.group(2)
        if name in packages and packages[name] != version:
            raise ValueError('conflicting package versions in dependency lock')
        packages[name] = version
    if not packages or b'--hash=sha256:' not in raw:
        raise ValueError('hash-bearing pinned dependency lock required')
    return packages


def _verify_dependency_lock(lock_path, expected_digest):
    import importlib.metadata

    lock_path = Path(lock_path)
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError('regular dependency lock file required')
    raw = lock_path.read_bytes()
    actual_digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise PermissionError('dependency lock digest differs from admission')
    expected = _parse_dependency_lock(raw)
    actual = {}
    for name in expected:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
    mismatches = {name: (version, actual.get(name))
                  for name, version in expected.items()
                  if actual.get(name) != version}
    if mismatches:
        raise PermissionError('installed versions do not match the admitted lock')
    return {'lock_digest': actual_digest, 'package_count': len(expected),
            'versions_match': True,
            'wheel_hash_origin': 'NOT_INDEPENDENTLY_VERIFIED'}


def _bounded_regular_file(path, maximum=1_000_000):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('absolute regular controller file required')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= maximum:
            raise ValueError('bounded regular controller file required')
        raw = os.read(fd, maximum + 1)
        if len(raw) != info.st_size or len(raw) > maximum:
            raise ValueError('controller file changed while reading')
        return raw, info
    finally:
        os.close(fd)


def load_trust_store(path):
    """Load public trust anchors from a controller-owned file, never admission input."""
    raw, info = _bounded_regular_file(path)
    if info.st_uid not in (os.geteuid(), 0) or stat.S_IMODE(info.st_mode) & 0o022:
        raise PermissionError('trust store owner or write permissions are not controller-safe')
    store = parse_unique_json(raw)
    if (type(store) is not dict or set(store) != {'schema', 'admission_keys', 'attempt_authority'}
            or store['schema'] != TRUST_STORE_SCHEMA):
        raise ValueError('static-pilot trust-store schema')
    keys = store['admission_keys']
    if type(keys) is not dict or not keys:
        raise ValueError('admission trust anchors required')
    normalized = {}
    from .validation import key_bytes
    for key_id, entry in keys.items():
        if (type(key_id) is not str or not re.fullmatch(r'[0-9a-f]{64}', key_id)
                or type(entry) is not dict or set(entry) != {'public_key', 'roles'}
                or entry['roles'] != ['static-pilot-admission']):
            raise ValueError('admission trust-store entry')
        public = key_bytes(entry['public_key'])
        if hashlib.sha256(public).hexdigest() != key_id:
            raise ValueError('admission public-key identity mismatch')
        normalized[key_id] = entry['public_key']
    witness = store['attempt_authority']
    if (type(witness) is not dict or set(witness) != {'public_key', 'role'}
            or witness['role'] != 'static-pilot-attempt-authority'):
        raise ValueError('separate attempt-authority trust anchor required')
    witness_key = key_bytes(witness['public_key'])
    witness_id = hashlib.sha256(witness_key).hexdigest()
    if witness_id in normalized:
        raise PermissionError('admission and attempt-authority keys must be distinct')
    return normalized, witness['public_key'], witness_id


def _verify_admission(config, dataset, envelope, trusted_keys, *, now=None):
    if (type(envelope) is not dict or set(envelope) != {
            'payload', 'key_sha256', 'signature_b64'}
            or type(envelope['key_sha256']) is not str):
        raise ValueError('signed v5 admission envelope required')
    pin = trusted_keys.get(envelope['key_sha256'])
    if pin is None:
        raise PermissionError('admission signer is not in controller trust store')
    payload = verify(envelope, pin)
    fields = {
        'schema', 'campaign_id', 'attempt_id', 'issued_at', 'expires_at',
        'configuration_digest', 'dataset_digest', 'source_digest', 'model_digest',
        'tokenizer_digest', 'dependency_digest', 'metric_policy_digest', 'max_optimizer_updates',
        'max_sample_exposures', 'budget', 'runtime',
    }
    if type(payload) is not dict or set(payload) != fields or payload['schema'] != ADMISSION_SCHEMA:
        raise ValueError('closed static-pilot admission schema required')
    for field in ('campaign_id', 'attempt_id'):
        if type(payload[field]) is not str or not _NAME.fullmatch(payload[field]):
            raise ValueError('safe immutable campaign and attempt identities required')
    current = int(time.time()) if now is None else now
    if (type(payload['issued_at']) is not int or type(payload['expires_at']) is not int
            or payload['issued_at'] > current + 300 or payload['expires_at'] <= current
            or payload['expires_at'] <= payload['issued_at']
            or payload['expires_at'] - payload['issued_at'] > 24 * 3600):
        raise PermissionError('admission expired, not yet valid, or outside its validity window')

    validate_pilot_config(config)
    validate_training_data(dataset)
    identity = config['identity']
    expected = {
        'configuration_digest': sha(config),
        'dataset_digest': sha(dataset),
        'source_digest': identity['source_digest'],
        'model_digest': identity['model_digest'],
        'tokenizer_digest': identity['tokenizer_digest'],
        'dependency_digest': identity['dependency_digest'],
        'metric_policy_digest': identity['metric_policy_digest'],
        'max_optimizer_updates': config['steps'],
        'max_sample_exposures': config['sample_exposures'],
    }
    if any(payload[key] != value for key, value in expected.items()):
        raise PermissionError('signed admission does not bind this exact experiment')

    budget = payload['budget']
    if (type(budget) is not dict or set(budget) != {
            'max_wall_seconds', 'max_compute_units', 'compute_unit', 'max_output_bytes'}
            or type(budget['max_wall_seconds']) is not int
            or not 1 <= budget['max_wall_seconds'] <= 7200
            or type(budget['max_compute_units']) is not float
            or not 0 < budget['max_compute_units'] <= 100
            or budget['compute_unit'] != 'colab-compute-unit'
            or type(budget['max_output_bytes']) is not int
            or not 1_000_000 <= budget['max_output_bytes'] <= 2_000_000_000):
        raise ValueError('bounded signed time, compute, and artifact budget required')
    runtime = payload['runtime']
    if (type(runtime) is not dict or set(runtime) != {
            'accelerator', 'minimum_gpu_memory_bytes', 'minimum_free_gpu_memory_bytes',
            'minimum_host_memory_bytes', 'require_dependency_hash_lock'}
            or runtime['accelerator'] != 'A100'
            or type(runtime['minimum_gpu_memory_bytes']) is not int
            or runtime['minimum_gpu_memory_bytes'] < 30 * GIB
            or type(runtime['minimum_free_gpu_memory_bytes']) is not int
            or runtime['minimum_free_gpu_memory_bytes'] < 6 * GIB
            or type(runtime['minimum_host_memory_bytes']) is not int
            or runtime['minimum_host_memory_bytes'] < 32 * GIB
            or runtime['require_dependency_hash_lock'] is not True):
        raise ValueError('signed A100/high-RAM/hash-locked runtime requirements required')
    spec = identity['model_spec']
    if spec['compute_backend'] != 'colab' or spec['precision'] != identity['precision']:
        raise ValueError('Colab model contract required for this pilot')
    return payload


def validate_attempt_receipt(receipt, witness_pin, payload, admission_digest, challenge):
    body = verify(receipt, witness_pin)
    fields = {'schema', 'campaign_id', 'attempt_id', 'admission_digest',
              'request_nonce', 'configuration_digest', 'dataset_digest',
              'bindings_digest', 'sequence', 'status', 'consumed_at'}
    if type(body) is not dict or set(body) != fields or body['schema'] != ATTEMPT_SCHEMA:
        raise ValueError('one-use attempt receipt schema')
    if (body['campaign_id'] != payload['campaign_id']
            or body['attempt_id'] != payload['attempt_id']
            or body['admission_digest'] != admission_digest
            or body['request_nonce'] != challenge
            or body['configuration_digest'] != payload['configuration_digest']
            or body['dataset_digest'] != payload['dataset_digest']
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', body['bindings_digest'])
            or type(body['sequence']) is not int or body['sequence'] < 1
            or body['status'] != 'CONSUMED'
            or type(body['consumed_at']) is not int
            or not payload['issued_at'] <= body['consumed_at'] <= payload['expires_at']):
        raise PermissionError('attempt receipt replay or binding mismatch')
    return body


def validate_outcome_receipt(receipt, witness_pin, payload, attempt_digest,
                             outcome_digest, admission_digest, attempt_sequence,
                             challenge, status):
    body = verify(receipt, witness_pin)
    fields = {'schema', 'campaign_id', 'attempt_id', 'admission_digest',
              'attempt_receipt_digest', 'outcome_digest', 'request_nonce',
              'sequence', 'status', 'finalized_at'}
    if type(body) is not dict or set(body) != fields or body['schema'] != OUTCOME_SCHEMA:
        raise ValueError('terminal attempt receipt schema')
    if (body['campaign_id'] != payload['campaign_id']
            or body['attempt_id'] != payload['attempt_id']
            or body['admission_digest'] != admission_digest
            or body['attempt_receipt_digest'] != attempt_digest
            or body['outcome_digest'] != outcome_digest
            or body['request_nonce'] != challenge
            or body['status'] != status
            or type(body['sequence']) is not int or body['sequence'] <= attempt_sequence
            or type(body['finalized_at']) is not int):
        raise PermissionError('terminal outcome receipt binding mismatch')
    return body


class _ExecutionPermit:
    def __init__(self, *, issuer, config_digest, dataset_digest, admission_digest,
                 attempt_digest, deadline, output_limit):
        if issuer is not _ISSUER:
            raise PermissionError('execution permit is issued only after signed admission')
        self.config_digest = config_digest
        self.dataset_digest = dataset_digest
        self.admission_digest = admission_digest
        self.attempt_digest = attempt_digest
        self.deadline = deadline
        self.output_limit = output_limit


def _observe_colab_runtime(runtime, dependency_digest):
    sys = _run_preflight_stage('python-runtime', lambda: __import__('sys'))
    torch = _run_preflight_stage('torch-import', lambda: __import__('torch'))
    adaptive_backend = _run_preflight_stage(
        'saal-preflight-imports',
        lambda: importlib.import_module('.adaptive_backend', __package__),
    )
    adaptive_runtime = _run_preflight_stage(
        'saal-preflight-imports',
        lambda: importlib.import_module('.adaptive_runtime_v4', __package__),
    )
    learning_backend = _run_preflight_stage(
        'saal-preflight-imports',
        lambda: importlib.import_module('.learning_backend', __package__),
    )

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    versions = _run_preflight_stage('dependency-versions', learning_backend.verify_dependencies)
    implementation = _run_preflight_stage(
        'peft-source-identity', adaptive_backend.verify_peft_implementation)
    lock_path = (Path(__file__).resolve().parents[2] / 'configs/dependency-locks'
                 / 'pretrained-linux-x86_64-py313.txt')
    lock_status = _run_preflight_stage(
        'dependency-lock-versions',
        lambda: _verify_dependency_lock(lock_path, dependency_digest),
    )
    os_name = _run_preflight_stage('platform-and-python', platform.system)
    machine = _run_preflight_stage('platform-and-python', platform.machine)
    if (os_name != 'Linux' or machine != 'x86_64'
            or sys.version_info[:2] != (3, 13)):
        raise PilotPreflightError('platform-and-python', 'UnsupportedRuntime')
    if not _run_preflight_stage('cuda-detection', torch.cuda.is_available):
        raise PilotPreflightError('cuda-detection', 'CudaUnavailable')
    properties = _run_preflight_stage('gpu-properties', lambda: torch.cuda.get_device_properties(0))
    free, _ = _run_preflight_stage('gpu-memory', lambda: torch.cuda.mem_get_info(0))
    name = _run_preflight_stage('gpu-identity', lambda: torch.cuda.get_device_name(0))
    observed_ram = _run_preflight_stage('host-memory', adaptive_runtime.host_memory_bytes)
    if (runtime['accelerator'].lower() not in name.lower()
            or int(properties.total_memory) < runtime['minimum_gpu_memory_bytes']
            or int(free) < runtime['minimum_free_gpu_memory_bytes']
            or observed_ram < runtime['minimum_host_memory_bytes']):
        raise PilotPreflightError('resource-requirements', 'InsufficientRuntimeResources')
    if os.environ.get('SAAL_HASH_LOCK_ENFORCED') != '1':
        raise PilotPreflightError('dependency-lock-installation', 'HashLockNotConfirmed')
    return {
        'device': 'cuda:0', 'gpu': name,
        'gpu_memory_bytes': int(properties.total_memory),
        'free_gpu_memory_bytes': int(free), 'host_memory_bytes': observed_ram,
        'dependency_versions': versions,
        'peft_implementation_sha256': sha(implementation),
        'dependency_lock_status': lock_status,
        'hash_lock_installer_flag_asserted': True,
    }


def _write_artifact_inventory(root, maximum_bytes):
    files = {}
    total = 0
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in pilot output')
        if path.is_file() and path.name != 'artifact_inventory.json':
            size = path.stat().st_size
            total += size
            if size > maximum_bytes or total > maximum_bytes:
                raise ValueError('pilot artifact output budget exceeded')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            files[path.relative_to(root).as_posix()] = {
                'size': size,
                'sha256': digest,
            }
    inventory = {'schema': 'saal-static-pilot-artifact-inventory-v5', 'files': files}
    (Path(root) / 'artifact_inventory.json').write_bytes(canonical(inventory))
    return inventory


def execute_admitted_static_pilot(config, dataset, admission_envelope, *,
                                  attempt_authority, model_cache, output_dir):
    """Run one exact public pilot arm after signed preflight and external consumption.

    ``attempt_authority.consume_attempt(bindings, nonce)`` must be a client to
    an independently administered one-use authority. A local fake is suitable
    only for tests. No fallback, retry, resume, or budget reset is provided.
    """
    config = copy.deepcopy(config)
    dataset = copy.deepcopy(dataset)
    validate_pilot_config(config)
    validate_training_data(dataset)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError('attempt output path must be new')
    trust_store_path = os.environ.get(TRUST_STORE_ENV)
    if not trust_store_path:
        raise PermissionError(f'{TRUST_STORE_ENV} must point to controller-managed trust anchors')
    trust_path = Path(trust_store_path)
    admission_keys, witness_pin, _ = load_trust_store(trust_path)
    payload = _verify_admission(config, dataset, admission_envelope, admission_keys)
    admission_digest = sha(admission_envelope)
    output.mkdir(parents=True, mode=0o700)
    (output / 'admission.json').write_bytes(canonical(admission_envelope))
    try:
        runtime = _observe_colab_runtime(
            payload['runtime'], config['identity']['dependency_digest'])
    except PilotPreflightError as exc:
        (output / 'preflight_failure.json').write_bytes(canonical({
            'schema': 'saal-static-pilot-preflight-failure-v5',
            'campaign_id': payload['campaign_id'], 'attempt_id': payload['attempt_id'],
            'admission_digest': admission_digest,
            'status': 'PREFLIGHT_REJECTED_NO_ATTEMPT_CONSUMED',
            'stage': exc.stage, 'failure_type': exc.failure_type,
        }))
        raise
    (output / 'runtime_preflight.json').write_bytes(canonical(runtime))
    challenge = secrets.token_hex(32)
    bindings = {
        'schema': 'saal-static-pilot-attempt-bindings-v1',
        'campaign_id': payload['campaign_id'],
        'attempt_id': payload['attempt_id'],
        'admission_digest': admission_digest,
        'configuration_digest': sha(config),
        'dataset_digest': sha(dataset),
        'source_digest': config['identity']['source_digest'],
        'dependency_digest': config['identity']['dependency_digest'],
        'metric_policy_digest': config['identity']['metric_policy_digest'],
        'model_digest': config['identity']['model_digest'],
        'budget_digest': sha(payload['budget']),
    }
    if (not callable(getattr(attempt_authority, 'consume_attempt', None))
            or not callable(getattr(attempt_authority, 'finalize_attempt', None))):
        raise PermissionError('separately administered one-use attempt authority required')
    try:
        receipt = attempt_authority.consume_attempt(bindings, challenge)
    except Exception as exc:
        # A timeout can mean the authority consumed the one-use attempt but its
        # response was lost. Preserve uncertainty and never submit a retry.
        (output / 'launch_uncertain.json').write_bytes(canonical({
            'schema': 'saal-static-pilot-launch-uncertain-v1',
            'campaign_id': payload['campaign_id'], 'attempt_id': payload['attempt_id'],
            'admission_digest': admission_digest, 'request_nonce': challenge,
            'status': 'LAUNCH_UNCERTAIN', 'automatic_retry': False,
            'failure_type': type(exc).__name__,
        }))
        raise RuntimeError('attempt authority response uncertain; preserve and do not retry') from exc
    try:
        receipt_body = validate_attempt_receipt(
            receipt, witness_pin, payload, admission_digest, challenge)
    except Exception as exc:
        (output / 'attempt_receipt_untrusted.json').write_bytes(canonical(receipt))
        (output / 'launch_uncertain.json').write_bytes(canonical({
            'schema': 'saal-static-pilot-launch-uncertain-v1',
            'campaign_id': payload['campaign_id'], 'attempt_id': payload['attempt_id'],
            'admission_digest': admission_digest, 'request_nonce': challenge,
            'status': 'LAUNCH_UNCERTAIN', 'automatic_retry': False,
            'failure_type': type(exc).__name__,
        }))
        raise RuntimeError('attempt receipt failed verification; preserve and do not retry') from exc
    receipt_digest = sha(receipt)
    permit = _ExecutionPermit(
        issuer=_ISSUER, config_digest=sha(config), dataset_digest=sha(dataset),
        admission_digest=admission_digest, attempt_digest=receipt_digest,
        deadline=time.monotonic() + payload['budget']['max_wall_seconds'],
        output_limit=payload['budget']['max_output_bytes'],
    )
    finalization_attempted = False
    terminal_receipt_verified = False
    try:
        # Any post-consumption validation or persistence failure is finalized
        # as a failed spent attempt; it must not escape the accounting path.
        if receipt_body['bindings_digest'] != sha(bindings):
            raise PermissionError('attempt authority did not bind every signed resource field')
        (output / 'attempt_receipt.json').write_bytes(canonical(receipt))
        from .adaptive_backend_v5 import StaticPilotBackendV5
        from .adaptive_training_v5 import _train_static_pilot

        remaining = permit.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('admission wall-clock budget exhausted before model retrieval')
        spec = config['identity']['model_spec']
        snapshot = fetch_snapshot(model_cache, spec['model_repository'], spec['model_revision'])
        snapshot_manifest = validate_snapshot(snapshot, spec['model_repository'], spec['model_revision'])
        if time.monotonic() >= permit.deadline:
            raise TimeoutError('admission wall-clock budget exhausted during model retrieval')
        (output / 'snapshot_manifest.json').write_bytes(canonical(snapshot_manifest))
        backend = StaticPilotBackendV5(config, snapshot).load_pretrained()
        if time.monotonic() >= permit.deadline:
            raise TimeoutError('admission wall-clock budget exhausted during model initialization')
        import torch
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        result = _train_static_pilot(
            config, dataset, backend, output / 'training',
            deadline_seconds=permit.deadline-time.monotonic(),
            execution_permit=permit,
        )
        result['admission_digest'] = admission_digest
        result['attempt_receipt_digest'] = receipt_digest
        result['attempt_sequence'] = receipt_body['sequence']
        result['runtime_preflight'] = runtime
        result['campaign_status'] = 'COMPLETED_UNQUALIFIED'
        result['resource_accounting'].update({
            'compute_units_budgeted': payload['budget']['max_compute_units'],
            'compute_unit': payload['budget']['compute_unit'],
            'compute_units_actual': None,
            'compute_units_status': 'NOT_INDEPENDENTLY_MEASURED',
        })
        (output / 'result.json').write_bytes(canonical(result))
        training_inventory = _write_artifact_inventory(output, permit.output_limit)
        training_inventory_digest = sha(training_inventory)
        outcome_digest = sha({
            'result_digest': sha(result),
            'training_artifact_inventory_digest': training_inventory_digest,
        })
        finish_nonce = secrets.token_hex(32)
        finalization_attempted = True
        terminal_receipt = attempt_authority.finalize_attempt(
            receipt, 'COMPLETED', outcome_digest, finish_nonce,
        )
        validate_outcome_receipt(
            terminal_receipt, witness_pin, payload, receipt_digest,
            outcome_digest, admission_digest, receipt_body['sequence'],
            finish_nonce, 'COMPLETED',
        )
        terminal_receipt_verified = True
        (output / 'terminal_receipt.json').write_bytes(canonical(terminal_receipt))
        result['finalized_outcome_digest'] = outcome_digest
        result['training_artifact_inventory_digest'] = training_inventory_digest
        result['terminal_receipt_digest'] = sha(terminal_receipt)
        (output / 'result.json').write_bytes(canonical(result))
        _write_artifact_inventory(output, permit.output_limit)
        return result
    except BaseException as exc:
        failure = {
            'schema': 'saal-static-pilot-attempt-failure-v1',
            'campaign_id': payload['campaign_id'], 'attempt_id': payload['attempt_id'],
            'admission_digest': admission_digest, 'attempt_receipt_digest': receipt_digest,
            'status': ('COMPLETION_ARTIFACT_COMMIT_UNCERTAIN'
                       if finalization_attempted and terminal_receipt_verified
                       else 'INTERRUPTED_OR_FAILED'),
            'failure_type': type(exc).__name__,
            'failure_message': str(exc)[:2000], 'budget_refunded': False,
            'automatic_retry': False,
        }
        try:
            if finalization_attempted:
                raise RuntimeError('terminal outcome request was already attempted; do not retry')
            outcome_digest = sha(failure)
            finish_nonce = secrets.token_hex(32)
            finalization_attempted = True
            terminal_receipt = attempt_authority.finalize_attempt(
                receipt, 'FAILED', outcome_digest, finish_nonce,
            )
            validate_outcome_receipt(
                terminal_receipt, witness_pin, payload, receipt_digest,
                outcome_digest, admission_digest, receipt_body['sequence'],
                finish_nonce, 'FAILED',
            )
            failure['finalized_outcome_digest'] = outcome_digest
            failure['terminal_receipt_digest'] = sha(terminal_receipt)
            (output / 'terminal_receipt.json').write_bytes(canonical(terminal_receipt))
        except (OSError, RuntimeError, TypeError, ValueError, PermissionError) as receipt_error:
            failure['terminal_receipt_status'] = 'UNCONFIRMED_AUTHORITY_UNAVAILABLE'
            failure['terminal_receipt_error'] = type(receipt_error).__name__
        (output / 'failure.json').write_bytes(canonical(failure))
        try:
            _write_artifact_inventory(output, permit.output_limit)
        except (OSError, ValueError) as inventory_error:
            # Preserve the attempt failure even when it cannot fit in the signed
            # output allowance; never retry or replace this spent attempt.
            failure['artifact_inventory_status'] = 'UNAVAILABLE'
            failure['artifact_inventory_error'] = type(inventory_error).__name__
            (output / 'failure.json').write_bytes(canonical(failure))
        raise
