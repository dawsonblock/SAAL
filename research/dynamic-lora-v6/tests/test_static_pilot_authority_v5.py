import copy
import hashlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.adaptive_contract_v5 import (
    build_static_adalora_pilot,
    scientific_source_digest,
)
from saal_rsi.adaptive_dataset import training_projection
from saal_rsi.adaptive_static_pilot_v5 import (
    ADMISSION_SCHEMA,
    ATTEMPT_SCHEMA,
    TRUST_STORE_ENV,
    PilotPreflightError,
    _run_preflight_stage,
    _verify_admission,
    _verify_dependency_lock,
    execute_admitted_static_pilot,
    validate_attempt_receipt,
    validate_outcome_receipt,
)
from saal_rsi.model_contract import sign
from saal_rsi.numerical_metrics_v2 import POLICY_DIGEST
from saal_rsi.pretrained_contract import model_spec, targets
from saal_rsi.spec import sha


def _key_id(key):
    return hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()


def _admission(config, dataset, key, *, issued=1000, expires=2000):
    identity = config['identity']
    payload = {
        'schema': ADMISSION_SCHEMA,
        'campaign_id': 'static-pilot-test',
        'attempt_id': 'arm-zero-seed-41',
        'issued_at': issued,
        'expires_at': expires,
        'configuration_digest': sha(config),
        'dataset_digest': sha(dataset),
        'source_digest': identity['source_digest'],
        'model_digest': identity['model_digest'],
        'tokenizer_digest': identity['tokenizer_digest'],
        'dependency_digest': identity['dependency_digest'],
        'metric_policy_digest': identity['metric_policy_digest'],
        'max_optimizer_updates': config['steps'],
        'max_sample_exposures': config['sample_exposures'],
        'budget': {
            'max_wall_seconds': 1800,
            'max_compute_units': 2.0,
            'compute_unit': 'colab-compute-unit',
            'max_output_bytes': 100_000_000,
        },
        'runtime': {
            'accelerator': 'A100',
            'minimum_gpu_memory_bytes': 30 * 1024**3,
            'minimum_free_gpu_memory_bytes': 6 * 1024**3,
            'minimum_host_memory_bytes': 32 * 1024**3,
            'require_dependency_hash_lock': True,
        },
    }
    return sign(payload, key)


def _fixture_config(dataset):
    spec = model_spec('answerdotai/ModernBERT-base', method='none', backend='colab',
                      precision='float16', budget='dynamic-lora-v5-pilot')
    lock = Path('configs/dependency-locks/pretrained-linux-x86_64-py313.txt').read_bytes()
    identity = {
        'model_spec': spec,
        'model_digest': spec['model_files_digest'],
        'tokenizer_digest': spec['tokenizer_digest'],
        'dataset_digest': sha(dataset),
        'source_digest': scientific_source_digest(),
        'dependency_digest': 'sha256:' + hashlib.sha256(lock).hexdigest(),
        'metric_policy_digest': POLICY_DIGEST,
        'sample_order_digest': 'sha256:' + 'a' * 64,
        'seed': 41,
        'sampler_seed': 43,
        'dropout_seed': 47,
        'batch_size': 8,
        'accumulation': 2,
        'adapter_learning_rate': 0.0003,
        'head_learning_rate': 0.0003,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'target_modules': targets('modernbert-seqcls-v1'),
        'precision': 'float16',
        'class_balance': False,
    }
    return build_static_adalora_pilot(identity, 0.0)


def test_v5_admission_binds_every_scientific_input_and_signed_resource_gate(population):
    dataset = training_projection(population)
    config = _fixture_config(dataset)
    signer = Ed25519PrivateKey.generate()
    envelope = _admission(config, dataset, signer)
    keys = {_key_id(signer): signer.public_key().public_bytes_raw().hex()}

    payload = _verify_admission(config, dataset, envelope, keys, now=1200)
    assert payload['configuration_digest'] == sha(config)
    assert payload['dataset_digest'] == sha(dataset)
    assert payload['runtime']['accelerator'] == 'A100'
    assert payload['max_optimizer_updates'] == 200

    changed = copy.deepcopy(dataset)
    changed['sampling_description'] += ' modified'
    with pytest.raises(PermissionError, match='exact experiment'):
        _verify_admission(config, changed, envelope, keys, now=1200)


def test_v5_admission_rejects_untrusted_signer_expiry_and_weakened_runtime(population):
    dataset = training_projection(population)
    config = _fixture_config(dataset)
    trusted = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    trusted_keys = {_key_id(trusted): trusted.public_key().public_bytes_raw().hex()}

    with pytest.raises(PermissionError, match='not in controller trust store'):
        _verify_admission(config, dataset, _admission(config, dataset, other), trusted_keys, now=1200)
    with pytest.raises(PermissionError, match='expired'):
        _verify_admission(config, dataset, _admission(config, dataset, trusted, expires=1100),
                          trusted_keys, now=1200)

    altered = _admission(config, dataset, trusted)
    altered['payload']['runtime']['minimum_host_memory_bytes'] = 1
    altered = sign(altered['payload'], trusted)
    with pytest.raises(ValueError, match='A100/high-RAM'):
        _verify_admission(config, dataset, altered, trusted_keys, now=1200)


def test_one_use_receipt_binds_admission_configuration_data_and_fresh_nonce(population):
    dataset = training_projection(population)
    config = _fixture_config(dataset)
    signer = Ed25519PrivateKey.generate()
    envelope = _admission(config, dataset, signer, issued=1000, expires=2000)
    payload = envelope['payload']
    admission_digest = sha(envelope)
    witness = Ed25519PrivateKey.generate()
    challenge = 'c' * 64
    bindings = {
        'campaign_id': payload['campaign_id'],
        'attempt_id': payload['attempt_id'],
        'admission_digest': admission_digest,
        'configuration_digest': sha(config),
        'dataset_digest': sha(dataset),
    }
    body = {
        'schema': ATTEMPT_SCHEMA,
        'campaign_id': payload['campaign_id'],
        'attempt_id': payload['attempt_id'],
        'admission_digest': admission_digest,
        'request_nonce': challenge,
        'configuration_digest': sha(config),
        'dataset_digest': sha(dataset),
        'bindings_digest': sha(bindings),
        'sequence': 1,
        'status': 'CONSUMED',
        'consumed_at': 1200,
    }
    receipt = sign(body, witness)
    pin = witness.public_key().public_bytes_raw().hex()
    assert validate_attempt_receipt(
        receipt, pin, payload, admission_digest, challenge)['sequence'] == 1
    with pytest.raises(PermissionError, match='replay or binding'):
        validate_attempt_receipt(receipt, pin, payload, admission_digest, 'd' * 64)
    forged = copy.deepcopy(receipt)
    forged['payload']['sequence'] = 2
    with pytest.raises(ValueError, match='invalid signature'):
        validate_attempt_receipt(forged, pin, payload, admission_digest, challenge)


def test_outcome_receipt_binds_complete_admission_and_advances_attempt_sequence(population):
    dataset = training_projection(population)
    config = _fixture_config(dataset)
    signer = Ed25519PrivateKey.generate()
    envelope = _admission(config, dataset, signer, issued=1000, expires=2000)
    payload = envelope['payload']
    admission_digest = sha(envelope)
    witness = Ed25519PrivateKey.generate()
    witness_pin = witness.public_key().public_bytes_raw().hex()
    attempt_digest = 'sha256:' + '1' * 64
    outcome_digest = 'sha256:' + '2' * 64
    challenge = 'e' * 64
    body = {
        'schema': 'saal-static-pilot-outcome-receipt-v1',
        'campaign_id': payload['campaign_id'],
        'attempt_id': payload['attempt_id'],
        'admission_digest': admission_digest,
        'attempt_receipt_digest': attempt_digest,
        'outcome_digest': outcome_digest,
        'request_nonce': challenge,
        'sequence': 3,
        'status': 'COMPLETED',
        'finalized_at': 1500,
    }
    receipt = sign(body, witness)
    assert validate_outcome_receipt(
        receipt, witness_pin, payload, attempt_digest, outcome_digest,
        admission_digest, 2, challenge, 'COMPLETED',
    )['sequence'] == 3
    with pytest.raises(PermissionError, match='terminal outcome receipt binding'):
        validate_outcome_receipt(
            receipt, witness_pin, payload, attempt_digest, outcome_digest,
            admission_digest, 3, challenge, 'COMPLETED',
        )


def test_controller_fails_closed_before_attempt_consumption_without_pinned_trust_store(
    population, tmp_path, monkeypatch,
):
    dataset = training_projection(population)
    config = _fixture_config(dataset)
    signer = Ed25519PrivateKey.generate()
    admission = _admission(config, dataset, signer)

    class Authority:
        consumed = False

        def consume_attempt(self, _bindings, _nonce):
            self.consumed = True
            raise AssertionError('attempt must not be consumed')

    authority = Authority()
    monkeypatch.delenv(TRUST_STORE_ENV, raising=False)
    with pytest.raises(PermissionError, match='controller-managed trust anchors'):
        execute_admitted_static_pilot(
            config, dataset, admission, attempt_authority=authority,
            model_cache=tmp_path / 'cache', output_dir=tmp_path / 'out',
        )
    assert authority.consumed is False
    assert not (tmp_path / 'out').exists()


def test_preflight_failure_keeps_exact_stage_and_excludes_untrusted_error_text():
    def broken_probe():
        raise OSError('temporary path / secret payload should not be copied into evidence')

    with pytest.raises(PilotPreflightError) as error:
        _run_preflight_stage('gpu-properties', broken_probe)

    assert error.value.stage == 'gpu-properties'
    assert error.value.failure_type == 'OSError'
    assert 'secret payload' not in str(error.value)


def test_runtime_dependency_lock_checks_pinned_versions_and_exact_lock_digest(tmp_path):
    import importlib.metadata

    version = importlib.metadata.version('pytest')
    raw = f'pytest=={version} \\\n    --hash=sha256:{"a" * 64}\n'.encode()
    lock_path = tmp_path / 'requirements.lock'
    lock_path.write_bytes(raw)
    expected_digest = 'sha256:' + hashlib.sha256(raw).hexdigest()

    status = _verify_dependency_lock(lock_path, expected_digest)
    assert status['versions_match'] is True
    assert status['package_count'] == 1
    assert status['wheel_hash_origin'] == 'NOT_INDEPENDENTLY_VERIFIED'
    with pytest.raises(PermissionError, match='lock digest'):
        _verify_dependency_lock(lock_path, 'sha256:' + '0' * 64)


def test_dependency_lock_parser_rejects_missing_hashes():
    from saal_rsi.adaptive_static_pilot_v5 import _parse_dependency_lock

    with pytest.raises(ValueError, match='hash-bearing'):
        _parse_dependency_lock(b'pytest==8.0.0\\n')
