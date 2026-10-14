import importlib.util
from pathlib import Path
import zipfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('adaptive_capsule_tested', ROOT/'tools/adaptive_capsule.py')
capsule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capsule)


@pytest.fixture
def release(tmp_path):
    import base64
    import hashlib
    key = Ed25519PrivateKey.generate()
    pin = key.public_key().public_bytes_raw().hex()
    source = b'preserved source bytes\n'
    payload = b'actual binary tensor fixture, not an LFS pointer'
    h = hashlib.sha256(payload).hexdigest()
    files = {'src/example.py': source, 'tensor_payloads/'+h+'.safetensors': payload,
        'research/reliable-learning-v3/PREREGISTERED_FILES.json': capsule.canonical(
            {'src/example.py': hashlib.sha256(source).hexdigest()}),
        'TENSOR_PAYLOADS.json': capsule.canonical({'schema': 'saal-physical-tensor-payloads-v1',
            'files': {'example.safetensors': {'sha256': h, 'size': len(payload),
                      'payload': 'tensor_payloads/'+h+'.safetensors'}}}),
        'SOURCE_IDENTITY.json': capsule.canonical({'schema': 'saal-published-source-identity-v1',
            'source_commit': 'a'*40, 'files': {'src/example.py': hashlib.sha256(source).hexdigest()}}),
        'PACKAGE_LAYOUT.json': capsule.canonical({'schema': 'saal-complete-physical-capsule-v6',
            'source_commit': 'a'*40, 'evidence_commit': 'b'*40,
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False})}
    hashes = {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}
    hashes['example.safetensors'] = h
    manifest = ''.join(h+'  '+n+'\n' for n, h in sorted(hashes.items())).encode()
    signed = {'schema': 'saal-physical-delivery-v6',
        'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
        'source_commit': 'a'*40, 'evidence_commit': 'b'*40,
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    files['MANIFEST.sha256'] = manifest
    files['DELIVERY_SIGNATURE.json'] = capsule.canonical({'payload': signed,
        'key_sha256': hashlib.sha256(bytes.fromhex(pin)).hexdigest(),
        'signature_b64': base64.b64encode(key.sign(capsule.canonical(signed))).decode()})
    for n, data in files.items():
        p = tmp_path/n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return tmp_path, pin


def test_clean_extraction_all_bytes_signature_and_alias_restoration(release, tmp_path):
    root, pin = release
    archive = tmp_path.parent/(tmp_path.name+'.zip')
    with zipfile.ZipFile(archive, 'x') as z:
        for p in root.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())
    clean = tmp_path.parent/(tmp_path.name+'-clean')
    extracted = capsule.extract(archive, clean, capsule.digest(archive))
    result = capsule.verify_files(clean, pin)
    assert extracted['members_read_completely'] == 8
    assert result['checkpoint_paths_restored'] == 1
    assert result['historical_frozen_files'] == 1
    assert not result['external_deployment_allowed']


@pytest.mark.parametrize('fault', ['bytes', 'extra', 'missing', 'signer', 'manifest', 'symlink'])
def test_physical_release_negative_cases(release, fault):
    root, pin = release
    if fault == 'bytes':
        (root/'src/example.py').write_bytes(b'substituted')
    elif fault == 'extra':
        (root/'extra.txt').write_text('unregistered')
    elif fault == 'missing':
        (root/'src/example.py').unlink()
    elif fault == 'signer':
        pin = '0'*64
    elif fault == 'manifest':
        with (root/'MANIFEST.sha256').open('ab') as stream:
            stream.write(b'0'*64+b'  extra.txt\n')
    else:
        p = root/'src/example.py'
        p.unlink()
        p.symlink_to(root/'PACKAGE_LAYOUT.json')
    with pytest.raises(ValueError):
        capsule.verify_files(root, pin)


@pytest.mark.parametrize('names', [['../escape'], ['/absolute'], ['same', 'SAME'], ['a\\b']])
def test_unsafe_archive_paths_rejected_before_extraction(tmp_path, names):
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'x') as z:
        for n in names:
            z.writestr(n, b'bad')
    with pytest.raises(ValueError):
        capsule.extract(archive, tmp_path/'never-created', capsule.digest(archive))
    assert not (tmp_path/'never-created').exists()
