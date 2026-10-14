"""V6 physical capsule: authenticate real bytes, preserve v5, verify before use.

The upstream pretrained cache is an explicitly separate prerequisite. A signed
package proves controller attribution and integrity, not scientific qualification.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
META = {'MANIFEST.sha256', 'DELIVERY_SIGNATURE.json', 'PACKAGE_LAYOUT.json',
        'SOURCE_IDENTITY.json', 'TENSOR_PAYLOADS.json', 'materialize_tensors.py'}
V5_HASH = '47b417156cea940814d7456c59507f036358bc127dc83914eb737984b0efd00b'
V5_PIN = 'ccc792a3f49c372eaa3c66a9089e02e4350e19e63f4e677768561298caf32fc2'
LFS = b'version https://git-lfs.github.com/spec/v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def read_json(raw):
    def pairs(rows):
        out = {}
        for name, value in rows:
            if name in out:
                raise ValueError('duplicate JSON field')
            out[name] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    if (type(name) is not str or not name or Path(name).is_absolute()
            or '..' in Path(name).parts or '\\' in name or '\n' in name
            or '\r' in name or Path(name).as_posix() != name):
        raise ValueError('unsafe capsule path')
    return name


def parse_manifest(raw):
    entries = {}
    for line in raw.decode().splitlines():
        h, name = line.split('  ', 1)
        safe_name(name)
        if not re.fullmatch('[0-9a-f]{64}', h) or name.casefold() in {n.casefold() for n in entries}:
            raise ValueError('manifest hash or duplicate path')
        entries[name] = h
    if not entries:
        raise ValueError('empty manifest')
    return entries


def verify_signature(envelope, pin):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if (not re.fullmatch('[0-9a-f]{64}', pin) or type(envelope) is not dict
            or set(envelope) != {'payload', 'key_sha256', 'signature_b64'}):
        raise ValueError('signature envelope or pinned public identity')
    public = bytes.fromhex(pin)
    if envelope['key_sha256'] != hashlib.sha256(public).hexdigest():
        raise ValueError('signer substitution')
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(
            base64.b64decode(envelope['signature_b64'], validate=True), canonical(envelope['payload']))
    except Exception as exc:
        raise ValueError('signature verification failed') from exc
    return envelope['payload']


def capsule_identity(raw, envelope, pin, schema='saal-physical-delivery-v6'):
    signed = verify_signature(envelope, pin)
    if (signed.get('schema') != schema
            or signed.get('manifest_sha256') != hashlib.sha256(raw).hexdigest()
            or signed.get('qualification_status') != 'NOT_EVALUATED'
            or signed.get('external_deployment_allowed') is not False):
        raise ValueError('delivery signature scope')
    return signed


def inventory_zip(z):
    infos = z.infolist()
    names = [safe_name(i.filename) for i in infos]
    if (len(infos) > 12000 or len({n.casefold() for n in names}) != len(names)
            or sum(i.file_size for i in infos) > 20_000_000_000
            or any(i.is_dir() or stat.S_ISLNK(i.external_attr >> 16) for i in infos)):
        raise ValueError('ZIP inventory bounds or links')
    return infos


def verify_files(root, pin):
    root = Path(root).resolve()
    raw = (root/'MANIFEST.sha256').read_bytes()
    signed = capsule_identity(raw, read_json((root/'DELIVERY_SIGNATURE.json').read_bytes()), pin)
    entries = parse_manifest(raw)
    for n in ('PACKAGE_LAYOUT.json', 'SOURCE_IDENTITY.json', 'TENSOR_PAYLOADS.json'):
        if digest(root/n) != entries.get(n):
            raise ValueError('untrusted identity metadata')
    layout = read_json((root/'PACKAGE_LAYOUT.json').read_bytes())
    source = read_json((root/'SOURCE_IDENTITY.json').read_bytes())
    if (layout.get('schema') != 'saal-complete-physical-capsule-v6'
            or source.get('schema') != 'saal-published-source-identity-v1'
            or layout.get('source_commit') != signed.get('source_commit')
            or layout.get('evidence_commit') != signed.get('evidence_commit')
            or source.get('source_commit') != signed.get('source_commit')
            or not source.get('files')
            or any(entries.get(n) != h for n, h in source['files'].items())
            or layout.get('qualification_status') != 'NOT_EVALUATED'
            or layout.get('external_deployment_allowed') is not False):
        raise ValueError('source/evidence identity mismatch')
    mapping = read_json((root/'TENSOR_PAYLOADS.json').read_bytes())
    if set(mapping) != {'schema', 'files'} or mapping['schema'] != 'saal-physical-tensor-payloads-v1':
        raise ValueError('tensor restoration schema')
    aliases = mapping['files']
    for n, item in aliases.items():
        safe_name(n)
        if (not n.endswith('.safetensors') or set(item) != {'sha256', 'size', 'payload'}
                or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
                or type(item['size']) is not int or not 1 <= item['size'] <= 650_000_000
                or item['payload'] != 'tensor_payloads/'+item['sha256']+'.safetensors'
                or entries.get(n) != item['sha256'] or entries.get(item['payload']) != item['sha256']):
            raise ValueError('tensor alias identity')
    # Authenticate every physical file before restoring paths or importing code.
    for n, h in entries.items():
        p = root/n
        if not p.parent.resolve().is_relative_to(root):
            raise ValueError('parent symlink escape')
        if not p.exists():
            if n not in aliases:
                raise ValueError('missing release file '+n)
            continue
        if p.is_symlink() or not p.is_file() or digest(p) != h:
            raise ValueError('physical file substitution '+n)
        with p.open('rb') as stream:
            if stream.read(80).startswith(LFS):
                raise ValueError('LFS pointer instead of payload '+n)
    restored = 0
    for n, item in aliases.items():
        source_path, destination = root/item['payload'], root/n
        if source_path.stat().st_size != item['size']:
            raise ValueError('tensor payload length')
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_path, destination)
            restored += 1
    # Unregistered bytecode can override authenticated Python source. Verify a
    # clean extraction before execution; generated caches are not release files.
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file()} - {'MANIFEST.sha256', 'DELIVERY_SIGNATURE.json'}
    if actual != set(entries):
        raise ValueError('extra/missing capsule files')
    for n, item in aliases.items():
        if digest(root/n) != item['sha256']:
            raise ValueError('restored tensor mismatch')
    frozen = read_json((root/'research/reliable-learning-v3/PREREGISTERED_FILES.json').read_bytes())
    for n, h in frozen.items():
        safe_name(n)
        if digest(root/n) != h:
            raise ValueError('historical frozen source changed '+n)
    return {'logical_files_verified': len(entries), 'checkpoint_paths_restored': restored,
            'historical_frozen_files': len(frozen), 'source_files_bound': len(source['files']),
            'source_commit': signed['source_commit'], 'evidence_commit': signed['evidence_commit'],
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}


def extract(archive, destination, expected_sha256):
    archive, destination = Path(archive), Path(destination)
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256) or digest(archive) != expected_sha256:
        raise ValueError('archive authentication failed')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('new clean destination required')
    with zipfile.ZipFile(archive) as z:
        infos = inventory_zip(z)
        destination.mkdir(parents=True)
        for item in infos:
            p = destination/item.filename
            p.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, p.open('xb') as target:
                shutil.copyfileobj(source, target, 1024*1024)
    return {'archive_sha256': expected_sha256, 'members_read_completely': len(infos),
            'clean_directory': str(destination)}


def package(destination, baseline, key_path, source_ref, evidence_ref):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    destination, baseline = Path(destination), Path(baseline)
    if digest(baseline) != V5_HASH:
        raise ValueError('authenticated v5 baseline changed')
    source_commit = subprocess.check_output(['git', 'rev-parse', source_ref], cwd=ROOT, text=True).strip()
    evidence_commit = subprocess.check_output(['git', 'rev-parse', evidence_ref], cwd=ROOT, text=True).strip()
    with zipfile.ZipFile(baseline) as old:
        inventory_zip(old)
        old_raw = old.read('MANIFEST.sha256')
        capsule_identity(old_raw, read_json(old.read('DELIVERY_SIGNATURE.json')), V5_PIN,
                         'saal-physical-delivery-v5')
        manifest = parse_manifest(old_raw)
        mapping = read_json(old.read('TENSOR_PAYLOADS.json'))
        aliases = mapping['files']
        base_names = set(old.namelist())-META
        manifest = {n: h for n, h in manifest.items() if n not in META}
        overrides = {}
        # Include tracked release files, not unrelated untracked transports.
        tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        for n in filter(None, tracked):
            if n in META or n in aliases:
                continue
            p = ROOT/safe_name(n)
            if p.is_symlink() or not p.is_file() or p.suffix in ('.key', '.ed25519'):
                raise ValueError('unsafe or missing tracked file '+n)
            with p.open('rb') as stream:
                if stream.read(80).startswith(LFS):
                    raise ValueError('unhydrated tracked payload '+n)
            h = digest(p)
            if n not in base_names or manifest.get(n) != h:
                overrides[n] = p
                manifest[n] = h
        generated = {}
        # Retain old signed delivery metadata byte-for-byte under provenance.
        for n in META:
            name = 'provenance/evidence-closure-v5-delivery/'+n
            data = old.read(n)
            generated[name] = data
            manifest[name] = hashlib.sha256(data).hexdigest()
        source_identity = {}
        archive = subprocess.check_output(['git', 'archive', '--format=tar', source_commit,
            'src', 'configs', 'tests', 'docs', 'tools', 'pyproject.toml',
            'research/dynamic-lora-v6'], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
            for member in tree:
                if (not member.isfile() or '/native-engineering-' in member.name
                        or Path(member.name).suffix not in ('.py', '.json', '.toml', '.txt', '.yaml', '.yml', '.md', '.pub')):
                    continue
                data = tree.extractfile(member).read()
                if data.startswith(LFS):
                    # Compare the actual LFS object hash, never authenticate a pointer as payload.
                    match = re.search(rb'oid sha256:([0-9a-f]{64})', data)
                    if not match:
                        raise ValueError('malformed LFS source pointer')
                    h = match[1].decode()
                else:
                    h = hashlib.sha256(data).hexdigest()
                if manifest.get(member.name) != h:
                    raise ValueError('source differs from published commit '+member.name)
                source_identity[member.name] = h
        results = read_json((ROOT/'research/dynamic-lora-v6/RESULTS.json').read_bytes())
        layout = {'schema': 'saal-complete-physical-capsule-v6', 'version': '0.9.5.dev6',
            'source_commit': source_commit, 'evidence_commit': evidence_commit,
            'baseline_archive_sha256': V5_HASH, 'tensor_aliases': len(aliases),
            'historical_checkpoint_payloads_included': True, 'upstream_pretrained_cache_included': False,
            'results': results, 'qualification_status': 'NOT_EVALUATED',
            'external_deployment_allowed': False,
            'verify_command': 'python tools/adaptive_capsule.py verify . PINNED_CONTROLLER_PUBLIC_KEY',
            'limits': 'Controller signature is not independent custody or qualification. The pinned official upstream model cache must be downloaded separately. Historical failures remain failures.'}
        generated.update({'PACKAGE_LAYOUT.json': canonical(layout),
            'SOURCE_IDENTITY.json': canonical({'schema': 'saal-published-source-identity-v1',
                'source_commit': source_commit, 'files': source_identity}),
            'TENSOR_PAYLOADS.json': canonical(mapping),
            'materialize_tensors.py': old.read('materialize_tensors.py')})
        for n, data in generated.items():
            manifest[n] = hashlib.sha256(data).hexdigest()
        generated['MANIFEST.sha256'] = ''.join(h+'  '+n+'\n' for n, h in sorted(manifest.items())).encode()
        key = Ed25519PrivateKey.from_private_bytes(Path(key_path).read_bytes())
        public = key.public_key().public_bytes_raw()
        signed = {'schema': 'saal-physical-delivery-v6',
            'manifest_sha256': hashlib.sha256(generated['MANIFEST.sha256']).hexdigest(),
            'source_commit': source_commit, 'evidence_commit': evidence_commit,
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False,
            'trust': 'Controller attribution only; independently trusted custody is not established.'}
        generated['DELIVERY_SIGNATURE.json'] = canonical({'payload': signed,
            'key_sha256': hashlib.sha256(public).hexdigest(),
            'signature_b64': base64.b64encode(key.sign(canonical(signed))).decode()})
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as z:
            for n in sorted(base_names-set(overrides)):
                with old.open(n) as source, z.open(n, 'w', force_zip64=True) as target:
                    shutil.copyfileobj(source, target, 1024*1024)
            for n, p in sorted(overrides.items()):
                z.write(p, n)
            for n, data in sorted(generated.items()):
                z.writestr(n, data)
    meta = {'archive': str(destination), 'sha256': digest(destination),
            'bytes': destination.stat().st_size, 'logical_files': len(manifest),
            'source_commit': source_commit, 'evidence_commit': evidence_commit,
            'controller_public_key': public.hex(), 'qualification_status': 'NOT_EVALUATED'}
    Path(str(destination)+'.sha256').write_text(meta['sha256']+'  '+destination.name+'\n')
    Path(str(destination)+'.metadata.json').write_bytes(canonical(meta))
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('pack')
    for arg in ('destination', 'baseline', 'key', 'source_ref', 'evidence_ref'):
        p.add_argument(arg)
    p = actions.add_parser('extract')
    for arg in ('archive', 'destination', 'sha256'):
        p.add_argument(arg)
    p = actions.add_parser('verify')
    p.add_argument('root'); p.add_argument('pin')
    args = parser.parse_args()
    if args.action == 'pack':
        value = package(args.destination, args.baseline, args.key, args.source_ref, args.evidence_ref)
    elif args.action == 'extract':
        value = extract(args.archive, args.destination, args.sha256)
    else:
        value = verify_files(args.root, args.pin)
    print(json.dumps(value, indent=2))
