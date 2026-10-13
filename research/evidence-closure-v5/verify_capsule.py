"""Authenticate all physical release bytes before restoring checkpoint paths."""
from pathlib import Path
import sys,json,hashlib,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.model_contract import verify
from saal_rsi.spec import parse_unique_json


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_files(root,pin):
    root=Path(root).resolve();entries={}
    raw=(root/'MANIFEST.sha256').read_bytes()
    envelope=parse_unique_json((root/'DELIVERY_SIGNATURE.json').read_bytes())
    signature=verify(envelope,pin)
    if signature.get('schema')!='saal-physical-delivery-v5' or signature.get('manifest_sha256')!=hashlib.sha256(raw).hexdigest() or signature.get('qualification_status')!='NOT_EVALUATED' or signature.get('external_deployment_allowed') is not False:
        raise ValueError('delivery signature or scope mismatch')
    for line in raw.decode().splitlines():
        h,name=line.split('  ',1)
        if not re.fullmatch('[0-9a-f]{64}',h) or name in entries or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('manifest inventory or path')
        entries[name]=h
    for name in ('PACKAGE_LAYOUT.json','SOURCE_IDENTITY.json'):
        if digest(root/name)!=entries.get(name):raise ValueError('untrusted source/layout identity')
    layout=parse_unique_json((root/'PACKAGE_LAYOUT.json').read_bytes())
    source=parse_unique_json((root/'SOURCE_IDENTITY.json').read_bytes())
    if (layout.get('schema')!='saal-complete-physical-capsule-v5'
            or source.get('schema')!='saal-published-source-identity-v1'
            or source.get('source_commit')!=signature.get('source_commit')
            or layout.get('source_commit')!=signature.get('source_commit')
            or layout.get('evidence_commit')!=signature.get('evidence_commit')
            or not source.get('files') or any(entries.get(n)!=h for n,h in source['files'].items())):
        raise ValueError('release source/evidence identity mismatch')
    alias_file=root/'TENSOR_PAYLOADS.json'
    if digest(alias_file)!=entries.get('TENSOR_PAYLOADS.json'):raise ValueError('untrusted restoration map')
    aliases=json.loads(alias_file.read_text())['files']
    for name,h in entries.items():
        p=root/name
        if not p.exists():
            if name not in aliases:raise ValueError('missing physical release file '+name)
            continue
        if p.is_symlink() or not p.is_file() or digest(p)!=h:raise ValueError('physical file substitution '+name)
        with p.open('rb') as f:
            if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('LFS pointer instead of bytes '+name)
    # Verify the restoration program itself before importing it.
    program=root/'materialize_tensors.py'
    if digest(program)!=entries.get('materialize_tensors.py'):raise ValueError('restoration source substitution')
    import importlib.util
    spec=importlib.util.spec_from_file_location('capsule_materializer',program)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    restored=module.materialize(root)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix!='.pyc'}-{'MANIFEST.sha256','DELIVERY_SIGNATURE.json'}
    if actual!=set(entries):raise ValueError('extra or missing release files after restoration')
    for name,h in entries.items():
        if digest(root/name)!=h:raise ValueError('restored file mismatch '+name)
    frozen=json.loads((root/'research/reliable-learning-v3/PREREGISTERED_FILES.json').read_text())
    for name,h in frozen.items():
        if digest(root/name)!=h:raise ValueError('historical source modified '+name)
    return {'physical_and_restored_files':len(entries),'checkpoint_paths_restored':restored,
            'historical_frozen_files':len(frozen),'qualification_status':'NOT_EVALUATED',
            'trust':'Authenticated controller release; no independent qualification inferred.'}


if __name__=='__main__':print(json.dumps(verify_files(sys.argv[1],sys.argv[2]),indent=2))
