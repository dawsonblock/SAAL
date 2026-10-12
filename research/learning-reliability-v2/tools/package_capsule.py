"""Package this correction and all nine candidate tensors as physical bytes.

Historical fixture trees/pretrained weights remain separately pinned prerequisites.
No model cache, private keys, duplicate transfer archive or Git directory is shipped.
"""
from pathlib import Path
import hashlib,json,sys,zipfile
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.model_contract import sign
from saal_rsi.spec import canonical

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def package(destination,key_file):
    files={};omitted=[]
    def add(path):
        name=path.relative_to(REPO).as_posix()
        if path.is_symlink():raise ValueError('package symlink '+name)
        if path.suffix in ('.key','.ed25519') or 'private' in path.parts:raise ValueError('private material '+name)
        if '__pycache__' in path.parts or '.pytest_cache' in path.parts or path.suffix=='.pyc':return
        with path.open('rb') as f:
            if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('unhydrated payload '+name)
        files[name]=path
    frozen=json.loads((ROOT/'PREREGISTERED_FILES.json').read_text())
    for name,h in frozen.items():
        path=REPO/name
        if digest(path)!=h:raise ValueError('preregistered source drift '+name)
        add(path)
    for directory in ('src','configs','docs','tests','reviewed-decision-corpus'):
        for p in sorted((REPO/directory).rglob('*')):
            if p.is_file():add(p)
    for name in ('pyproject.toml','README.md','LICENSE','CHANGELOG.md','.gitattributes'):
        if (REPO/name).is_file():add(REPO/name)
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        relative=p.relative_to(ROOT)
        if 'packaging' in relative.parts:continue
        if p.name.startswith('package-build') and p.suffix=='.log':continue
        if relative.parts[:2]==('experiment','colab'):
            tail=relative.parts[2:]
            keep=(len(tail)==1 and (p.suffix in ('.json','.jsonl','.log','.py','.txt','.pub') or p.name=='worker.zip')) or (len(tail)==2 and p.name=='manifest.json')
            if not keep:omitted.append(p.relative_to(REPO).as_posix());continue
        add(p)
    prior=REPO/'research/diverse-learning-v1/experiment'
    for name in ('diverse_dataset.json','old_control_dataset.json','controller.pub'):add(prior/name)
    # Small authenticated historical LoRA fixture needed by the new real step-zero test.
    for p in sorted((prior/'artifacts/run-03-6616792a4767').iterdir()):
        if p.is_file():add(p)
    layout={'schema':'saal-public-reliability-capsule-v2','source_commit':json.loads((ROOT/'PREREGISTRATION_ANCHOR.json').read_text())['commit'],
        'included':'All source packages, configs/docs/tests, unchanged review component, frozen corpus/jobs/protocol, all three candidate initial/best/terminal tensors, raw results/errors, monitoring/cleanup/CI/CPU reproduction evidence and the historical LoRA step-zero fixture.',
        'excluded_prerequisites':{'upstream_pretrained_weights':'Official pinned ModernBERT snapshot fetched through authenticated registry manifests; not bundled.',
          'full_historical_fixtures':'Entire core test suite requires the complete repository LFS/example/history fixtures. Clean capsule tests run the 59 new integration checks, four reporting checks and 38 existing review checks. Complete core checks already ran locally and on fresh Linux CI.',
          'retention_parent_full_tensors':'v1 complete experiment archive or hydrated results/diverse-learning-v1 commit f05876e43408f4300fc30b923ca0e2852feb44b3. RESULTS.json contains exact paired effects; re-running native parent validation needs those preserved full-tuning payloads.'},
        'omitted_redundant_transfer_files':omitted,'private_keys_included':False,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    generated={'PACKAGE_LAYOUT.json':json.dumps(layout,sort_keys=True,indent=2).encode()}
    manifest=''.join(digest(p)+'  '+n+'\n' for n,p in sorted(files.items()))
    manifest+=hashlib.sha256(generated['PACKAGE_LAYOUT.json']).hexdigest()+'  PACKAGE_LAYOUT.json\n'
    generated['MANIFEST.sha256']=manifest.encode()
    pin=(ROOT/'experiment/controller.pub').read_text().strip();key=Ed25519PrivateKey.from_private_bytes(Path(key_file).read_bytes())
    if key.public_key().public_bytes_raw().hex()!=pin:raise ValueError('delivery signing identity')
    payload={'schema':'saal-public-delivery-v2','manifest_sha256':hashlib.sha256(generated['MANIFEST.sha256']).hexdigest(),'preregistration_commit':layout['source_commit'],'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'trust':'Controller attribution, not independent custody or evaluation.'}
    generated['DELIVERY_SIGNATURE.json']=canonical(sign(payload,key))
    metadata=ROOT/'packaging';metadata.mkdir(exist_ok=False)
    for name,value in generated.items():(metadata/name).write_bytes(value)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,p in sorted(files.items()):z.write(p,name)
        for name,value in generated.items():z.writestr(name,value)
    info={'archive':str(destination),'physical_files':len(files)+len(generated),'archive_bytes':Path(destination).stat().st_size,'archive_sha256':digest(Path(destination)),'qualification_status':'NOT_EVALUATED','preregistered_files':len(frozen)}
    (metadata/'ARCHIVE.json').write_text(json.dumps(info,indent=2))
    Path(str(destination)+'.sha256').write_text(info['archive_sha256']+'  '+Path(destination).name+'\n')
    print(json.dumps(info,indent=2));return info
if __name__=='__main__':package(Path(sys.argv[1]),sys.argv[2])
