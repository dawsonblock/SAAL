"""Complete physical release bytes, with deduplicated full checkpoint payloads."""
from pathlib import Path
import json,sys,zipfile,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];PROJECT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.spec import canonical
from saal_rsi.model_contract import sign
from saal_rsi.attempt_service import private_key


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def package(destination,key_file,baseline):
    files={};omitted=[]
    def add(p):
        name=p.relative_to(ROOT).as_posix()
        if '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.suffix=='.pyc':return
        if p.is_symlink() or p.suffix in ('.key','.ed25519') or 'private' in Path(name).parts:raise ValueError('private or symlink payload '+name)
        with p.open('rb') as f:
            if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('missing LFS bytes '+name)
        files[name]=p
    with zipfile.ZipFile(baseline) as z:
        for name in z.namelist():
            if name in ('MANIFEST.sha256','PACKAGE_LAYOUT.json','DELIVERY_SIGNATURE.json'):continue
            p=ROOT/name
            if not p.is_file():raise ValueError('baseline fixture missing '+name)
            add(p)
    frozen=json.loads((ROOT/'research/reliable-learning-v3/PREREGISTERED_FILES.json').read_text())
    for name,h in frozen.items():
        if digest(ROOT/name)!=h:raise ValueError('old source changed '+name)
        add(ROOT/name)
    for name,h in json.loads((PROJECT/'colab-tests-02/input-inventory.json').read_text()).items():
        if digest(ROOT/name)!=h:raise ValueError('current QA source changed '+name)
        add(ROOT/name)
    for folder in ('src','configs','docs','tests','reviewed-decision-corpus'):
        for p in (ROOT/folder).rglob('*'):
            if folder=='docs' and 'reviews' in p.parts and p.name not in ('review.json','metadata.json','peer-outcome.txt'):
                continue
            if p.is_file():add(p)
    for project in (ROOT/'research/reliable-learning-v3',PROJECT):
        for p in project.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(project)
            if any(x.startswith('packaging') for x in rel.parts) or 'stage' in rel.parts or p.name in ('test-inputs.zip','results.zip','worker.zip') or p.name.startswith('part-'):
                omitted.append(p.relative_to(ROOT).as_posix());continue
            add(p)
    payloads={};aliases={};manifest={}
    for name,p in files.items():
        h=digest(p);manifest[name]=h
        if p.suffix=='.safetensors' and p.stat().st_size>50_000_000:
            payload='tensor_payloads/'+h+'.safetensors'
            aliases[name]={'sha256':h,'size':p.stat().st_size,'payload':payload};payloads[payload]=p;manifest[payload]=h
    results=json.loads((PROJECT/'RESULTS.json').read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    layout={'schema':'saal-complete-physical-capsule-v4','source_commit':commit,
        'completed_models':len(results['runs']),'tensor_aliases':len(aliases),
        'complete_checkpoint_bytes_included':True,
        'restore_command':'python materialize_tensors.py .',
        'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,
        'human_reviews':0,'omitted_redundant_transport':omitted,
        'limits':'All current recovered tensor bytes are included; duplicate full checkpoint bytes are stored once and restored as hard links. The exact pinned upstream model cache is a separate prerequisite. Historical reports retain their original snapshot scope. No protected data or independent deployment is supplied.'}
    generated={'PACKAGE_LAYOUT.json':canonical(layout),'TENSOR_PAYLOADS.json':canonical({'schema':'saal-physical-tensor-payloads-v1','files':aliases}),
               'materialize_tensors.py':(PROJECT/'materialize_tensors.py').read_bytes()}
    for name,data in generated.items():manifest[name]=hashlib.sha256(data).hexdigest()
    generated['MANIFEST.sha256']=''.join(h+'  '+n+'\n' for n,h in sorted(manifest.items())).encode()
    key=private_key(key_file)
    generated['DELIVERY_SIGNATURE.json']=canonical(sign({'schema':'saal-physical-delivery-v4',
        'manifest_sha256':hashlib.sha256(generated['MANIFEST.sha256']).hexdigest(),
        'source_commit':commit,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,
        'trust':'Controller attribution and integrity; not independent qualification.'},key))
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,p in sorted(files.items()):
            if name not in aliases:z.write(p,name)
        for name,p in sorted(payloads.items()):z.write(p,name)
        for name,data in generated.items():z.writestr(name,data)
    meta={'archive':str(destination),'sha256':digest(destination),'bytes':destination.stat().st_size,
          'logical_files':len(manifest),'tensor_payloads':len(payloads),'qualification_status':'NOT_EVALUATED'}
    Path(str(destination)+'.sha256').write_text(meta['sha256']+'  '+destination.name+'\n')
    Path(str(destination)+'.metadata.json').write_bytes(canonical(meta));print(json.dumps(meta,indent=2))


if __name__=='__main__':package(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]))
