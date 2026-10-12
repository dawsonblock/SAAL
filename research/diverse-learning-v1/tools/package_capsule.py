"""Freeze a complete learning capsule, excluding redundant transfer copies."""
from pathlib import Path
import hashlib,json,sys,zipfile,stat
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.model_contract import sign
from saal_rsi.spec import canonical
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def package(destination,key_file):
 paths=[];omitted=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(ROOT);name=rel.as_posix()
  if '__pycache__' in rel.parts or '.pytest_cache' in rel.parts or p.suffix=='.pyc':continue
  if name in ['MANIFEST.sha256','DELIVERY_SIGNATURE.json','PACKAGE_LAYOUT.json']:continue
  if name.startswith('experiment/colab/'):
   tail=rel.parts[2:]
   keep=(len(tail)==1 and (p.suffix in ['.json','.log','.py','.txt','.pub'] or p.name=='worker.zip')) or (len(tail)==2 and p.name=='manifest.json')
   if not keep:omitted.append(name);continue
  if p.is_symlink():raise ValueError('package symlink '+name)
  paths.append(p)
 layout={'schema':'saal-public-capsule-layout-v1','included':'All source/configs, data, signed jobs, initial/best/terminal tensors for every run, metrics/predictions, logs, CPU replay and cleanup. Upstream pretrained snapshot is separately fetched using pinned authenticated file/tensor manifests.','omitted_redundant_transfer_files':omitted,'private_keys_included':False,'historical_release':'Separate complete v0.9.4.3 assets; unchanged source subset authenticated by SOURCE_ORIGIN.json.'}
 (ROOT/'PACKAGE_LAYOUT.json').write_text(json.dumps(layout,sort_keys=True,indent=2));paths.append(ROOT/'PACKAGE_LAYOUT.json')
 paths.sort();(ROOT/'MANIFEST.sha256').write_text(''.join(digest(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in paths))
 key=Ed25519PrivateKey.from_private_bytes(Path(key_file).read_bytes());pin=(ROOT/'experiment/controller.pub').read_text().strip()
 if key.public_key().public_bytes_raw().hex()!=pin:raise ValueError('delivery signing identity')
 payload={'schema':'saal-public-delivery-v1','manifest_sha256':digest(ROOT/'MANIFEST.sha256'),'preregistration_commit':'812a67fe2af03163ffc280abb6a34bb4fbf6357e','qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'trust':'Controller attribution and package self-consistency, not independent qualification.'}
 (ROOT/'DELIVERY_SIGNATURE.json').write_bytes(canonical(sign(payload,key)))
 paths.extend([ROOT/'MANIFEST.sha256',ROOT/'DELIVERY_SIGNATURE.json'])
 with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for p in paths:z.write(p,p.relative_to(ROOT).as_posix())
 info={'archive':str(destination),'files':len(paths),'size':Path(destination).stat().st_size,'sha256':digest(Path(destination)),'qualification_status':'NOT_EVALUATED'}
 Path(str(destination)+'.sha256').write_text(info['sha256']+'  '+Path(destination).name+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':package(Path(sys.argv[1]),sys.argv[2])
