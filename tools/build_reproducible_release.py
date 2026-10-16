"""Freeze signed physical bytes and build a self-contained ZIP outside the checkout."""
from pathlib import Path
import argparse,hashlib,json,tomllib,zipfile
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from verify_reproducible_release import canonical,file_hash,physical_file,included_files,verify

LEGACY_VERSION='0.9.4.3'

def require_legacy_release(root):
 project=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))
 version=project.get('project',{}).get('version')
 if version!=LEGACY_VERSION:
  raise ValueError(f'legacy v1 release builder only supports {LEGACY_VERSION}; current package is {version!r}')

def freeze(root,key_path):
 root=root.resolve()
 require_legacy_release(root)
 if key_path.resolve().is_relative_to(root):raise ValueError('Signing key must remain outside checkout')
 e=root/'reproducibility-evidence';key=Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
 excluded={'MANIFEST.sha256','reproducibility-evidence/changes.json'}
 names=sorted(included_files(root)-excluded)
 (root/'MANIFEST.sha256').write_text(''.join(file_hash(physical_file(root,n))+'  '+n+'\n' for n in names))
 parent=json.loads((e/'parent-inventory.json').read_text());changes={}
 for name,before in parent['files'].items():
  after=file_hash(physical_file(root,name))
  if before!=after:changes[name]={'before':before,'after':after}
 (e/'changes.json').write_bytes(canonical(changes))
 inventory={n:file_hash(physical_file(root,n)) for n in sorted(included_files(root))}
 (e/'physical-inventory.json').write_bytes(canonical(inventory))
 body={'schema':'saal-physical-repair-release-v1','version':'0.9.4.3','qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'inventory_digest':hashlib.sha256(canonical(inventory)).hexdigest(),'parent_digest':hashlib.sha256(canonical(parent)).hexdigest(),'changes_digest':hashlib.sha256(canonical(changes)).hexdigest(),'limits':'Controller integrity attribution only. Bundled public key has no independently established identity; no empirical qualification.'}
 (e/'signed-release.json').write_bytes(canonical({'payload':body,'signature':key.sign(canonical(body)).hex()}));pin=key.public_key().public_bytes_raw().hex();(e/'controller.pub').write_text(pin+'\n');return verify(root,pin)
def archive(root,output,pin):
 root=root.resolve();output=output.resolve()
 require_legacy_release(root)
 if output.is_relative_to(root):raise ValueError('Archive must be outside checkout')
 result=verify(root,pin)
 names=sorted(included_files(root)|{'reproducibility-evidence/physical-inventory.json','reproducibility-evidence/signed-release.json','reproducibility-evidence/controller.pub'})
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
  for name in names:z.write(physical_file(root,name),root.name+'/'+name)
 with zipfile.ZipFile(output) as z:assert z.testzip() is None
 output.with_suffix('.sha256').write_text(file_hash(output)+'  '+output.name+'\n')
 return result|{'archive':str(output),'sha256':file_hash(output),'archive_files':len(names),'bytes':output.stat().st_size}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--private-key',type=Path);p.add_argument('--output',type=Path);p.add_argument('--pin');a=p.parse_args()
 if a.private_key:print(json.dumps(freeze(a.root,a.private_key),indent=2))
 if a.output:
  pin=a.pin or (a.root/'reproducibility-evidence/controller.pub').read_text().strip();print(json.dumps(archive(a.root,a.output,pin),indent=2))
