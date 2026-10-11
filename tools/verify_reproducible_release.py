"""Verify a hydrated physical release; controller signature is not qualification."""
from pathlib import Path
import argparse,hashlib,json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def file_hash(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def physical_file(root,name):
 if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('file boundary')
 p=root/name
 if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):raise ValueError('missing/substituted file: '+name)
 with p.open('rb') as f:prefix=f.read(200)
 if prefix.startswith(b'version https://git-lfs.github.com/spec/v1\n'):raise ValueError('unhydrated LFS pointer: '+name)
 return p

def included_files(root):
 return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix not in ('.pyc','.pyo') and p.relative_to(root).as_posix() not in ('.git','reproducibility-evidence/physical-inventory.json','reproducibility-evidence/signed-release.json','reproducibility-evidence/controller.pub')}

def verify(root,pin,allow_results=False):
 root=Path(root).resolve();e=root/'reproducibility-evidence'
 statement=json.loads((e/'signed-release.json').read_text());body=statement['payload']
 Ed25519PublicKey.from_public_bytes(bytes.fromhex(pin)).verify(bytes.fromhex(statement['signature']),canonical(body))
 if body['schema']!='saal-physical-repair-release-v1' or body['version']!='0.9.4.3' or body['qualification_status']!='NOT_EVALUATED' or body['external_deployment_allowed'] is not False:raise ValueError('scope')
 inventory=json.loads((e/'physical-inventory.json').read_text())
 if hashlib.sha256(canonical(inventory)).hexdigest()!=body['inventory_digest']:raise ValueError('inventory substitution')
 if not isinstance(inventory,dict) or not inventory:raise ValueError('empty inventory')
 actual=included_files(root)
 if allow_results:actual={n for n in actual if not n.startswith('results/')}
 if set(inventory)!=actual:raise ValueError('missing or unlisted physical file')
 for name,sha in inventory.items():
  if file_hash(physical_file(root,name))!=sha:raise ValueError('physical hash mismatch: '+name)
 parent=json.loads((e/'parent-inventory.json').read_text());changes=json.loads((e/'changes.json').read_text())
 if hashlib.sha256(canonical(parent)).hexdigest()!=body['parent_digest'] or hashlib.sha256(canonical(changes)).hexdigest()!=body['changes_digest']:raise ValueError('history substitution')
 changed={}
 for name,before in parent['files'].items():
  after=file_hash(physical_file(root,name))
  if before!=after:changed[name]={'before':before,'after':after}
 if changed!=changes:raise ValueError('unaccounted parent drift')
 return {'version':body['version'],'physical_files_verified':len(inventory),'parent_files':len(parent['files']),'accounted_changes':sorted(changed),'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'signer_identity':'REQUIRES_INDEPENDENT_PIN_AUTHENTICATION','results_scope':'EXCLUDED_VERIFY_SEPARATELY' if allow_results else 'NO_ADDITIONAL_FILES'}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--pin',required=True);parser.add_argument('--allow-results',action='store_true',help='Verify the unchanged main subset on an additive results branch; results require separate verification');args=parser.parse_args();print(json.dumps(verify(args.root,args.pin,args.allow_results),indent=2))
