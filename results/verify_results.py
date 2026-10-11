"""Verify main plus research-results integrity; grants no qualification."""
from pathlib import Path
import argparse,sys,json,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'tools'))
from verify_reproducible_release import verify,physical_file,file_hash,canonical
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

def check(pin):
 main=verify(root,pin,allow_results=True);r=root/'results';statement=json.loads((r/'SIGNED_PUBLICATION.json').read_text());body=statement['payload'];Ed25519PublicKey.from_public_bytes(bytes.fromhex(pin)).verify(bytes.fromhex(statement['signature']),canonical(body))
 if body['schema']!='saal-research-results-publication-v1' or body['qualification_status']!='NOT_EVALUATED' or body['external_deployment_allowed'] is not False:raise ValueError('results authority scope')
 if file_hash(root/'reproducibility-evidence/signed-release.json')!=body['main_statement_sha256'] or file_hash(r/'PUBLICATION_MANIFEST.sha256')!=body['manifest_sha256']:raise ValueError('results identity substitution')
 inventory={}
 for line in (r/'PUBLICATION_MANIFEST.sha256').read_text().splitlines():
  sha,name=line.split('  ',1)
  if name in inventory:raise ValueError('duplicate result path')
  inventory[name]=sha
  if file_hash(physical_file(root,'results/'+name))!=sha:raise ValueError('result bytes changed')
 actual={p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix not in ('.pyc','.pyo') and p.name not in ('PUBLICATION_MANIFEST.sha256','SIGNED_PUBLICATION.json')}
 # Nested historical publication manifests remain ordinary evidence files.
 actual={p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix not in ('.pyc','.pyo') and p.relative_to(r).as_posix() not in ('PUBLICATION_MANIFEST.sha256','SIGNED_PUBLICATION.json')}
 if set(inventory)!=actual:raise ValueError('missing/unlisted results')
 return {'main':main,'results_files_verified':len(inventory),'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pin',required=True);a=p.parse_args();print(json.dumps(check(a.pin),indent=2))
