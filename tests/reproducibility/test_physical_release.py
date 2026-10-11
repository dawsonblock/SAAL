import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('physical_release',Path(__file__).resolve().parents[2]/'tools/verify_reproducible_release.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_lfs_pointer_is_not_an_artifact(tmp_path):
 (tmp_path/'adapter.safetensors').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+'1'*64+'\nsize 4000000\n')
 with pytest.raises(ValueError,match='LFS pointer'):module.physical_file(tmp_path,'adapter.safetensors')

@pytest.mark.parametrize('name',['../outside','/tmp/outside','missing'])
def test_invalid_artifact_boundary(tmp_path,name):
 with pytest.raises(ValueError):module.physical_file(tmp_path,name)

def test_symlink_substitution(tmp_path):
 (tmp_path/'real').write_bytes(b'actual bytes');(tmp_path/'fake').symlink_to(tmp_path/'real')
 with pytest.raises(ValueError):module.physical_file(tmp_path,'fake')

def test_physical_file_and_hash(tmp_path):
 p=tmp_path/'real';p.write_bytes(b'actual bytes');assert module.physical_file(tmp_path,'real')==p
 assert len(module.file_hash(p))==64

def capsule(root):
 import hashlib,json
 from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 e=root/'reproducibility-evidence';e.mkdir();(root/'artifact').write_bytes(b'physical checkpoint')
 parent={'commit':'fixture','files':{'artifact':module.file_hash(root/'artifact')}};changes={}
 for name,data in [('parent-inventory.json',parent),('changes.json',changes)]: (e/name).write_bytes(module.canonical(data))
 inventory={n:module.file_hash(root/n) for n in module.included_files(root)};(e/'physical-inventory.json').write_bytes(module.canonical(inventory))
 body={'schema':'saal-physical-repair-release-v1','version':'0.9.4.3','qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'inventory_digest':hashlib.sha256(module.canonical(inventory)).hexdigest(),'parent_digest':hashlib.sha256(module.canonical(parent)).hexdigest(),'changes_digest':hashlib.sha256(module.canonical(changes)).hexdigest()}
 key=Ed25519PrivateKey.generate();(e/'signed-release.json').write_bytes(module.canonical({'payload':body,'signature':key.sign(module.canonical(body)).hex()}));return key.public_key().public_bytes_raw().hex()

def test_signed_physical_capsule(tmp_path):
 pin=capsule(tmp_path);assert module.verify(tmp_path,pin)['qualification_status']=='NOT_EVALUATED'

@pytest.mark.parametrize('mutation',['bytes','missing','extra','signature','inventory','history','approval'])
def test_release_substitution_is_rejected(tmp_path,mutation):
 import json
 from cryptography.exceptions import InvalidSignature
 pin=capsule(tmp_path);e=tmp_path/'reproducibility-evidence'
 if mutation=='bytes':(tmp_path/'artifact').write_bytes(b'substituted')
 elif mutation=='missing':(tmp_path/'artifact').unlink()
 elif mutation=='extra':(tmp_path/'unlisted').write_text('hidden data')
 elif mutation=='inventory':(e/'physical-inventory.json').write_text('{}')
 elif mutation=='history':(e/'parent-inventory.json').write_text('{}')
 else:
  statement=json.loads((e/'signed-release.json').read_text())
  if mutation=='signature':statement['signature']='0'*128
  else:statement['payload']['external_deployment_allowed']=True
  (e/'signed-release.json').write_bytes(module.canonical(statement))
 with pytest.raises((ValueError,InvalidSignature)):module.verify(tmp_path,pin)

def test_additive_results_scope_is_explicit(tmp_path):
 pin=capsule(tmp_path);(tmp_path/'results').mkdir();(tmp_path/'results/report').write_text('unverified additional data')
 with pytest.raises(ValueError):module.verify(tmp_path,pin)
 assert module.verify(tmp_path,pin,allow_results=True)['results_scope']=='EXCLUDED_VERIFY_SEPARATELY'
 (tmp_path/'artifact').write_bytes(b'changed main artifact')
 with pytest.raises(ValueError):module.verify(tmp_path,pin,allow_results=True)
