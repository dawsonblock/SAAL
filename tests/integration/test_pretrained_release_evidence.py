import hashlib,pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import sign
from saal_rsi.release_evidence import verify_release_evidence

def make(tmp_path):
    e=tmp_path/'release-evidence';e.mkdir();s=tmp_path/'source.py';s.write_text('source');r=tmp_path/'report.json';r.write_text('{}')
    inventory={'schema':'saal-source-inventory-v1','files':{'source.py':hashlib.sha256(s.read_bytes()).hexdigest()}};(e/'source-inventory.json').write_bytes(canonical(inventory))
    p={'schema':'saal-controller-release-evidence-v1','version':'0.9.4','signer_role':'CONTROLLER_RESEARCH_ATTRIBUTION','source_inventory_digest':sha(inventory),'evidence_files':{'report.json':hashlib.sha256(r.read_bytes()).hexdigest()},'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'limits':'Integrity only, no independent qualification.'};key=Ed25519PrivateKey.generate();(e/'signed-statement.json').write_bytes(canonical(sign(p,key)));return p,key

def test_release_signature_is_integrity_only(tmp_path):
    p,k=make(tmp_path);r=verify_release_evidence(tmp_path,k.public_key().public_bytes_raw().hex());assert r['qualification_status']=='NOT_EVALUATED' and r['external_deployment_allowed']is False

@pytest.mark.parametrize('file',['source.py','report.json'])
def test_release_substitution_rejected(tmp_path,file):
    p,k=make(tmp_path);(tmp_path/file).write_text('changed')
    with pytest.raises(ValueError):verify_release_evidence(tmp_path,k.public_key().public_bytes_raw().hex())

def test_controller_signature_cannot_expand_release_authority(tmp_path):
    p,k=make(tmp_path);p['external_deployment_allowed']=True;(tmp_path/'release-evidence/signed-statement.json').write_bytes(canonical(sign(p,k)))
    with pytest.raises(ValueError):verify_release_evidence(tmp_path,k.public_key().public_bytes_raw().hex())
