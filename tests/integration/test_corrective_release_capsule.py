import hashlib
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.corrective_release import verify_capsule
from saal_rsi.model_contract import sign
from saal_rsi.spec import sha,canonical


def make(root):
    e=root/'corrective-evidence';e.mkdir();(root/'provenance').mkdir();(root/'src').mkdir();source=root/'src/source.py';source.write_text('safe = True\n');report=root/'report.json';report.write_text('{"qualified":false}')
    freeze={'archive_sha256':'frozen-history'};(root/'provenance/v0.9.4.1-historical-input.json').write_bytes(canonical(freeze))
    inventory={'schema':'saal-source-inventory-v1','files':{'src/source.py':hashlib.sha256(source.read_bytes()).hexdigest()}};(e/'source-inventory.json').write_bytes(canonical(inventory))
    body={'schema':'saal-corrective-controller-evidence-v1','version':'0.9.4.1','role':'CONTROLLER_RESEARCH_ATTRIBUTION','source_inventory_digest':sha(inventory),'historical_freeze_digest':sha(freeze),'evidence_files':{'report.json':hashlib.sha256(report.read_bytes()).hexdigest()},'qualification_status':'NOT_EVALUATED','independent_deployment_verified':False,'external_deployment_allowed':False,'limits':'Controller evidence only'}
    key=Ed25519PrivateKey.generate();(e/'signed-statement.json').write_bytes(canonical(sign(body,key)))
    return body,key


def test_capsule_attribution_only(tmp_path):
    body,key=make(tmp_path);result=verify_capsule(tmp_path,key.public_key().public_bytes_raw().hex())
    assert result['source_files_verified']==1 and result['qualification_status']=='NOT_EVALUATED'


@pytest.mark.parametrize('mutation',['source','extra_source','report','independence','signature'])
def test_capsule_scope_and_substitution(tmp_path,mutation):
    body,key=make(tmp_path)
    if mutation=='source':(tmp_path/'src/source.py').write_text('unsafe = True')
    elif mutation=='extra_source':(tmp_path/'src/injected.py').write_text('unexpected = True')
    elif mutation=='report':(tmp_path/'report.json').write_text('{"qualified":true}')
    elif mutation=='independence':body['independent_deployment_verified']=True;(tmp_path/'corrective-evidence/signed-statement.json').write_bytes(canonical(sign(body,key)))
    elif mutation=='signature':(tmp_path/'corrective-evidence/signed-statement.json').write_bytes(canonical(sign(body,Ed25519PrivateKey.generate())))
    with pytest.raises(ValueError):verify_capsule(tmp_path,key.public_key().public_bytes_raw().hex())
