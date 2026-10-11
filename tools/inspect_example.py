"""Portable cryptographic inspection; never certifies the inspecting host or activation."""
from pathlib import Path
import base64
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from saal_rsi.spec import canonical,sha,parse_unique_json
from saal_rsi.identity import fingerprint,runtime_identity
from saal_rsi.registry import verify_receipt

root=Path(__file__).resolve().parents[1]
envelope=parse_unique_json((root/'examples/example_negative_qualification.json').read_bytes())
eval_hex=(root/'examples/example_evaluator_public_key.txt').read_text().strip()
anchor_hex=(root/'examples/example_custodian_public_key.txt').read_text().strip()
pub=bytes.fromhex(eval_hex)
assert envelope['public_key_sha256']==hashlib.sha256(pub).hexdigest()
Ed25519PublicKey.from_public_bytes(pub).verify(base64.b64decode(envelope['signature_b64'],validate=True),canonical(envelope['report']))
report=envelope['report'];verify_receipt(report['anchor_receipt'],anchor_hex)
assert report['implementation_digest']==fingerprint()
proposal=parse_unique_json((root/'examples/public_development_proposal.json').read_bytes())
assert report['proposal_digest']==proposal['proposal_digest']
assert report['qualification_passed'] is False and report['promotion_allowed'] is False
print(json.dumps({'signatures_valid':True,'source_identity_matches':True,
    'inspecting_runtime_matches':runtime_identity()==report['execution_identity'],
    'qualification_passed':False,'external_activation_allowed':False}))
