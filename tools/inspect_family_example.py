"""Verify the shipped public diagnostic family evidence, never grant activation."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.model_contract import verify
from saal_rsi.model_backend import validate_bundle
from saal_rsi.model_evaluation import verify_model_report
root=Path(__file__).resolve().parents[1]/'examples/family_protocol'
def read(name):return json.loads((root/name).read_text())
controller=(root/'controller-public-key.txt').read_text().strip()
custodian=(root/'custodian-public-key.txt').read_text().strip()
evaluator=(root/'evaluator-public-key.txt').read_text().strip()
payload=verify(read('job.json'),controller);validate_bundle(root/'checkpoint',payload)
report=verify_model_report(read('qualification.json'),evaluator,custodian)
assert report['metrics']['research_qualified'] is False
assert report['external_deployment_allowed'] is False
print(json.dumps({'signatures_and_bindings_valid':True,'research_qualified':False,'external_deployment_allowed':False,'sampling':'Public synthetic diagnostic fixtures; no independent sampling claim'}))
