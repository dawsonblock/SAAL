"""Verify shipped public campaign signatures, tensors, reload and historical freeze."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.model_contract import verify
from saal_rsi.neural_artifacts import validate_neural_bundle
from saal_rsi.neural_campaign import campaign_summary
from saal_rsi.spec import canonical,parse_unique_json
root=Path(__file__).resolve().parents[1];example=root/'examples/neural_campaign'
pin=json.loads((example/'public-keys.json').read_text())['controller']
protocol=verify(parse_unique_json((example/'protocol.json').read_bytes()),pin)
freeze=json.loads((root/'provenance/v0.9.3-historical-freeze.json').read_text())
for relative,digest in freeze['protected_files'].items():
    if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:raise ValueError('historical preservation '+relative)
runs=json.loads((example/'run-outcomes.json').read_text());resolved=[]
for run in runs:
    record=dict(run)
    if run['status']=='COMPLETED':
        for field in ('bundle','job'):record[field]=str(example/run[field])
    resolved.append(record)
report=campaign_summary(resolved,pin)
shipped=json.loads((example/'report.json').read_text())
if canonical(report)!=canonical(shipped):raise ValueError('campaign reconstruction mismatch')
print(json.dumps({'historical_files_preserved':len(freeze['protected_files']),'reconstructed_runs':len(report['runs']),
    'independent_generalization':report['independent_generalization'],'qualification_status':report['qualification_status'],'external_deployment_allowed':False}))
