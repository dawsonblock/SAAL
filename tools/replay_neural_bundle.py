"""Bounded CPU reconstruction from an admitted signed v4 job."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.model_contract import verify
from saal_rsi.neural_artifacts import validate_neural_bundle
from saal_rsi.spec import parse_unique_json
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--pin',required=True);p.add_argument('--bundle',required=True);a=p.parse_args()
envelope=parse_unique_json(Path(a.job).read_bytes());job=verify(envelope,a.pin)
validate_neural_bundle(a.bundle,job,replay=True,controller_pin=a.pin)
print(json.dumps({'checkpoint_replay':'PASSED','qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}))
