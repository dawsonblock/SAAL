"""Local controller for a named-session campaign; its private key is never uploaded."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.assurance import load_signing_key
from saal_rsi.spec import parse_unique_json
from saal_rsi.neural_campaign import run_registered_jobs
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--pin',required=True);p.add_argument('--controller-key',required=True);p.add_argument('--session',required=True);a=p.parse_args()
root=Path(a.workspace);key=load_signing_key(Path(a.controller_key))
run_registered_jobs(parse_unique_json((root/'protocol.json').read_bytes()),a.pin,key,parse_unique_json((root/'dataset.json').read_bytes()),root,remote_session=a.session)
