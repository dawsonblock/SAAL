"""Verify the controller release capsule against an explicit public-key pin."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.release_evidence import verify_release_evidence
p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--pin',required=True);a=p.parse_args();print(json.dumps(verify_release_evidence(a.root,a.pin),indent=2))
