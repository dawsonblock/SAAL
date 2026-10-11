"""Verify the controller capsule with an explicitly pinned public key."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.corrective_release import verify_capsule
p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--pin',required=True);a=p.parse_args()
print(json.dumps(verify_capsule(a.root,a.pin)))
