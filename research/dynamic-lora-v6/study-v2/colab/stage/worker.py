from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import argparse
from saal_rsi.pretrained_contract import job_bytes
from saal_rsi.spec import parse_unique_json
parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--job');known,_=parser.parse_known_args()
schema=parse_unique_json(job_bytes(known.job))['payload']['schema'] if known.job else 'saal-adaptive-job-v1'
if schema=='saal-adaptive-job-v2':
 from saal_rsi.adaptive_worker_v2 import main
elif schema=='saal-adaptive-job-v1':
 from saal_rsi.adaptive_worker import main
else:raise ValueError('unknown signed worker version')
main()
