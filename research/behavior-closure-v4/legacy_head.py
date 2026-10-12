"""Execute the previously signed, unrun v3 head control with verified transfer."""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.model_contract import verify
from saal_rsi.spec import sha
from saal_rsi.epoch_contract import worker_identity
from saal_rsi.epoch_backend_v2 import execute_campaign


def run():
    parent=ROOT/'research/reliable-learning-v3/head-control/experiment'
    pin=(parent/'controller.pub').read_text().strip()
    protocol=verify(json.loads((parent/'signed_protocol.json').read_text()),pin)
    envelope=json.loads((parent/'signed_job.json').read_text())
    for name,digest in protocol['source_files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('old control source changed')
    if sha(verify(envelope,pin))!=protocol['job_digest'] or worker_identity()!=protocol['worker_identity']:
        raise ValueError('old signed head control identity')
    if (parent/'development-ledger.sqlite3').exists():
        raise ValueError('old control already has an execution ledger; no implicit restart')
    out=Path(__file__).parent/'legacy-head'
    out.mkdir(exist_ok=False)
    return execute_campaign(out/'colab',[envelope],pin,out/'artifacts',out/'budget.sqlite3',max_seconds=5000)


if __name__=='__main__':run()
