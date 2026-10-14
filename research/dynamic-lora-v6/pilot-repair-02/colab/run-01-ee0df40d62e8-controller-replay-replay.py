import sys,json
from pathlib import Path
sys.path.insert(0,'/Users/dawsonblock/Downloads/SAAL-diverse-publish/src')
from saal_rsi.adaptive_campaign import fresh_local_replay
from saal_rsi.spec import canonical
from saal_rsi.model_contract import verify
pin='e3db435cf14ebd609a7950c039a15809001bf00b23fa7deb363a7f1bffe0df4b'
job=verify(json.loads(Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/dynamic-lora-v6/pilot-repair-02/fixed_r12_60606-job.json').read_text()),pin)
value=fresh_local_replay('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/dynamic-lora-v6/pilot-repair-02/artifacts/run-01-ee0df40d62e8',job,pin,'/Users/dawsonblock/Downloads/SAAL-RSI-model-cache')
with Path('/Users/dawsonblock/Downloads/SAAL-diverse-publish/research/dynamic-lora-v6/pilot-repair-02/colab/run-01-ee0df40d62e8-controller-replay.json').open("xb") as output:output.write(canonical(value))
if not value["verified"]:raise ValueError("CPU/GPU numerical replay failed at unchanged tolerance")
