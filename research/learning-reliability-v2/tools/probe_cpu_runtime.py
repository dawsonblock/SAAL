"""Investigate CPU kernel/thread effects without relaxing registered tolerance."""
import os
os.environ['VECLIB_MAXIMUM_THREADS']='1'
from pathlib import Path
import json,sys,time
from saal_rsi.model_contract import verify
from saal_rsi.learning_backend import create_backend,verify_dependencies
from saal_rsi.pretrained_artifacts import validate_bundle
from saal_rsi.pretrained_registry import validate_snapshot
from saal_rsi.sequence_replay_v2 import prediction_equal
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    index=int(sys.argv[1]);root=ROOT/'experiment';pin=(root/'controller.pub').read_text().strip()
    run=json.loads((root/'colab/outcomes.json').read_text())['runs'][index];bundle=root/'artifacts'/run['run']
    job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin);result=validate_bundle(bundle,job,pin);spec=job['model_spec']
    cache=Path(sys.argv[2])/spec['model_repository'].split('/')[-1]/spec['model_revision'];validate_snapshot(cache,spec['model_repository'],spec['model_revision'])
    backend=create_backend(spec,cache,job['seed']).load_pretrained();backend.restore_checkpoint(bundle/'terminal.safetensors')
    records=[r for r in job['dataset']['records'] if r['split']=='development'];started=time.monotonic();actual=backend.evaluate(records);expected=result['snapshots']['terminal']['predictions']['development']
    delta=max(abs(a-b) for x,y in zip(actual,expected) for a,b in zip(x['probabilities'],y['probabilities']))
    report={'schema':'saal-public-cpu-runtime-probe-v1','run':run['run'],'checkpoint':'terminal','split':'development','threads':1,'interop_threads':1,'VECLIB_MAXIMUM_THREADS':'1','float32_matmul_precision':torch.get_float32_matmul_precision(),'dependencies':verify_dependencies(),'probability_tolerance':1e-4,'agreement':prediction_equal(actual,expected,spec['model_kind'],1e-4),'max_probability_difference':delta,'decision_differences':sum(x['decision']!=y['decision'] for x,y in zip(actual,expected)),'predictions':actual,'elapsed_seconds':time.monotonic()-started,'qualification_status':'NOT_EVALUATED','scope':'Posthoc CPU numerical investigation. Original failing replay remains preserved. This probe is not replacement training, threshold relaxation or independent qualification.'}
    with (ROOT/(run['run']+'-cpu-runtime-probe.json')).open('x') as f:json.dump(report,f,sort_keys=True,indent=2)
    print({k:v for k,v in report.items() if k not in ('predictions','dependencies')})
