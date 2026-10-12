"""Controller CPU replay: public predictions, no independent qualification."""
from pathlib import Path
import json,time,sys,hashlib
from saal_rsi.model_contract import verify
from saal_rsi.learning_backend import create_backend,verify_dependencies
from saal_rsi.pretrained_artifacts import validate_bundle
from saal_rsi.pretrained_registry import validate_snapshot
from saal_rsi.sequence_replay_v2 import prediction_equal
ROOT=Path(__file__).resolve().parents[1]

def replay(index,cache_root,output=None):
 import torch
 torch.set_num_threads(6)
 started=time.monotonic();deadline=started+1800
 pin=(ROOT/'experiment/controller.pub').read_text().strip()
 outcomes=json.loads((ROOT/'experiment/colab/outcomes.json').read_text())['runs']
 run=outcomes[index]
 if run['state']!='COMPLETED':raise ValueError('incomplete training')
 out=Path(output) if output else ROOT/'experiment/controller-replay';out.mkdir(exist_ok=True,parents=True)
 destination=out/(run['run']+'.json')
 if destination.exists():raise FileExistsError('preserve previous CPU replay; use a fresh output directory')
 bundle=ROOT/'experiment/artifacts'/run['run'];job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin)
 result=validate_bundle(bundle,job,pin);versions=verify_dependencies();spec=job['model_spec']
 cache=Path(cache_root)/spec['model_repository'].split('/')[-1]/spec['model_revision']
 validate_snapshot(cache,spec['model_repository'],spec['model_revision'])
 backend=create_backend(spec,cache,job['seed']).load_pretrained()
 if next(backend.model.parameters()).device.type!='cpu':raise ValueError('expected controller CPU verification')
 measurements=[]
 for name in ['initial','best','terminal']:
  if time.monotonic()>deadline:raise TimeoutError('controller replay budget')
  backend.restore_checkpoint(bundle/(name+'.safetensors'))
  for split in ['development','regression']:
   records=[r for r in job['dataset']['records'] if r['split']==split]
   actual=backend.evaluate(records);expected=result['snapshots'][name]['predictions'][split]
   agreement=prediction_equal(actual,expected,spec['model_kind'],1e-4)
   delta=max(abs(a-b) for x,y in zip(actual,expected) for a,b in zip(x['probabilities'],y['probabilities']))
   measurements.append({'checkpoint':name,'split':split,'records':len(records),'prediction_agreement':agreement,'max_probability_difference':delta,'predictions':actual})
   print(json.dumps({'run':run['run'],'checkpoint':name,'split':split,'agreement':agreement,'max_probability_difference':delta}),flush=True)
 report={'schema':'saal-public-controller-cpu-replay-v1','run':run['run'],'job_digest':run['job_digest'],'runtime_dependencies':versions,'device':'cpu','probability_tolerance':1e-4,'all_predictions_agree':all(m['prediction_agreement'] for m in measurements),'measurements':measurements,'frozen_integrity':backend.verify_frozen_weights(),'elapsed_seconds':time.monotonic()-started,'independent_evaluator':False,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
 with destination.open('x') as f:f.write(json.dumps(report,sort_keys=True,indent=2))
 if not report['all_predictions_agree']:raise ValueError('CPU replay mismatch preserved')
 return report
if __name__=='__main__':replay(int(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else '/Users/dawsonblock/Downloads/SAAL-RSI-model-cache',sys.argv[3] if len(sys.argv)>3 else None)
