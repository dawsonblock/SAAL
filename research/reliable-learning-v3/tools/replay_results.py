"""Fresh controller CPU inference of all retained snapshots; preserve failures."""
from pathlib import Path
import json,time,sys
from saal_rsi.model_contract import verify
from saal_rsi.learning_backend import create_backend,verify_dependencies
from saal_rsi.epoch_artifacts import validate_bundle
from saal_rsi.pretrained_registry import validate_snapshot
from saal_rsi.epoch_metrics import prediction_equal
ROOT=Path(__file__).resolve().parents[1]

def verified_measurement(report,result,digest):
 if report['job_digest']!=digest or report['probability_tolerance']!=1e-4:raise ValueError('CPU replay identity')
 expected_pairs={(c,s) for c in ('initial','best','terminal') for s in ('development','regression')};seen=set();agrees=[];labels=[];deltas=[]
 for m in report['measurements']:
  pair=m['checkpoint'],m['split']
  if pair not in expected_pairs or pair in seen:raise ValueError('CPU measurement inventory')
  seen.add(pair);a=m['predictions'];b=result['snapshots'][pair[0]]['predictions'][pair[1]]
  if len(a)!=m['records'] or len(a)!=len(b) or not prediction_equal(a,a,'modernbert-seqcls-v1',1e-4):raise ValueError('CPU malformed predictions')
  ok=prediction_equal(a,b,'modernbert-seqcls-v1',1e-4);delta=max(abs(x-y) for p,q in zip(a,b) for x,y in zip(p['probabilities'],q['probabilities']))
  if ok!=m['prediction_agreement'] or delta!=m['max_probability_difference']:raise ValueError('forged CPU comparison')
  agrees.append(ok);labels.append(all(p['decision']==q['decision'] for p,q in zip(a,b)));deltas.append(delta)
 if seen!=expected_pairs or all(agrees)!=report['all_predictions_agree']:raise ValueError('CPU aggregate mismatch')
 return {'passes_registered_probability_tolerance':all(agrees),'all_decisions_match':all(labels),'max_probability_difference':max(deltas)}

def replay(index,cache_root):
 import torch
 torch.set_num_threads(6);started=time.monotonic();deadline=started+1800
 pin=(ROOT/'experiment/controller.pub').read_text().strip();runs=json.loads((ROOT/'experiment/colab/outcomes.json').read_text())['runs'];run=runs[index]
 if run['state']!='COMPLETED':raise ValueError('incomplete GPU run')
 out=ROOT/'experiment/controller-replay';out.mkdir(exist_ok=True);destination=out/(run['run']+'.json')
 if destination.exists():raise FileExistsError('preserve earlier inference record')
 bundle=ROOT/'experiment/artifacts'/run['run'];job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin);result=validate_bundle(bundle,job,pin)
 spec=job['model_spec'];cache=Path(cache_root)/spec['model_repository'].split('/')[-1]/spec['model_revision'];validate_snapshot(cache,spec['model_repository'],spec['model_revision'])
 backend=create_backend(spec,cache,job['seed']).load_pretrained()
 if next(backend.model.parameters()).device.type!='cpu':raise ValueError('CPU expected')
 measurements=[]
 for name in ('initial','best','terminal'):
  if time.monotonic()>deadline:raise TimeoutError('bounded CPU restoration')
  backend.restore_checkpoint(bundle/(name+'.safetensors'))
  for split in ('development','regression'):
   rows=[r for r in job['dataset']['records'] if r['split']==split];actual=backend.evaluate(rows);expected=result['snapshots'][name]['predictions'][split]
   ok=prediction_equal(actual,expected,spec['model_kind'],1e-4);delta=max(abs(x-y) for p,q in zip(actual,expected) for x,y in zip(p['probabilities'],q['probabilities']))
   measurements.append({'checkpoint':name,'split':split,'records':len(rows),'prediction_agreement':ok,'max_probability_difference':delta,'predictions':actual})
   print(json.dumps({'run':run['run'],'checkpoint':name,'split':split,'agreement':ok,'max_probability_difference':delta}),flush=True)
 report={'schema':'saal-public-controller-cpu-replay-v1','run':run['run'],'job_digest':run['job_digest'],'runtime_dependencies':verify_dependencies(),'device':'cpu','probability_tolerance':1e-4,'all_predictions_agree':all(m['prediction_agreement'] for m in measurements),'measurements':measurements,'frozen_integrity':backend.verify_frozen_weights(),'elapsed_seconds':time.monotonic()-started,'independent_evaluator':False,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
 verified_measurement(report,result,run['job_digest'])
 with destination.open('x') as f:json.dump(report,f,sort_keys=True,indent=2)
 return report

if __name__=='__main__':
 if sys.argv[1]=='watch':
  deadline=time.monotonic()+44000;screens=[]
  for i in range(11):
   while True:
    if time.monotonic()>deadline:raise TimeoutError('controller wait budget')
    p=ROOT/'experiment/colab/outcomes.json';runs=json.loads(p.read_text())['runs'] if p.exists() else []
    if len(runs)>i:break
    time.sleep(10)
   if runs[i]['state']!='COMPLETED':raise RuntimeError('negative GPU outcome preserved')
   record=replay(i,sys.argv[2]);result=json.loads((ROOT/'experiment/artifacts'/runs[i]['run']/'result.json').read_text());s=verified_measurement(record,result,runs[i]['job_digest']);screens.append(s|{'run':runs[i]['run']});print(json.dumps({'controller_cpu_replay':runs[i]['run'],**s}),flush=True)
  with (ROOT/'experiment/controller-replay-status.json').open('x') as f:json.dump({'runs':screens,'all_recorded':True,'all_probability_screens_pass':all(s['passes_registered_probability_tolerance'] for s in screens),'all_decisions_match':all(s['all_decisions_match'] for s in screens),'qualification_status':'NOT_EVALUATED'},f,indent=2)
 else:replay(int(sys.argv[1]),sys.argv[2])
