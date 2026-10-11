"""Controller-host prediction replay using authenticated upstream cache and snapshots."""
from pathlib import Path
import argparse,time,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.spec import parse_unique_json,canonical,sha
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_artifacts import validate_bundle
from saal_rsi.pretrained_registry import validate_snapshot
from saal_rsi.learning_backend import create_backend,verify_dependencies
from saal_rsi.pretrained_contract import worker_identity
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--pin',required=True);p.add_argument('--cache',required=True);p.add_argument('--records',type=int,default=0);p.add_argument('--out',required=True);a=p.parse_args()
if not 0<=a.records<=2000:raise ValueError('replay record bound')
root=Path(a.bundle);job=verify(parse_unique_json((root/'signed_training_job.json').read_bytes()),a.pin);result=validate_bundle(root,job,a.pin);verify_dependencies()
import torch
torch.set_num_threads(2);start=time.monotonic();spec=job['model_spec'];snapshot=Path(a.cache)/spec['model_repository'].split('/')[-1]/spec['model_revision'];validate_snapshot(snapshot,spec['model_repository'],spec['model_revision'])
b=create_backend(spec,snapshot,job['seed']).load_pretrained();b.restore_checkpoint(root/'best.safetensors');records=[r for r in job['dataset']['records'] if r['split']=='development'];count=min(a.records or len(records),len(records));pred=b.evaluate(records[:count],job['max_new_tokens']);expected=result['snapshots']['best']['predictions']['development'][:count]
if spec['model_kind']=='modernbert-seqcls-v1':
    agree=all(x['decision']==y['decision'] and max(abs(u-v) for u,v in zip(x['probabilities'],y['probabilities']))<=1e-4 for x,y in zip(pred,expected));tolerance=1e-4
else:agree=pred==expected;tolerance='exact decoded strings'
record={'schema':'saal-controller-replay-v1','job_digest':sha(job),'model_identity':result['model_identity'],'replay_implementation':worker_identity(),'device':str(next(b.model.parameters()).device),'records':count,'full_development_population':count==len(records),'agreement':agree,'tolerance':tolerance,'predictions':pred,'elapsed_seconds':time.monotonic()-start,'qualification_status':'NOT_EVALUATED','independent_evaluator':False}
Path(a.out).write_bytes(canonical(record));print('Controller replay:',agree,'records:',count,'device:',record['device'])
if not agree:raise SystemExit(1)
