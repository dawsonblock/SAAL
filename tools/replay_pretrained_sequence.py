"""Restore every sequence stage on the controller and compare recorded accuracies.

Original sequence bundles record accuracy matrices, not per-record predictions.
The report therefore claims accuracy parity only and preserves replay predictions.
"""
from pathlib import Path
import argparse,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.spec import parse_unique_json,canonical,sha
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_contract import validate_job
from saal_rsi.pretrained_continual import validate_sequence
from saal_rsi.pretrained_registry import validate_snapshot,file_hash
from saal_rsi.learning_backend import create_backend,verify_dependencies
from saal_rsi.checkpoint_selection import metrics
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--pin',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.bundle);job=verify(parse_unique_json((root/'signed_job.json').read_bytes()),a.pin);validate_job(job,execution=False);protocol=verify(parse_unique_json((root/'signed_protocol.json').read_bytes()),a.pin)
if protocol['job_digest']!=sha(job):raise ValueError('sequence predecessor mismatch')
validate_sequence(protocol,job,execution=False)
if protocol['method']!='shared':raise ValueError('this replay tool supports shared-adapter sequence evidence only')
result=parse_unique_json((root/'result.json').read_bytes());inventory=parse_unique_json((root/'inventory.json').read_bytes())['files']
expected={'signed_job.json','signed_protocol.json','result.json'}|{f'{i}-{b}.safetensors' for i in range(len(protocol['groups'])) for b in ('experimental','fresh_control')}
if set(inventory)!=expected or {q.name for q in root.iterdir()}!=expected|{'inventory.json'}:raise ValueError('sequence inventory')
for n,record in inventory.items():
 q=root/n
 if q.is_symlink() or q.stat().st_size!=record['size'] or file_hash(q)!=record['sha256']:raise ValueError('sequence bytes')
if result['protocol_digest']!=sha(protocol) or result['qualification_status']!='NOT_EVALUATED' or result['external_deployment_allowed']is not False:raise ValueError('sequence identity/authority')
verify_dependencies();import torch
torch.set_num_threads(2);start=time.monotonic();spec=job['model_spec'];snapshot=Path(a.cache)/spec['model_repository'].split('/')[-1]/spec['model_revision'];validate_snapshot(snapshot,spec['model_repository'],spec['model_revision'])
b=create_backend(spec,snapshot,job['seed']).load_pretrained();groups=[[r for r in job['dataset']['records'] if r['split']=='development' and r['category'] in g] for g in protocol['groups']];replayed=[];matrix=[];fresh=[]
for i in range(len(groups)):
 b.restore_checkpoint(root/f'{i}-experimental.safetensors');row=[]
 for k,g in enumerate(groups):
  predictions=b.evaluate(g,job['max_new_tokens']);row.append(metrics(g,predictions,spec['model_kind'])['accuracy']);replayed.append({'stage':i,'branch':'experimental','group':k,'predictions':predictions})
 matrix.append(row);b.restore_checkpoint(root/f'{i}-fresh_control.safetensors');predictions=b.evaluate(groups[i],job['max_new_tokens']);fresh.append(metrics(groups[i],predictions,spec['model_kind'])['accuracy']);replayed.append({'stage':i,'branch':'fresh_control','group':i,'predictions':predictions})
record={'schema':'saal-sequence-controller-replay-v1','device':str(next(b.model.parameters()).device),'protocol_digest':sha(protocol),'stage_checkpoints_restored':len(groups)*2,'accuracy_matrix':matrix,'fresh_control_accuracy':fresh,'agreement':matrix==result['accuracy_matrix'] and fresh==result['fresh_control_accuracy'],'predictions':replayed,'elapsed_seconds':time.monotonic()-start,'qualification_status':'NOT_EVALUATED','independent_evaluator':False,'limits':'Accuracy parity after restoring every stage; original per-record predictions were not recorded, so identical per-record prediction parity cannot be established from this bundle.'}
Path(a.out).write_bytes(canonical(record));print('Sequence controller replay:',record['agreement'])
if not record['agreement']:raise SystemExit(1)
