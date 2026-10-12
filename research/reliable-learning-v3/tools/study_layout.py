"""Reconcile explicitly amended roles without rewriting the original history."""
from pathlib import Path
import copy,hashlib,json
from saal_rsi.model_contract import verify
from saal_rsi.spec import sha
def load_runs(project,pin):
 project=Path(project);original=project/'experiment';protocol=verify(json.loads((original/'signed_protocol.json').read_text()),pin);old=json.loads((original/'colab/outcomes.json').read_text())['runs'];runs=[]
 for i,r in enumerate(old):
  if r['job_digest']!=protocol['job_digests'][i]:raise ValueError('original attempt substitution')
  if r['state']=='COMPLETED':runs.append(r|{'role_index':i,'original_job_digest':r['job_digest'],'bundle_relative':'experiment/artifacts/'+r['run']})
 repair=project/'recovery/experiment';amendment=repair/'signed_amendment.json'
 if amendment.exists():
  a=verify(json.loads(amendment.read_text()),pin)
  if a['parent_protocol_digest']!=sha(protocol) or hashlib.sha256((original/'colab/registry-export.json').read_bytes()).hexdigest()!=a['parent_registry_export_sha256']:raise ValueError('attempt ancestry changed')
  execution=repair/'signed_execution_location.json'
  if execution.exists():
   location=verify(json.loads(execution.read_text()),pin)['relative_workspace']
   if Path(location).is_absolute() or '..' in Path(location).parts:raise ValueError('execution workspace boundary')
   p=project/location/'outcomes.json'
  else:p=repair/'colab/outcomes.json'
  new=json.loads(p.read_text())['runs'] if p.exists() else []
  for n,r in enumerate(new):
   role=a['roles'][n];index=role['original_index'];name=role['name'];job=verify(json.loads((repair/(name+'-job.json')).read_text()),pin);parent=verify(json.loads((original/(name+'-job.json')).read_text()),pin);comparison=copy.deepcopy(job);comparison['model_spec']['resource_budget_id']=parent['model_spec']['resource_budget_id']
   if r['job_digest']!=role['job_digest'] or sha(job)!=r['job_digest'] or sha(parent)!=role['original_job_digest'] or comparison!=parent:raise ValueError('amendment changed learning or identity')
   if r['state']=='COMPLETED':runs.append(r|{'role_index':index,'original_job_digest':role['original_job_digest'],'bundle_relative':'recovery/experiment/artifacts/'+r['run']})
 runs.sort(key=lambda r:r['role_index'])
 if len({r['role_index'] for r in runs})!=len(runs):raise ValueError('duplicate completed primary role')
 return runs,protocol
def bundle_path(project,run):
 name=run['bundle_relative']
 if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('bundle layout boundary')
 return Path(project)/name
