"""Explicit signed budget amendment; never erase a failed or reserved attempt."""
from pathlib import Path
import copy,datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import sign,verify
from saal_rsi.epoch_contract import validate_job,worker_identity,SOURCES
from saal_rsi.epoch_backend_v2 import execute_campaign
PROJECT=Path(__file__).parent;OUT=PROJECT/'recovery/experiment'
def sources():
 paths=['src/saal_rsi/'+n for n in SOURCES]+['src/saal_rsi/epoch_backend.py','src/saal_rsi/epoch_backend_v2.py','src/saal_rsi/epoch_registry.py','src/saal_rsi/model_monitor.py','research/reliable-learning-v3/recover_campaign.py']
 return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in paths}
def prepare(key_file):
 from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 key=Ed25519PrivateKey.from_private_bytes(Path(key_file).read_bytes());pin=key.public_key().public_bytes_raw().hex();original=PROJECT/'experiment';protocol=verify(json.loads((original/'signed_protocol.json').read_text()),pin)
 state=json.loads((original/'colab/monitor-state.json').read_text());attempts=json.loads((original/'colab/outcomes.json').read_text())['runs']
 if not state['terminal'] or not state['cleanup_verified'] or len(attempts)!=5 or [r['state'] for r in attempts]!=['COMPLETED']*4+['INTERRUPTED']:raise ValueError('unexpected recovery state; no new compute admitted')
 OUT.mkdir(parents=True,exist_ok=False);mapping=[]
 for index in range(4,11):
  name=protocol['runs'][index]['name'];original_job=verify(json.loads((original/(name+'-job.json')).read_text()),pin);job=copy.deepcopy(original_job)
  if sha(job)!=protocol['job_digests'][index]:raise ValueError('original frozen job substitution')
  job['model_spec']['resource_budget_id']='recovery3_'+name;validate_job(job)
  (OUT/(name+'-job.json')).write_bytes(canonical(sign(job,key)));mapping.append({'original_index':index,'name':name,'original_job_digest':sha(original_job),'job_digest':sha(job)})
 body={'schema':'saal-public-budget-amendment-v1','registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':sources(),'worker_identity':worker_identity(),'parent_protocol_digest':sha(protocol),'parent_registry_export_sha256':hashlib.sha256((original/'colab/registry-export.json').read_bytes()).hexdigest(),'parent_cleanup_sha256':hashlib.sha256((original/'colab/cleanup-status.log').read_bytes()).hexdigest(),'roles':mapping,'prior_budget_reserved_seconds':41800,'additional_reserved_seconds':26600,'cumulative_reserved_seconds':68400,'new_job_slots':7,'max_seconds':28000,'reason':'Original fifth job reported training completion and GPU replay, but the manifest read failed. Its tensors were not recovered. Four completed jobs retained; one failed complete-training attempt is counted. Six original reservations remain unstarted and are not erased.','permitted_changes':['new resource_budget_id for each fresh attempt','versioned controller transport: explicit acknowledgement and bounded immutable reads'],'learning_parameters_changed':False,'resume_policy':'NO_RESUME','automatic_training_retries':0,'read_retries':5,'financial_cap':False,'protected_alpha_consumed':False,'human_reviews':0,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
 (OUT/'signed_amendment.json').write_bytes(canonical(sign(body,key)));(OUT/'controller.pub').write_text(pin+'\n');print(json.dumps({'jobs':7,'cumulative_reserved_seconds':68400,'source_files':len(sources())}))
def run():
 pin=(OUT/'controller.pub').read_text().strip();p=verify(json.loads((OUT/'signed_amendment.json').read_text()),pin);jobs=[]
 if p['source_files']!=sources() or p['worker_identity']!=worker_identity():raise ValueError('amended controller source substitution')
 original=PROJECT/'experiment'
 if hashlib.sha256((original/'colab/registry-export.json').read_bytes()).hexdigest()!=p['parent_registry_export_sha256']:raise ValueError('attempt history changed')
 for role in p['roles']:
  e=json.loads((OUT/(role['name']+'-job.json')).read_text());job=verify(e,pin);old=verify(json.loads((original/(role['name']+'-job.json')).read_text()),pin);restored=copy.deepcopy(job);restored['model_spec']['resource_budget_id']=old['model_spec']['resource_budget_id']
  if restored!=old or sha(job)!=role['job_digest'] or sha(old)!=role['original_job_digest']:raise ValueError('learning configuration changed')
  jobs.append(e)
 return execute_campaign(OUT/'colab',jobs,pin,OUT/'artifacts',OUT/'additional-budget.sqlite3',max_seconds=28000)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(sys.argv[2])
 elif sys.argv[1]=='run':run()
 else:raise ValueError('closed amendment command')
