"""Frozen public replay experiment using the unchanged v5 GPU worker."""
from pathlib import Path
import sys,json,hashlib,random,collections
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import sign,verify
from saal_rsi.pretrained_contract import new_job,model_spec,worker_identity
from saal_rsi.reliability_corpus import validate
from saal_rsi.pretrained_backend import execute_campaign
RUNS=[('mixed_full_47','full',47,.00003),('mixed_lora_47','lora',47,.0003),('mixed_lora_83','lora',83,.0003)]
SOURCE_FILES=['src/saal_rsi/'+n for n in ['choice_policy_v1.py','decision_guard.py','learning_audit.py','public_review.py','reliability_corpus.py','reliability_cli.py']]+['src/saal_diverse/'+n for n in ['__init__.py','policy.py','pair_recipe.py','dataset.py','campaign.py']]+['research/learning-reliability-v2/run_campaign.py']
def source_identity():return sha({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCE_FILES})
def prepare(key_file):
 from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 project=Path(__file__).parent;out=project/'experiment';out.mkdir(exist_ok=False)
 data=json.loads((project/'corpus/dataset.json').read_text());audit=json.loads((project/'corpus/state-audit.json').read_text());prior=ROOT/'research/diverse-learning-v1/experiment'
 validate(data,audit,json.loads((prior/'diverse_dataset.json').read_text()),json.loads((prior/'old_control_dataset.json').read_text()))
 key=Ed25519PrivateKey.from_private_bytes(Path(key_file).read_bytes());pin=key.public_key().public_bytes_raw().hex();jobs=[];plans={}
 for name,method,seed,lr in RUNS:
  spec=model_spec('answerdotai/ModernBERT-base',method=method,rank=4,backend='colab',precision='float32',budget='reliability2_'+name,max_tokens=256)
  job=new_job(spec,data,seed=seed,steps=1000,batch_size=8,learning_rate=lr,max_seconds=2400,class_balance=True,checkpoint_interval=100)
  envelope=sign(job,key);jobs.append(envelope);(out/(name+'-job.json')).write_bytes(canonical(envelope))
  train=[r for r in data['records'] if r['split']=='train'];rng=random.Random(seed);ids=[rng.choice(train)['record_id'] for _ in range(1000*8)]
  plan={'scope':'Controller reconstruction of admitted native uniform sampler; not hostile-worker attestation.','record_ids':ids,'source_exposures':dict(collections.Counter(audit['record_sources'][r] for r in ids)),'unique_records':len(set(ids))}
  plans[name]=sha(plan);(out/(name+'-expected-sampling.json')).write_bytes(canonical(plan))
 protocol={'schema':'saal-public-reliability-protocol-v2','source_identity':source_identity(),'worker_identity':worker_identity(),'dataset_digest':sha(data),'state_audit_digest':sha(audit),'job_digests':[sha(e['payload']) for e in jobs],'runs':[{'name':n,'method':m,'seed':s,'learning_rate':lr} for n,m,s,lr in RUNS],'sampling_plan_digests':plans,'steps_per_job':1000,'batch_size':8,'precision':'float32','max_seconds_per_job':2400,'public_screen':{'complete_pair_accuracy':.9,'worst_recipe_accuracy':.8,'class_coverage':4,'critical_proposals':0,'invalid_outputs':0},'max_public_family_retention_drop':.05,'retention_reference':{'run':'run-04-4f133755fa72','snapshot':'terminal','job_digest':None,'authority':'Prior controller pin; exact reference verified before reporting.'},'selection':'Historical safety-first worker selector unchanged; initial,best,terminal retained. Separate public learning audit over physically retained checkpoints only. Regression never selects weights.','hypothesis':'Compact failure-informed instructions plus public training replay may improve new-format learning and preserve prior cases. Changes to data and update budget confound comparison against v1. Full/LoRA seed47 have matched source initialization,data sampler and update budgets; adapter/dropout methods differ.','human_reviews':0,'independence_established':False,'protected_alpha_consumed':False,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'limits':'Shared-author/oracle recipes and previously exposed public diagnostics; descriptive learning/retention only. Two LoRA seeds, one full seed. No evaluator/custodian secrets or protected labels transferred.'}
 # Obtain the actual frozen reference digest rather than inventing lineage.
 reference=json.loads((prior/'artifacts/run-04-4f133755fa72/signed_training_job.json').read_text())
 protocol['retention_reference']['controller_pin']=(prior/'controller.pub').read_text().strip()
 protocol['retention_reference']['job_digest']=sha(verify(reference,protocol['retention_reference']['controller_pin']))
 protocol['retention_reference']['inventory_sha256']=hashlib.sha256((prior/'artifacts/run-04-4f133755fa72/artifact_inventory.json').read_bytes()).hexdigest()
 (out/'signed_protocol.json').write_bytes(canonical(sign(protocol,key)));(out/'controller.pub').write_text(pin+'\n');print(json.dumps({'jobs':len(jobs),'dataset_digest':sha(data),'source_identity':source_identity()},indent=2))
def run():
 root=Path(__file__).parent/'experiment';pin=(root/'controller.pub').read_text().strip();protocol=verify(json.loads((root/'signed_protocol.json').read_text()),pin)
 if protocol['source_identity']!=source_identity() or protocol['worker_identity']!=worker_identity():raise ValueError('frozen source substitution')
 jobs=[json.loads((root/(n+'-job.json')).read_text()) for n,_,_,_ in RUNS]
 if [sha(verify(e,pin)) for e in jobs]!=protocol['job_digests']:raise ValueError('job substitution')
 for n,digest in protocol['sampling_plan_digests'].items():
  if sha(json.loads((root/(n+'-expected-sampling.json')).read_text()))!=digest:raise ValueError('sampler evidence substitution')
 return execute_campaign(root/'colab',jobs,pin,root/'artifacts',root/'development-ledger.sqlite3',max_seconds=14000)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(sys.argv[2])
 elif sys.argv[1]=='run':run()
 else:raise ValueError('closed campaign command')
