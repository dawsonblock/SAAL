"""Separate exploratory head-only control; primary 11-job protocol unchanged."""
from pathlib import Path
import hashlib,json,sys,datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.spec import sha,canonical
from saal_rsi.model_contract import verify,sign
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.epoch_contract import new_job,worker_identity,SOURCES
from saal_rsi.epoch_backend import execute_campaign
PROJECT=Path(__file__).parent;OUT=PROJECT/'head-control/experiment'
def sources():
 names=['src/saal_rsi/'+n for n in SOURCES]+['src/saal_rsi/epoch_backend.py','src/saal_rsi/epoch_registry.py','src/saal_rsi/model_monitor.py','research/reliable-learning-v3/head_control.py']
 return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
def prepare(key_file):
 from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 OUT.mkdir(parents=True,exist_ok=False);key=Ed25519PrivateKey.from_private_bytes(Path(key_file).read_bytes());pin=key.public_key().public_bytes_raw().hex()
 primary=PROJECT/'experiment';protocol=verify(json.loads((primary/'signed_protocol.json').read_text()),pin);data=json.loads((ROOT/'research/learning-reliability-v2/corpus/dataset.json').read_text())
 spec=model_spec('answerdotai/ModernBERT-base',method='head',backend='colab',precision='float32',budget='reliable3_head_control_47',max_tokens=256)
 job=new_job(spec,data,sampler='complete-balanced-epochs-v1',seed=47,steps=1000,batch_size=8,learning_rate=.0003,head_learning_rate=.0003,encoder_learning_rate=.0003,max_seconds=2400,class_balance=True,checkpoint_interval=100)
 observed=json.loads((primary/'colab/outcomes.json').read_text())['runs']
 body={'schema':'saal-public-head-control-v1','source_files':sources(),'worker_identity':worker_identity(),'job_digest':sha(job),'primary_protocol_digest':sha(protocol),'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'added_after_observed_job_digests':[r['job_digest'] for r in observed],'purpose':'Exploratory matched head-only control, added after observing early LoRA results. Not part of the preregistered primary five-seed hypothesis. Measures contribution of head training; not independent qualification.','matched_reference':{'seed':47,'sampler':'complete-balanced-epochs-v1','primary_job_digest':protocol['job_digests'][1],'same_data_digest':sha(data),'same_initial_prediction_tolerance':1e-4,'same_steps':1000,'same_batch':8,'same_learning_rate':.0003,'limits':'Methods have different trainable parameter sets and dropout, recorded resources; a single head-only seed is not a repeated-seed method comparison.'},'max_seconds':5000,'human_reviews':0,'protected_alpha_consumed':False,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
 (OUT/'signed_job.json').write_bytes(canonical(sign(job,key)));(OUT/'signed_protocol.json').write_bytes(canonical(sign(body,key)));(OUT/'controller.pub').write_text(pin+'\n');print(json.dumps({'job_digest':sha(job),'observed_primary_runs':len(observed),'source_files':len(sources())}))
def run():
 pin=(OUT/'controller.pub').read_text().strip();p=verify(json.loads((OUT/'signed_protocol.json').read_text()),pin);e=json.loads((OUT/'signed_job.json').read_text())
 if p['source_files']!=sources() or p['worker_identity']!=worker_identity() or sha(verify(e,pin))!=p['job_digest']:raise ValueError('exploratory control substitution')
 return execute_campaign(OUT/'colab',[e],pin,OUT/'artifacts',OUT/'development-ledger.sqlite3',max_seconds=5000)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(sys.argv[2])
 elif sys.argv[1]=='run':run()
 else:raise ValueError('closed head control command')
