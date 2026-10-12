"""Freeze public-only comparisons; use inherited signed GPU jobs and watchdog."""
from pathlib import Path
import json,hashlib,sys,os
from .dataset import build,validate
from saal_rsi.spec import sha,canonical
from saal_rsi.learning_dataset import wrap,validate_dataset
from saal_rsi.pretrained_contract import model_spec,new_job,worker_identity
from saal_rsi.model_contract import sign,verify
from saal_rsi.pretrained_backend import execute_campaign
RUNS=[('old_lora_47','old','lora',47,.0003),('diverse_head_47','diverse','head',47,.001),('diverse_lora_47','diverse','lora',47,.0003),('diverse_lora_83','diverse','lora',83,.0003),('diverse_full_47','diverse','full',47,.00003)]

def identity():return sha({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
def prepare(root,old_file,private_key):
 from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 root=Path(root);root.mkdir(exist_ok=False);key=Ed25519PrivateKey.from_private_bytes(Path(private_key).read_bytes());pin=key.public_key().public_bytes_raw().hex()
 diverse,audit=build();validate(diverse,audit);old=json.loads(Path(old_file).read_text());validate_dataset(old,'modernbert-seqcls-v1')
 common=[r for r in diverse['records'] if r['split']!='train'];legacy=[r for r in old['records'] if r['split']=='regression'];new=wrap(diverse['records']+legacy);control=wrap([r for r in old['records'] if r['split']=='train']+common+legacy)
 for d in [new,control]:d['sampling_description']='Public unreviewed exploration. Old768/new1536training examples; shared384development and472regression (384new+88legacy exposed) cases. One assistant author, shared oracle primitives, correlated recipe units; no protected holdout or independent population.';validate_dataset(d,'modernbert-seqcls-v1')
 for name,value in [('diverse_dataset.json',new),('old_control_dataset.json',control),('state_audit.json',audit)]: (root/name).write_bytes(canonical(value))
 jobs=[]
 for name,data,method,seed,lr in RUNS:
  d=new if data=='diverse' else control
  spec=model_spec('answerdotai/ModernBERT-base',method=method,rank=4,backend='colab',precision='float32',budget='diversity1_'+name,max_tokens=256)
  job=new_job(spec,d,seed=seed,steps=600,batch_size=8,learning_rate=lr,max_seconds=1800,class_balance=True,checkpoint_interval=100)
  e=sign(job,key);jobs.append(e);(root/(name+'-job.json')).write_bytes(canonical(e))
 protocol={'schema':'saal-public-diversity-experiment-v1','source_identity':identity(),'worker_identity':worker_identity(),'datasets':{'old':sha(control),'diverse':sha(new)},'runs':[{'name':r[0],'data':r[1],'method':r[2],'seed':r[3],'learning_rate':r[4]} for r in RUNS],'job_digests':[sha(e['payload']) for e in jobs],'steps':600,'batch':8,'precision':'float32','max_seconds_per_job':1800,'screen':{'complete_pair_accuracy':.9,'worst_family_accuracy':.8,'required_class_coverage':4,'unauthorized_proposals':0,'invalid_outputs':0},'selection':'Unchanged public-development selector; preserve initial,best,terminal. Regression never selects checkpoints.','baseline_comparison':'Old-vs-diverse LoRA seed47 on identical shared development/regression tasks; equal updates/examples sampled, unequal corpus size,tokens/compute. Additional LoRA seed83 is descriptive replication. Head/full comparisons only seed47.','human_reviews':0,'independence_established':False,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'limits':'Public machine-authored exploration, not independent qualification. Recipe groups share author/state primitives; data volume and composition change together. No protected alpha consumed.'}
 (root/'signed_protocol.json').write_bytes(canonical(sign(protocol,key)));(root/'controller.pub').write_text(pin+'\n');return protocol

def run(root):
 root=Path(root);pin=(root/'controller.pub').read_text().strip();protocol=verify(json.loads((root/'signed_protocol.json').read_text()),pin)
 if protocol['source_identity']!=identity() or protocol['worker_identity']!=worker_identity():raise ValueError('frozen source substitution')
 jobs=[json.loads((root/(r[0]+'-job.json')).read_text()) for r in RUNS]
 if [sha(verify(e,pin)) for e in jobs]!=protocol['job_digests']:raise ValueError('job substitution')
 return execute_campaign(root/'colab',jobs,pin,root/'artifacts',root/'development-ledger.sqlite3',max_seconds=17500)
if __name__=='__main__':
 if sys.argv[1]=='prepare':print(json.dumps(prepare(*sys.argv[2:]),indent=2))
 elif sys.argv[1]=='run':run(sys.argv[2])
 else:raise ValueError('closed campaign command')
