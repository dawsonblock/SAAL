"""Preregistered public sampler ablation; no protected qualification or retries."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.spec import canonical,sha
from saal_rsi.model_contract import sign,verify
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.epoch_contract import new_job,worker_identity,SOURCES
from saal_rsi.epoch_sampler import manifests
from saal_rsi.epoch_backend import execute_campaign

SEEDS=(47,83,109,157,211)
RUNS=[(f'{mode}_{seed}','lora',seed,mode) for seed in SEEDS for mode in ('uniform-v1','complete-balanced-epochs-v1')]+[('full_warm_47','full',47,'complete-balanced-epochs-v1')]
FILES=['src/saal_rsi/'+n for n in SOURCES]+['src/saal_rsi/epoch_backend.py','src/saal_rsi/epoch_registry.py','src/saal_rsi/model_monitor.py','research/reliable-learning-v3/run_campaign.py','research/reliable-learning-v3/prepare_review.py','configs/pretrained/registry.json','configs/pretrained/tensor_inventories.json','configs/dependency-locks/pretrained-linux-x86_64-py313.txt']
def source_map():return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in FILES}
def prepare(key_file):
 from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 project=Path(__file__).parent;out=project/'experiment';out.mkdir(exist_ok=False)
 data=json.loads((ROOT/'research/learning-reliability-v2/corpus/dataset.json').read_text())
 key=Ed25519PrivateKey.from_private_bytes(Path(key_file).read_bytes());pin=key.public_key().public_bytes_raw().hex();jobs=[];plans={}
 for name,method,seed,mode in RUNS:
  spec=model_spec('answerdotai/ModernBERT-base',method=method,rank=4,backend='colab',precision='float32',budget='reliable3_'+name,max_tokens=256)
  job=new_job(spec,data,sampler=mode,diagnostic_seed=730,diagnostic_per_class=128,seed=seed,steps=1000,batch_size=8,learning_rate=.0003,max_seconds=2400,class_balance=True,checkpoint_interval=100,head_learning_rate=.0003,encoder_learning_rate=.00001 if method=='full' else .0003,head_warmup_steps=100 if method=='full' else 0)
  envelope=sign(job,key);jobs.append(envelope);(out/(name+'-job.json')).write_bytes(canonical(envelope))
  sample,diag=manifests(job);plan={'sampler':sample,'diagnostic':diag};plans[name]=sha(plan);(out/(name+'-expected-sampling.json')).write_bytes(canonical(plan))
 protocol={'schema':'saal-public-reliable-learning-protocol-v3','source_identity':sha(source_map()),'source_files':source_map(),'worker_identity':worker_identity(),'dataset_digest':sha(data),'review_pilot_digest':sha(json.loads((project/'review-pilot/dataset.json').read_text())),'job_digests':[sha(e['payload']) for e in jobs],'runs':[{'name':n,'method':m,'seed':s,'sampler':mode} for n,m,s,mode in RUNS],'sampling_plan_digests':plans,'hypothesis':'Complete class-balanced epochs may improve learning over the native uniform-with-replacement sampler with the same source initialization, available public data and optimizer update budget. The full-tuning warm-up and reduced encoder learning rate are a single-seed exploratory diagnostic.','matched_controls':{'five_paired_seeds':list(SEEDS),'lora_rank':4,'steps':1000,'batch_size':8,'accumulation':1,'learning_rate':.0003,'class_balance':True,'checkpoint_interval':100,'precision':'float32','exposures_per_run':8000,'tokens':'Measured, not guaranteed identical: different sampled input lengths.','diagnostic':'Identical fixed 512 training examples; never used for checkpoint selection. Entire 3840 training population evaluated before and after.'},'selection':'Native public development safety-first rule unchanged. Initial, best-development and terminal all preserved. Training and exposed regression do not select weights.','public_screen':{'complete_pair_accuracy':.9,'worst_recipe_accuracy':.8,'correct_class_coverage':4,'critical_proposals':0,'invalid_outputs':0},'conditional_sequential_phase':{'status':'PENDING_BASIC_LEARNING_GATE','eligibility':'At least one non-initial candidate passes the development public screen with positive family-macro adaptation gain.','rule':'If none is eligible, sequential adaptation is SKIPPED_NOT_ELIGIBLE. If eligible, freeze a separate matched sequential protocol before execution; this experiment alone does not measure sequential forgetting.'},'resource_policy':{'max_seconds_per_run':2400,'campaign_stage_deadline_seconds':43000,'runs':11,'automatic_retries':0,'resume':'NO_RESUME','financial_cap':False},'human_reviews':0,'independence_established':False,'protected_alpha_consumed':False,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,'limits':'Unreviewed public synthetic labels; shared oracle and recipe mechanisms, previously exposed diagnostics. This sampler ablation does not repair independent data custody or demonstrate protected generalization. No authority/evaluator secrets or protected answers are staged.'}
 (out/'signed_protocol.json').write_bytes(canonical(sign(protocol,key)));(out/'controller.pub').write_text(pin+'\n')
 print(json.dumps({'jobs':len(jobs),'dataset_digest':sha(data),'source_identity':sha(source_map()),'worker_identity':worker_identity()},indent=2))
def run():
 root=Path(__file__).parent/'experiment';pin=(root/'controller.pub').read_text().strip();protocol=verify(json.loads((root/'signed_protocol.json').read_text()),pin)
 if protocol['source_files']!=source_map() or protocol['worker_identity']!=worker_identity():raise ValueError('frozen source substitution')
 jobs=[json.loads((root/(n+'-job.json')).read_text()) for n,_,_,_ in RUNS]
 if [sha(verify(e,pin)) for e in jobs]!=protocol['job_digests']:raise ValueError('job substitution')
 for n,digest in protocol['sampling_plan_digests'].items():
  if sha(json.loads((root/(n+'-expected-sampling.json')).read_text()))!=digest:raise ValueError('sample plan substitution')
 return execute_campaign(root/'colab',jobs,pin,root/'artifacts',root/'development-ledger.sqlite3',max_seconds=43000)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(sys.argv[2])
 elif sys.argv[1]=='run':run()
 else:raise ValueError('closed campaign command')
