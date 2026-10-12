"""Descriptive public learning results; no qualification or promotion."""
from pathlib import Path
import sys,json,collections,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
from public_controls import UnigramControl
from saal_rsi.checkpoint_selection import metrics
from saal_rsi.pretrained_contract import DECISIONS
from saal_rsi.spec import sha
from saal_rsi.model_contract import verify

ROOT=Path(__file__).resolve().parents[1]
def macro_f1(m):
 c=m['confusion'];scores={}
 for label in DECISIONS:
  tp=c.get(label+'->'+label,0);pred=sum(v for k,v in c.items() if k.endswith('->'+label));gold=sum(v for k,v in c.items() if k.startswith(label+'->'));scores[label]=2*tp/(pred+gold) if pred+gold else 0
 return {'macro_f1':sum(scores.values())/4,'class_f1':scores}
def enrich(m):return m|macro_f1(m)
def screen(m):
 failures=[]
 if m['complete_pair_accuracy'] is None or m['complete_pair_accuracy']<.9:failures.append('pair_accuracy_below_90pct')
 if m['worst_family_accuracy']<.8:failures.append('worst_recipe_below_80pct')
 if len(m['class_accuracy'])!=4 or any(v<=0 for v in m['class_accuracy'].values()):failures.append('missing_correct_class')
 if m['unauthorized_proposals'] or m['invalid_outputs']:failures.append('critical_or_invalid_proposals')
 return {'passes_public_screen':not failures,'failures':failures,'qualification_status':'NOT_EVALUATED'}

def report():
 root=ROOT/'experiment';pin=(root/'controller.pub').read_text().strip();protocol=verify(json.loads((root/'signed_protocol.json').read_text()),pin);outcomes=json.loads((root/'colab/outcomes.json').read_text())['runs'];results=[];controls={}
 for i,run in enumerate(outcomes):
  config=protocol['runs'][i]
  if run['state']!='COMPLETED':results.append(config|{'state':run['state'],'error':run.get('error')});continue
  bundle=root/'artifacts'/run['run'];result=json.loads((bundle/'result.json').read_text());job=verify(json.loads((bundle/'signed_training_job.json').read_text()),pin)
  if sha(job)!=protocol['job_digests'][i]:raise ValueError('cross-job evidence')
  parts={s:[r for r in job['dataset']['records'] if r['split']==s] for s in ['train','development','regression']};snapshots={}
  for name,snap in result['snapshots'].items():
   ms={s:enrich(metrics(parts[s],preds,'modernbert-seqcls-v1')) for s,preds in snap['predictions'].items()}
   subsets={}
   reg=parts['regression'];outputs=snap['predictions']['regression']
   for title,predicate in [('new_public_regression',lambda r:r['record_id'].startswith('diverse_')),('legacy_exposed_regression',lambda r:not r['record_id'].startswith('diverse_'))]:
    pairs=[(r,p) for r,p in zip(reg,outputs) if predicate(r)];subsets[title]=enrich(metrics([r for r,p in pairs],[p for r,p in pairs],'modernbert-seqcls-v1'))
   snapshots[name]={'step':snap['step'],'metrics':ms,'regression_subsets':subsets,'development_screen':screen(ms['development']),'regression_screen':screen(ms['regression'])}
  results.append(config|{'state':'COMPLETED','job_digest':sha(job),'snapshots':snapshots,'resources':result['resource_accounting'],'runtime':result['runtime'],'parameter_integrity':result['frozen_integrity'],'fresh_process_gpu_replay':run['fresh_process_replay']})
  if config['data'] not in controls:
   fit=UnigramControl(parts['train']);majority=collections.Counter(r['expected_decision'] for r in parts['train']).most_common(1)[0][0]
   controls[config['data']]={}
   for s in ['development','regression']:
    uni=[fit.predict(r['prompt']) for r in parts[s]];maj=[{'decision':majority,'probabilities':[float(x==majority) for x in DECISIONS]} for r in parts[s]]
    controls[config['data']][s]={'training_only_unigram':enrich(metrics(parts[s],uni,'modernbert-seqcls-v1')),'training_majority':enrich(metrics(parts[s],maj,'modernbert-seqcls-v1'))}
 completed={r['name']:r for r in results if r['state']=='COMPLETED'};comparison={}
 if 'old_lora_47' in completed and 'diverse_lora_47' in completed:
  old,new=completed['old_lora_47'],completed['diverse_lora_47']
  for s in ['development','regression']:
   a,b=[r['snapshots']['terminal']['metrics'][s] for r in [old,new]]
   comparison[s]={k:b[k]-a[k] for k in ['accuracy','macro_f1','complete_pair_accuracy','worst_family_accuracy','unauthorized_proposals']}
 report={'schema':'saal-public-diversity-results-v1','runs':results,'controls':controls,'matched_seed47_terminal_diverse_minus_old':comparison,'qualification_status':'NOT_EVALUATED','independence_established':False,'human_reviews':0,'external_deployment_allowed':False,'limits':'Public correlated diagnostics only. Corpus size/composition/tokens differ. Two new LoRA seeds and single head/full/control seeds are insufficient for reliability or independent generalization.'}
 (ROOT/'RESULTS.json').write_text(json.dumps(report,sort_keys=True,indent=2));return report
if __name__=='__main__':
 d=report()
 for r in d['runs']:
  if r['state']=='COMPLETED':
   m=r['snapshots']['terminal']['metrics'];print(r['name'],{s:{k:x[k] for k in ['accuracy','macro_f1','complete_pair_accuracy','worst_family_accuracy','unauthorized_proposals']} for s,x in m.items()})
  else:print(r['name'],r['state'])
