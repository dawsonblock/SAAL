import json,collections
from pathlib import Path
from .policy import decide,digest,differences,LABELS,VERSION
from .build import render

def validate(corpus):
 if corpus['schema']!='saal-corpus-review-candidate-v1' or corpus['policy_version']!=VERSION or corpus['qualification_status']!='NOT_EVALUATED' or corpus['training_ready']is not False:raise ValueError('draft cannot grant training or qualification')
 records=corpus['records'];ids={};owners={};prompts=set();splits=collections.Counter();labels=collections.defaultdict(collections.Counter)
 for r in records:
  if set(r)!={'schema','record_id','task_family_id','cluster_id','counterfactual_pair_id','generation_rule_id','template_id','split','prompt','prompt_digest','state','state_digest','proposed_decision','label_rationale','policy_version','parent_record_id','changed_field','source_lineage','review_status','human_reviews'}:raise ValueError('annotation schema fields')
  if r['schema']!='saal-annotation-record-v1' or r['record_id'] in ids or r['review_status']!='PENDING' or r['human_reviews']!=[]:raise ValueError('record identity or invented review')
  if r['split'] not in ('train','validation','test') or r['policy_version']!=VERSION:raise ValueError('split/policy')
  if r['prompt_digest']!=digest(r['prompt']) or r['state_digest']!=digest(r['state']) or decide(r['state'])!=(r['proposed_decision'],r['label_rationale']):raise ValueError('label or content substitution')
  if r['source_lineage']['origin']!='ASSISTANT_AUTHORED_SYNTHETIC' or r['source_lineage']['upstream_dataset']is not None:raise ValueError('native public labels cannot become SAAL labels')
  # Reconstruct the exact untrusted display from its signed state and pair seed.
  pair=r['counterfactual_pair_id'];parts=pair.split('_');ci=int(parts[1][1:]);i=int(parts[2][1:]);offset={'train':0,'validation':10000,'test':20000}[r['split']];n=offset+ci*100+i
  if r['prompt']!=render(r['state'],r['source_lineage']['domain_surface'],n):raise ValueError('text/state mismatch')
  normalized=' '.join(r['prompt'].casefold().split())
  if normalized in prompts:raise ValueError('duplicate prompt')
  prompts.add(normalized)
  for key in ['task_family_id','generation_rule_id','template_id']:
   owner=(key,r[key])
   if owner in owners and owners[owner]!=r['split']:raise ValueError('family/generator split leakage')
   owners[owner]=r['split']
  ids[r['record_id']]=r;splits[r['split']]+=1;labels[r['split']][r['proposed_decision']]+=1
 seen=set()
 for p in corpus['pairs']:
  if p['pair_id'] in seen or len(p['record_ids'])!=2:raise ValueError('duplicate/malformed pair')
  seen.add(p['pair_id']);a,b=[ids[x] for x in p['record_ids']]
  if any(x['counterfactual_pair_id']!=p['pair_id'] or x['split']!=p['split'] or x['task_family_id']!=p['task_family_id'] or x['changed_field']!=p['changed_field'] for x in [a,b]):raise ValueError('pair lineage/split mismatch')
  if a['parent_record_id']is not None or b['parent_record_id']!=a['record_id'] or differences(a['state'],b['state'])!=[p['changed_field']] or a['proposed_decision']==b['proposed_decision']:raise ValueError('material counterfactual change')
  if p['proposed_decisions']!=[a['proposed_decision'],b['proposed_decision']]:raise ValueError('pair label substitution')
 if len(records)!=600 or len(corpus['pairs'])!=300 or len(seen)*2!=len(ids):raise ValueError('pilot inventory')
 if dict(splits)!={'train':360,'validation':120,'test':120}:raise ValueError('split inventory')
 for split,count in splits.items():
  if labels[split]!=collections.Counter({l:count//4 for l in LABELS}):raise ValueError('unbalanced labels')
 families={r['task_family_id']:r['split'] for r in records}
 if corpus['family_split_ownership']!=families:raise ValueError('family partition inventory')
 if corpus['sampling_assumptions']['independence_established']is not False or corpus['sampling_assumptions']['effective_independent_sample_size']is not None:raise ValueError('invented independent sample size')
 return {'records':600,'counterfactual_pairs':300,'families':len(families),'splits':dict(splits),'label_counts':{s:dict(c) for s,c in labels.items()},'mechanical_checks':'PASS','human_review':'PENDING','semantic_leakage_review':'PENDING','qualification_status':'NOT_EVALUATED','training_ready':False}
