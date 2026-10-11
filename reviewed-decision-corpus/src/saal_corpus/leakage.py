import re,collections
from .policy import digest

def screen(corpus):
 records=corpus['records'];tokens={r['record_id']:set(re.findall(r'\w+',r['prompt'].casefold())) for r in records};samples=[];groups=collections.Counter();count=0
 for i,a in enumerate(records):
  for b in records[i+1:]:
   if a['split']==b['split']:continue
   x,y=tokens[a['record_id']],tokens[b['record_id']];score=len(x&y)/len(x|y)
   if score>=.8:
    count+=1;group='__'.join(sorted([a['task_family_id'],b['task_family_id']]));groups[group]+=1
    if len(samples)<100:samples.append({'left':a['record_id'],'right':b['record_id'],'token_jaccard':score})
 return {'schema':'saal-corpus-lexical-screen-v2','corpus_digest':digest(corpus),'algorithm':'casefolded word-token Jaccard; fixed threshold 0.8','cross_split_flag_count':count,'family_pair_groups':dict(groups),'first_100_flags':samples,'semantic_review':'PENDING','independence_established':False,'limits':'Similarity flags are review leads, not proof of leakage or independence. Shared renderer and numeric concepts are disclosed.'}
