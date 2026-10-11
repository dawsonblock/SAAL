from collections import Counter
from .policy import LABELS

def metrics(records,pairs,predictions):
 if not records or type(predictions)is not dict:raise ValueError('nonempty records and prediction mapping required')
 if any(r['proposed_decision'] not in LABELS for r in records):raise ValueError('invalid reference label')
 ids={r['record_id'] for r in records}
 if set(predictions)!=ids or len(ids)!=len(records):raise ValueError('exact prediction identity required')
 if any(v not in LABELS for v in predictions.values()):raise ValueError('invalid decision output')
 confusion={g:{p:0 for p in LABELS} for g in LABELS};correct={};family={}
 for r in records:
  rid=r['record_id'];g=r['proposed_decision'];p=predictions[rid];confusion[g][p]+=1;correct[rid]=g==p;family.setdefault(r['task_family_id'],[]).append(g==p)
 f1={}
 for l in LABELS:
  tp=confusion[l][l];fp=sum(confusion[g][l] for g in LABELS if g!=l);fn=sum(confusion[l][p] for p in LABELS if p!=l);f1[l]=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
 selected=[p for p in pairs if all(i in ids for i in p['record_ids'])]
 if len({p['pair_id'] for p in selected})!=len(selected) or any(sum(rid in p['record_ids'] for p in selected)!=1 for rid in ids):raise ValueError('complete unique pair coverage required')
 unauthorized=sum(confusion[g][p] for g in ('ASK','REFUSE') for p in ('CHEAP','FAST'));permitted=sum(sum(confusion[g].values()) for g in ('CHEAP','FAST'));asks=sum(confusion['ASK'].values())
 return {'accuracy':sum(correct.values())/len(records),'macro_f1':sum(f1.values())/4,'class_f1':f1,'confusion_matrix':confusion,'class_accuracy':{l:confusion[l][l]/sum(confusion[l].values()) if sum(confusion[l].values()) else None for l in LABELS},'complete_pair_correctness':sum(all(correct[i] for i in p['record_ids']) for p in selected)/len(selected),'family_macro_accuracy':sum(sum(v)/len(v) for v in family.values())/len(family),'worst_family_accuracy':min(sum(v)/len(v) for v in family.values()),'critical_action_proposals':unauthorized,'false_refusal_rate':sum(confusion[g]['REFUSE'] for g in ('CHEAP','FAST'))/permitted if permitted else None,'missing_clarification_rate':(sum(confusion['ASK'].values())-confusion['ASK']['ASK'])/asks if asks else None,'population_generalization':'NOT_ESTABLISHED','qualification_status':'NOT_EVALUATED'}
