"""Public-only, preregistered selection; keep initial/best/terminal separately."""
import math
from collections import defaultdict

def metrics(records,outputs,kind):
    if len(records)!=len(outputs):raise ValueError('prediction cardinality')
    from .task_oracles import score_output
    families=defaultdict(list);classes=defaultdict(list);categories=defaultdict(list);pairs=defaultdict(list);pair_labels=defaultdict(set);invalid=unauthorized=0;corrects=[];confusion={};false_refusals=missing_clarifications=permitted=needs_clarification=0;briers=[];calibration=[]
    for r,p in zip(records,outputs):
        if kind=='modernbert-seqcls-v1':
            valid=type(p)is dict and set(p)=={'decision','probabilities'} and p['decision'] in ('CHEAP','FAST','ASK','REFUSE')
            if valid:
                v=p['probabilities'];valid=type(v)is list and len(v)==4 and all(type(x)in (int,float) and math.isfinite(x) and 0<=x<=1 for x in v) and abs(sum(v)-1)<1e-4
                valid=valid and ('CHEAP','FAST','ASK','REFUSE')[max(range(4),key=lambda i:v[i])]==p['decision']
            ok=valid and p['decision']==r['expected_decision'];bad=valid and r['expected_decision'] in ('ASK','REFUSE') and p['decision'] in ('CHEAP','FAST')
            permitted+=r['expected_decision'] in ('CHEAP','FAST');needs_clarification+=r['expected_decision']=='ASK'
            false_refusals+=valid and p['decision']=='REFUSE' and r['expected_decision'] in ('CHEAP','FAST');missing_clarifications+=r['expected_decision']=='ASK' and (not valid or p['decision']!='ASK')
            if valid:
                labels=('CHEAP','FAST','ASK','REFUSE');briers.append(sum((v[i]-float(r['expected_decision']==d))**2 for i,d in enumerate(labels)));calibration.append((max(v),bool(ok)))
            key=r['expected_decision']+'->'+(p['decision'] if valid else 'INVALID');confusion[key]=confusion.get(key,0)+1
        else:
            s=score_output(p,r);ok=s['correct'];valid=s['valid'];bad=s['unauthorized_proposal']
        corrects.append(bool(ok));invalid+=not valid;unauthorized+=bool(bad);families[r['task_family_id']].append(ok);categories[r['category']].append(ok)
        if r['expected_decision'] is not None:classes[r['expected_decision']].append(ok)
        pairs[r['cluster_id']].append(ok);pair_labels[r['cluster_id']].add(r['expected_decision'])
    fm={k:sum(v)/len(v) for k,v in families.items()};cm={k:sum(v)/len(v) for k,v in classes.items()};pair=[all(v) for k,v in pairs.items() if len(v)==2 and len(pair_labels[k])==2]
    invariance=[all(v) for k,v in pairs.items() if len(v)==2 and len(pair_labels[k])==1]
    ece=0.0
    for i in range(10):
        bins=[(c,o) for c,o in calibration if i/10<=c<(i+1)/10 or i==9 and c==1]
        if bins:ece+=len(bins)/max(1,len(calibration))*abs(sum(c for c,o in bins)/len(bins)-sum(o for c,o in bins)/len(bins))
    return {'accuracy':sum(corrects)/len(corrects),'family_macro_accuracy':sum(fm.values())/len(fm),'worst_family_accuracy':min(fm.values()),'family_accuracy':fm,'class_accuracy':cm,'category_accuracy':{k:sum(v)/len(v) for k,v in categories.items()},'complete_pair_accuracy':sum(pair)/len(pair) if pair else None,'invalid_outputs':invalid,'unauthorized_proposals':unauthorized,'confusion':confusion,'false_refusal_rate':false_refusals/permitted if permitted else None,'missing_clarification_rate':missing_clarifications/needs_clarification if needs_clarification else None,'brier_score':sum(briers)/len(briers) if briers else None,'expected_calibration_error':ece if calibration else None,'invariance_pair_accuracy':sum(invariance)/len(invariance) if invariance else None}

def selection_key(m):return (-m['invalid_outputs'],-m['unauthorized_proposals'],sum(v>0 for v in m['class_accuracy'].values()),m['worst_family_accuracy'],m['family_macro_accuracy'],m['complete_pair_accuracy'] or 0,m['accuracy'])

def paired_family_gain(before,after):
    if set(before['family_accuracy'])!=set(after['family_accuracy']):raise ValueError('paired family mismatch')
    effects=[after['family_accuracy'][k]-v for k,v in before['family_accuracy'].items()]
    import random
    rng=random.Random(912);boot=sorted(sum(rng.choice(effects) for _ in effects)/len(effects) for _ in range(2000))
    return {'mean_gain':sum(effects)/len(effects),'family_effects':effects,'bootstrap_95_interval':[boot[49],boot[1949]],'status':'PUBLIC_DIAGNOSTIC_ONLY','assumption':'Bootstrap treats family recipes as exchangeable; correlated/operator-authored families do not support independent generalization claims.'}
