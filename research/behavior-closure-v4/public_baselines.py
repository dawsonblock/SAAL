"""Fixed historical lexical and training-majority controls, never tuning labels."""
from pathlib import Path
import json,sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];PROJECT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.benchmarks import lexical_shortcut
from saal_rsi.checkpoint_selection import metrics
from saal_rsi.behavior_screen import screen
from saal_rsi.spec import canonical,sha


def report(dataset):
    labels=('CHEAP','FAST','ASK','REFUSE')
    counts=Counter(r['expected_decision'] for r in dataset['records'] if r['split']=='train')
    if set(counts)!=set(labels):raise ValueError('baseline training class coverage')
    # Declared fixed tie order, based exclusively on the training population.
    majority=max(labels,key=lambda label:counts[label]);result={}
    parent=json.loads((ROOT/'research/learning-reliability-v2/corpus/dataset.json').read_text())
    previous={r['record_id'] for r in parent['records']}
    for split in ('development','regression'):
        rows=[r for r in dataset['records'] if r['split']==split]
        result[split]={}
        for population,selected in [('all',rows),('original',[r for r in rows if r['record_id'] in previous]),('added',[r for r in rows if r['record_id'] not in previous])]:
            if not selected:continue
            result[split][population]={}
            for name,decisions in [('historical_first_keyword',[lexical_shortcut(r['prompt']) for r in selected]),('training_majority',[majority]*len(selected))]:
                predictions=[{'decision':d,'probabilities':[float(d==label) for label in labels]} for d in decisions]
                measured=metrics(selected,predictions,'modernbert-seqcls-v1')
                result[split][population][name]={'records':len(selected),'metrics':measured,'screen':screen(measured)}
    return {'schema':'saal-v4-fixed-public-baselines-v1','dataset_digest':sha(dataset),'training_label_counts':dict(counts),
            'majority_label':majority,'majority_tie_order':list(labels),'measurements':result,
            'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False,
            'limits':'Unchanged historical lexical rule and fixed training-majority control. One-hot scores are not calibrated confidence. Correlated, exposed diagnostics only; not an independent population estimate.'}


if __name__=='__main__':
    data=json.loads((PROJECT/'dataset.json').read_text())
    with Path(sys.argv[1]).open('xb') as f:f.write(canonical(report(data)))
