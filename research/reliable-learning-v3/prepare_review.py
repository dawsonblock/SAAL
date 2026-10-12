"""600 public cases for human review, preserving pairs and original partitions."""
from pathlib import Path
import json,random,collections,itertools,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from saal_rsi.learning_dataset import wrap,diagnostic_audit
from saal_rsi.public_review import packet,html_packet
from saal_rsi.spec import canonical,sha

def prepare():
    source=json.loads((ROOT/'research/learning-reliability-v2/corpus/dataset.json').read_text());audit=json.loads((ROOT/'research/learning-reliability-v2/corpus/state-audit.json').read_text());states={r['record_id']:r for r in audit['compact_states']}
    chosen=[];rng=random.Random(730)
    for split,base in [('train',33),('development',8),('regression',8)]:
        grouped=collections.defaultdict(list)
        for r in source['records']:
            if r['split']==split and r['record_id'].startswith('compact_v2_'):grouped[r['cluster_id']].append(r)
        contrasts=collections.defaultdict(list)
        for pair in grouped.values():
            assert len(pair)==2
            contrasts[tuple(sorted(r['expected_decision'] for r in pair))].append(pair)
        for labels,population in sorted(contrasts.items()):
            rng.shuffle(population);count=base+int(set(labels) in ({'CHEAP','FAST'},{'ASK','REFUSE'}))
            chosen.extend(r for pair in population[:count] for r in pair)
    data=wrap(chosen);report={'parent_dataset_digest':sha(source),'pilot_digest':sha(data),'record_count':len(chosen),'human_reviews':0,'selection':'Fixed seed, complete opposite-label pairs, original splits/families/sources preserved. Public targeted review, not independent evaluation.','class_counts':{s:dict(collections.Counter(r['expected_decision'] for r in chosen if r['split']==s)) for s in ('train','development','regression')},'mechanism_counts':dict(collections.Counter(states[r['record_id']]['changed_field'] for r in chosen)),'pilot_used_for_main_training':False,'qualification_status':'NOT_EVALUATED'}
    assert len(chosen)==600 and report['class_counts']['train']=={d:100 for d in ('CHEAP','FAST','ASK','REFUSE')}
    out=Path(__file__).parent/'review-pilot';out.mkdir(exist_ok=False)
    for name,value in [('dataset.json',data),('selection.json',report),('lexical-screen.json',diagnostic_audit(data)),('blind-review-draft.json',packet(data))]:(out/name).write_bytes(canonical(value))
    (out/'blind-review.html').write_text(html_packet(data));print(json.dumps(report,indent=2))
if __name__=='__main__':prepare()
