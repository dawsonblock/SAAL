"""Field-level and task-level oracle measurements; no learned reward proxy."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.spec import parse_unique_json,canonical
from saal_rsi.task_oracles import score_output
from saal_rsi.generative_diagnostics import decision_diagnostics

def diagnostics(bundle):
    root=Path(bundle);job=parse_unique_json((root/'signed_training_job.json').read_bytes())['payload'];result=parse_unique_json((root/'result.json').read_bytes());report={}
    for name,snapshot in result['snapshots'].items():
        report[name]={}
        for split,pred in snapshot['predictions'].items():
            records=[r for r in job['dataset']['records'] if r['split']==split];groups={}
            for r,p in zip(records,pred):groups.setdefault(r['task_type'],[]).append(score_output(p,r))
            report[name][split]={k:{'exact_accuracy':sum(v['correct'] for v in values)/len(values),'valid_schema_rate':sum(v['valid'] for v in values)/len(values),'field_precision':sum(v['field_correct'] for v in values)/sum(v['field_count'] for v in values),'field_recall':sum(v['field_correct'] for v in values)/sum(v['field_count'] for v in values),'unauthorized_proposals':sum(v['unauthorized_proposal'] for v in values)} for k,values in groups.items()}
            report[name][split]['decision_behavior']=decision_diagnostics(records,pred)
    return {'schema':'saal-generative-public-diagnostics-v1','measurements':report,'limitations':'Schema requires every field; invalid or extra-field outputs count as zero correct fields. Precision/recall use exact field values; public operator-authored task recipes do not establish independent generalization.'}

if __name__=='__main__':Path(sys.argv[2]).write_bytes(canonical(diagnostics(sys.argv[1])))
