"""Rebuild factual public summaries from authenticated completed bundles."""
from pathlib import Path
import json,sys,statistics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from saal_rsi.spec import canonical,parse_unique_json,sha
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_artifacts import validate_bundle


def build(artifacts,pin):
    rows=[]
    for root in sorted(Path(artifacts).iterdir()):
        if not root.is_dir() or not (root/'result.json').exists():continue
        e=parse_unique_json((root/'signed_training_job.json').read_bytes());j=verify(e,pin);r=validate_bundle(root,j,pin)
        rows.append({'cohort':Path(artifacts).name,'worker_identity':j['worker_identity'],'run':root.name,'repository':j['model_spec']['model_repository'],'method':j['model_spec']['adapter_type'],'seed':j['seed'],'dataset_digest':j['dataset_digest'],'steps':j['steps'],'learning_rate':j['learning_rate'],'job_digest':sha(j),'model_identity':r['model_identity'],'snapshot_hashes':parse_unique_json((root/'artifact_inventory.json').read_bytes())['files'],
           'initial':r['snapshots']['initial']['metrics'],'best':r['snapshots']['best']['metrics'],'terminal':r['snapshots']['terminal']['metrics'],'selected_step':r['snapshots']['best']['step'],'gain':r['adaptation_gain_terminal'],'learning_curve':r['selection_history'],'resource_accounting':r['resource_accounting'],'reload_parity':r['reload_parity'],'frozen_integrity':r['frozen_integrity'],'qualification_status':r['qualification_status']})
    groups={}
    for row in rows:
        key=row['repository']+' / '+row['method']+' / steps='+str(row['steps'])
        groups.setdefault(key,[]).append(row)
    distributions={}
    for key,rr in groups.items():
        scores=[r['best']['development']['accuracy'] for r in rr];gains=[r['gain']['mean_gain'] for r in rr]
        distributions[key]={'seeds':[r['seed'] for r in rr],'n':len(rr),'mean_best_accuracy':statistics.mean(scores),'sample_sd_best_accuracy':statistics.stdev(scores) if len(scores)>1 else None,'range_best_accuracy':[min(scores),max(scores)],'mean_terminal_family_gain':statistics.mean(gains),'mean_counterfactual_pair_accuracy':statistics.mean([r['best']['development']['complete_pair_accuracy'] for r in rr if r['best']['development']['complete_pair_accuracy'] is not None]) if any(r['best']['development']['complete_pair_accuracy'] is not None for r in rr) else None,'all_classes_correctly_represented':all(all(v>0 for v in r['best']['development']['class_accuracy'].values()) for r in rr),'zero_unauthorized_proposals':all(r['best']['development']['unauthorized_proposals']==0 for r in rr)}
    return {'schema':'saal-pretrained-public-summary-v1','runs':rows,'distributions':distributions,'qualification_status':'NOT_EVALUATED','independent_generalization':'NOT_ESTABLISHED','recursive_improvement':'NOT_TESTED','external_deployment_allowed':False,'limitations':'Operator-authored public diagnostics with shared recipe/oracle lineage. No independent custody, sealed final evaluation or production authority. Descriptive seed distributions do not repair task-family dependence.'}

if __name__=='__main__':
    p=Path(sys.argv[3]);p.write_bytes(canonical(build(sys.argv[1],sys.argv[2])))
