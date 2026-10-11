"""Reconstruct public summaries from complete authenticated artifact directories.

Heavy full-tuning bundles may be supplied separately. No weights are fetched and
no evaluation receipt, qualification signature or deployment approval is issued.
"""
from pathlib import Path
import argparse,json,statistics,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from report_pretrained_release import build
from saal_rsi.spec import canonical,parse_unique_json,sha
from saal_rsi.continual_learning import retention_metrics


def summarize(reports):
    runs=[row for report in reports for row in report['runs']]
    if len({r['job_digest'] for r in runs})!=len(runs):raise ValueError('duplicate signed run in aggregate')
    groups={}
    for r in runs:
        raw_cohort=r['cohort']
        if raw_cohort.endswith('artifacts-v5'):cohort='modernbert-five-seed-confirmation'
        elif r['repository'].startswith('Qwen/') and raw_cohort.endswith(('artifacts-v4','artifacts-v8')):cohort='qwen-fixed-protocol-public-replication'
        else:cohort='exploratory-'+raw_cohort
        key=f"{cohort} :: {r['repository']} / {r['method']} / {r['steps']} steps / {r['dataset_digest']}"
        groups.setdefault(key,[]).append(r)
    summary={}
    for key,rr in groups.items():
        def distribution(values):return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None,'minimum':min(values),'maximum':max(values)}
        summary[key]={'cohort':key.split(' :: ')[0],'repository':rr[0]['repository'],'method':rr[0]['method'],'steps':rr[0]['steps'],'n':len(rr),'seeds':[r['seed'] for r in rr],'initial_accuracy':distribution([r['initial']['development']['accuracy'] for r in rr]),'selected_accuracy':distribution([r['best']['development']['accuracy'] for r in rr]),'terminal_accuracy':distribution([r['terminal']['development']['accuracy'] for r in rr]),'terminal_family_gain':distribution([r['gain']['mean_gain'] for r in rr]),'step_zero_selected':sum(r['selected_step']==0 for r in rr),'terminal_unauthorized_proposals':[r['terminal']['development']['unauthorized_proposals'] for r in rr],'selected_unauthorized_proposals':[r['best']['development']['unauthorized_proposals'] for r in rr],'terminal_all_classes_correctly_represented':all(all(v>0 for v in r['terminal']['development']['class_accuracy'].values()) for r in rr)}
    return {'schema':'saal-pretrained-release-research-v1','runs':runs,'distributions':summary,'research_qualification':'NOT_EVALUATED','independent_generalization':'NOT_ESTABLISHED','recursive_improvement':'NOT_TESTED','external_deployment_allowed':False,'limits':'Public correlated diagnostics, exploratory runs and later fixed-protocol replications. Do not pool different step budgets or dataset digests, select seeds using these results for protected evaluation, or infer independent qualification from descriptive intervals.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',nargs='+',required=True);p.add_argument('--pin',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=summarize([build(path,a.pin) for path in a.artifacts]);Path(a.out).write_bytes(canonical(result));print('Authenticated runs:',len(result['runs']))

if __name__=='__main__':main()
