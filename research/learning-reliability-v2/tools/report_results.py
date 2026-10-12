"""Post-training public measurements: never selection, qualification or promotion."""
from pathlib import Path
import collections, json, sys
sys.path.insert(0, str(Path(__file__).parent))
from public_controls import UnigramControl
from saal_rsi.checkpoint_selection import metrics, paired_family_gain
from saal_rsi.learning_audit import load_bundle, audit_bundle, compare_retention, public_screen
from saal_rsi.model_contract import verify
from saal_rsi.spec import sha
from saal_rsi.decision_guard import preflight

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
LABELS = ('CHEAP', 'FAST', 'ASK', 'REFUSE')

def enriched(records, outputs):
    m = metrics(records, outputs, 'modernbert-seqcls-v1')
    f1 = {}
    for label in LABELS:
        c = m['confusion']; tp = c.get(label+'->'+label, 0)
        pred = sum(v for k,v in c.items() if k.endswith('->'+label))
        gold = sum(v for k,v in c.items() if k.startswith(label+'->'))
        f1[label] = 2*tp/(pred+gold) if pred+gold else 0
    return m | {'records':len(records), 'macro_f1':sum(f1.values())/4, 'class_f1':f1}

def strata(record):
    rid = record['record_id']
    return 'compact_v2' if rid.startswith('compact_v2_') else 'prior_diverse_v1' if rid.startswith('diverse_') else 'legacy_exposed'

def report():
    experiment = ROOT/'experiment'; pin = (experiment/'controller.pub').read_text().strip()
    protocol = verify(json.loads((experiment/'signed_protocol.json').read_text()), pin)
    outcomes = json.loads((experiment/'colab/outcomes.json').read_text())['runs']
    audit = json.loads((ROOT/'corpus/state-audit.json').read_text())
    state_by_id = {r['record_id']:r['state'] for r in audit['compact_states']}
    parent = REPO/'research/diverse-learning-v1/experiment/artifacts'/protocol['retention_reference']['run']
    parent_pin = protocol['retention_reference']['controller_pin']
    pjob, presult = load_bundle(parent, parent_pin)
    if sha(pjob) != protocol['retention_reference']['job_digest']:raise ValueError('retention reference substitution')
    expected_exposure=json.loads((ROOT/'EXPECTED_EXPOSURE.json').read_text())['expected']
    reports=[]; errors=[]; controls={}
    for index, run in enumerate(outcomes):
        config=protocol['runs'][index]
        if run['job_digest'] != protocol['job_digests'][index]:raise ValueError('cross-job outcome')
        if run['state'] != 'COMPLETED':
            reports.append(config|{'state':run['state'],'error':run.get('error')});continue
        bundle=experiment/'artifacts'/run['run']; job,result=load_bundle(bundle,pin)
        if sha(job)!=run['job_digest']:raise ValueError('cross-job bundle')
        parts={split:[r for r in job['dataset']['records'] if r['split']==split] for split in ('train','development','regression')}
        snapshots={}
        for name,snap in result['snapshots'].items():
            measures={};subsets={};guard={}
            for split in ('development','regression'):
                records=parts[split];outputs=snap['predictions'][split]
                measures[split]=enriched(records,outputs);subsets[split]={}
                for group in ('compact_v2','prior_diverse_v1','legacy_exposed'):
                    pairs=[(r,p) for r,p in zip(records,outputs) if strata(r)==group]
                    if pairs:subsets[split][group]=enriched([r for r,p in pairs],[p for r,p in pairs])
                for r,p in zip(records,outputs):
                    if p['decision']!=r['expected_decision']:
                        errors.append({'run':config['name'],'snapshot':name,'step':snap['step'],'split':split,'source':strata(r),'record_id':r['record_id'],'family':r['task_family_id'],'prompt':r['prompt'],'expected':r['expected_decision'],'prediction':p,'critical_proposal':r['expected_decision'] in ('ASK','REFUSE') and p['decision'] in ('CHEAP','FAST')})
                permitted=blocked=raw_wrong_actions=blocked_wrong=0;available=0
                for r,p in zip(records,outputs):
                    state=state_by_id.get(r['record_id'])
                    if state is None:continue
                    available+=1;decision=preflight(p,state,now=state['authorization']['now'])
                    allowed=decision['constraints_satisfied']
                    if p['decision'] in ('CHEAP','FAST'):
                        permitted+=allowed;blocked+=not allowed
                        wrong=p['decision']!=r['expected_decision'];raw_wrong_actions+=wrong;blocked_wrong+=wrong and not allowed
                guard[split]={'state_context_available':available,'total_records':len(records),'raw_wrong_action_proposals':raw_wrong_actions,'wrong_action_proposals_blocked':blocked_wrong,'feasible_correct_action_proposals':permitted,'all_action_proposals_blocked':blocked,'scope':'Diagnostic guard replay only on fresh compact cases with trusted state audits. No capability is issued or executed; raw neural errors remain errors.'}
            snapshots[name]={'step':snap['step'],'metrics':measures,'subsets':subsets,'screens':{s:public_screen(m) for s,m in measures.items()},'guard_diagnostics':guard}
        retention=compare_retention(parent,parent_pin,bundle,pin,max_family_drop=protocol['max_public_family_retention_drop'])
        reports.append(config|{'run':run['run'],'state':'COMPLETED','job_digest':sha(job),'snapshots':snapshots,'learning_audit':audit_bundle(bundle,pin),'retention':retention,'public_family_gain':{s:paired_family_gain(snapshots['initial']['metrics'][s],snapshots['terminal']['metrics'][s]) for s in ('development','regression')},'resource_accounting':result['resource_accounting'],'expected_exposure':expected_exposure[config['name']],'worker_token_total_matches_reconstruction':result['resource_accounting']['tokens']==expected_exposure[config['name']]['training_tokens'],'runtime':result['runtime'],'frozen_integrity':result['frozen_integrity'],'fresh_process_gpu_replay':run['fresh_process_replay']})
        if not controls:
            fit=UnigramControl(parts['train']);majority=collections.Counter(r['expected_decision'] for r in parts['train']).most_common(1)[0][0]
            for split in ('development','regression'):
                controls[split]={}
                for title,predictions in [('training_only_unigram',[fit.predict(r['prompt']) for r in parts[split]]),('training_majority',[{'decision':majority,'probabilities':[float(d==majority) for d in LABELS]} for r in parts[split]])]:
                    controls[split][title]={'metrics':enriched(parts[split],predictions),'subsets':{}}
                    for group in ('compact_v2','prior_diverse_v1','legacy_exposed'):
                        pairs=[(r,p) for r,p in zip(parts[split],predictions) if strata(r)==group]
                        if pairs:controls[split][title]['subsets'][group]=enriched([r for r,p in pairs],[p for r,p in pairs])
    data={'schema':'saal-public-reliability-results-v2','runs':reports,'controls':controls,'attempts_completed':len(outcomes),'planned_runs':len(protocol['runs']),'human_reviews':0,'qualification_status':'NOT_EVALUATED','independent_generalization':'NOT_ESTABLISHED','external_deployment_allowed':False,'limits':'Shared-author/recipe public diagnostics and failure-informed training; common old evaluation population preserved. v1 used 600 steps, v2 1000 with mixed training data: changes confounded. Two LoRA seeds, one full-tuning seed. Postprocessing does not select protected-evaluation candidates or alter admission thresholds.'}
    (ROOT/'RESULTS.json').write_text(json.dumps(data,sort_keys=True,indent=2))
    (ROOT/'RAW_ERRORS.json').write_text(json.dumps({'errors':errors,'raw_scoring_unchanged':True},sort_keys=True,indent=2))
    for r in reports:
        print(r['name'],r['state'],{s:{k:m[k] for k in ('accuracy','complete_pair_accuracy','worst_family_accuracy','unauthorized_proposals')} for s,m in r.get('snapshots',{}).get('terminal',{}).get('metrics',{}).items()})
    return data
if __name__=='__main__':report()
