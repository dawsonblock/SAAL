"""Reporting must retain raw errors even when trusted policy blocks an action."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from report_results import enriched, strata
from saal_rsi.reliability_corpus import compact_population
from saal_rsi.decision_guard import preflight
from saal_rsi.learning_audit import public_screen
from verify_capsule import verify_recorded_replay
import copy
import pytest

def prediction(label):
    return {'decision':label,'probabilities':[float(d==label) for d in ('CHEAP','FAST','ASK','REFUSE')]}

def test_reporting_retains_critical_errors_and_macro_f1():
    records=[{'expected_decision':d,'task_family_id':'f','category':'ordinary','cluster_id':'c'+str(i//2)} for i,d in enumerate(('CHEAP','FAST','ASK','REFUSE'))]
    m=enriched(records,[prediction(d) for d in ('CHEAP','FAST','CHEAP','CHEAP')])
    assert m['accuracy']==.5 and m['unauthorized_proposals']==2
    assert m['macro_f1']==.375 and m['complete_pair_accuracy']==.5
    assert not public_screen(m)['passes_public_screen']

def test_subsets_preserve_legacy_cases():
    assert strata({'record_id':'compact_v2_scope_c0_a'})=='compact_v2'
    assert strata({'record_id':'diverse_scope_c0_a'})=='prior_diverse_v1'
    assert strata({'record_id':'legacy_0'})=='legacy_exposed'

def test_guard_is_not_used_to_rescore_bad_predictions():
    data,audit=compact_population();by_id={r['record_id']:r for r in data['records']}
    cases=[a for a in audit if by_id[a['record_id']]['split']=='development' and by_id[a['record_id']]['expected_decision'] in ('ASK','REFUSE')]
    records=[by_id[a['record_id']] for a in cases];proposals=[prediction('CHEAP') for a in cases]
    raw=enriched(records,proposals)
    assert raw['accuracy']==0 and raw['unauthorized_proposals']==len(cases)
    assert all(not preflight(p,a['state'],now=a['state']['authorization']['now'])['constraints_satisfied'] for p,a in zip(proposals,cases))
    assert enriched(records,proposals)==raw

def test_negative_cpu_replay_is_preserved_but_forged_pass_rejected():
    original={'decision':'CHEAP','probabilities':[.6,.2,.1,.1]}
    actual={'decision':'CHEAP','probabilities':[.6002,.1998,.1,.1]}
    result={'snapshots':{n:{'predictions':{s:[original] for s in ('development','regression')}} for n in ('initial','best','terminal')}}
    delta=max(abs(a-b) for a,b in zip(actual['probabilities'],original['probabilities']))
    replay={'job_digest':'fixture','probability_tolerance':1e-4,'all_predictions_agree':False,'measurements':[{'checkpoint':n,'split':s,'records':1,'predictions':[actual],'prediction_agreement':False,'max_probability_difference':delta} for n in ('initial','best','terminal') for s in ('development','regression')]}
    verified=verify_recorded_replay(replay,result,'fixture')
    assert verified['all_decisions_match'] and not verified['passes_registered_probability_tolerance']
    tampered=copy.deepcopy(replay);tampered['all_predictions_agree']=True
    with pytest.raises(ValueError):verify_recorded_replay(tampered,result,'fixture')
    tampered=copy.deepcopy(replay);tampered['measurements'].pop()
    with pytest.raises(ValueError):verify_recorded_replay(tampered,result,'fixture')
