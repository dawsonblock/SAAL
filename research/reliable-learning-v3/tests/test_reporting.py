import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from replay_results import verified_measurement
from report_results import enriched
from saal_rsi.learning_audit import public_screen

def records():
 return [{'record_id':str(i),'task_family_id':'mechanism','cluster_id':'pair'+str(i//2),'expected_decision':d,'category':'public'} for i,d in enumerate(('CHEAP','FAST','ASK','REFUSE'))]
def pred(d):return {'decision':d,'probabilities':[float(x==d) for x in ('CHEAP','FAST','ASK','REFUSE')]}
def replay_fixture():
 predictions=[pred(r['expected_decision']) for r in records()]
 result={'snapshots':{c:{'predictions':{s:copy.deepcopy(predictions) for s in ('development','regression')}} for c in ('initial','best','terminal')}}
 report={'job_digest':'digest','probability_tolerance':1e-4,'all_predictions_agree':True,'measurements':[{'checkpoint':c,'split':s,'records':4,'prediction_agreement':True,'max_probability_difference':0.0,'predictions':copy.deepcopy(predictions)} for c in ('initial','best','terminal') for s in ('development','regression')]}
 return report,result

def test_all_six_restorations_required():
 r,result=replay_fixture();assert verified_measurement(r,result,'digest')['all_decisions_match']
 r['measurements'].pop()
 with pytest.raises(ValueError):verified_measurement(r,result,'digest')

@pytest.mark.parametrize('mutation',['tolerance','job','duplicate','delta','claim','label'])
def test_forged_cpu_evidence_fails(mutation):
 r,result=replay_fixture()
 if mutation=='tolerance':r['probability_tolerance']=1e-3
 elif mutation=='job':r['job_digest']='substituted'
 elif mutation=='duplicate':r['measurements'][1]=copy.deepcopy(r['measurements'][0])
 elif mutation=='delta':r['measurements'][0]['max_probability_difference']=.1
 elif mutation=='claim':r['all_predictions_agree']=False
 else:r['measurements'][0]['predictions'][0]['decision']='UNKNOWN'
 with pytest.raises(ValueError):verified_measurement(r,result,'digest')

def test_faithful_numerical_failure_is_preserved():
 r,result=replay_fixture();m=r['measurements'][0];m['predictions'][0]['probabilities']=[.999, .001,0,0];m['max_probability_difference']=max(abs(a-b) for a,b in zip(m['predictions'][0]['probabilities'],[1,0,0,0]));m['prediction_agreement']=False;r['all_predictions_agree']=False
 s=verified_measurement(r,result,'digest');assert not s['passes_registered_probability_tolerance'];assert s['all_decisions_match']

def test_raw_unsafe_action_is_not_rescored_as_guard_success():
 rows=records();predictions=[pred(r['expected_decision']) for r in rows];predictions[2]=pred('FAST');m=enriched(rows,predictions)
 assert m['accuracy']==.75 and m['unauthorized_proposals']==1
 assert not public_screen(m)['passes_public_screen']

def test_new_run_never_inherits_prior_run_progress():
 import json
 from monitor_progress import progress
 events=[{'monitor':'campaign_progress','completed_runs':0,'total_runs':11,'seed':47,'mode':'uniform'}, {'monitor':'training_progress','completed_steps':1000,'total_steps':1000,'objective_loss':.1,'tokens':1000000,'gpu_allocated_bytes':700000000}, {'monitor':'campaign_progress','completed_runs':1,'total_runs':11,'seed':47,'mode':'balanced'}]
 result=progress([json.dumps(x) for x in events]);assert result['mode']=='balanced';assert result['step'] is None and result['tokens'] is None and result['loss'] is None
