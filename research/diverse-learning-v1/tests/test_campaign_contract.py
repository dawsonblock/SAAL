"""Check the frozen comparison and postprocessing, not empirical qualification."""
from pathlib import Path
import json,collections,sys
import pytest
from saal_rsi.model_contract import verify
from saal_rsi.pretrained_contract import validate_job
from saal_rsi.spec import sha
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from report_results import macro_f1,screen
from public_controls import UnigramControl
@pytest.fixture(scope='module')
def jobs():
 p=(ROOT/'experiment/controller.pub').read_text().strip()
 protocol=verify(json.loads((ROOT/'experiment/signed_protocol.json').read_text()),p)
 result=[verify(json.loads((ROOT/'experiment'/(r['name']+'-job.json')).read_text()),p) for r in protocol['runs']]
 assert [sha(j) for j in result]==protocol['job_digests']
 for j in result:validate_job(j,execution=False)
 return protocol,result

def test_identical_evaluation_and_bounded_training(jobs):
 protocol,js=jobs
 split=lambda j,s:[r for r in j['dataset']['records'] if r['split']==s]
 for s in ['development','regression']:
  assert all(split(j,s)==split(js[0],s) for j in js[1:])
 assert len(split(js[0],'train'))==768 and all(len(split(j,'train'))==1536 for j in js[1:])
 assert all(j['steps']==600 and j['batch_size']==8 and j['class_balance'] for j in js)
 assert js[0]['seed']==js[2]['seed']==47 and js[0]['learning_rate']==js[2]['learning_rate']
 assert not protocol['independence_established'] and protocol['human_reviews']==0
 assert protocol['qualification_status']=='NOT_EVALUATED' and not protocol['external_deployment_allowed']

def test_unigram_rejects_eval_leakage(jobs):
 records=jobs[1][0]['dataset']['records']
 with pytest.raises(ValueError,match='training-only'):UnigramControl(records)

def test_metric_postprocessing_does_not_hide_class_failures():
 perfect={d+'->'+d:3 for d in ['CHEAP','FAST','ASK','REFUSE']}
 assert macro_f1({'confusion':perfect})['macro_f1']==1
 one={'CHEAP->REFUSE':3,'FAST->REFUSE':3,'ASK->REFUSE':3,'REFUSE->REFUSE':3}
 assert macro_f1({'confusion':one})['macro_f1']==.1
 m={'complete_pair_accuracy':1,'worst_family_accuracy':1,'class_accuracy':{'CHEAP':1,'FAST':1,'ASK':1,'REFUSE':0},'unauthorized_proposals':0,'invalid_outputs':0}
 assert 'missing_correct_class' in screen(m)['failures']
 m['class_accuracy']['REFUSE']=1;m['unauthorized_proposals']=1
 assert 'critical_or_invalid_proposals' in screen(m)['failures']
 assert screen(m)['qualification_status']=='NOT_EVALUATED'
