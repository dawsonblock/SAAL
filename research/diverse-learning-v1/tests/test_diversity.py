import copy,collections,re
import pytest
from saal_diverse.dataset import build,validate,contextual_pair,render,TRAIN,DEV,REG
from saal_diverse.policy import differences,decide
from saal_rsi.learning_dataset import validate_dataset
@pytest.fixture(scope='module')
def data():return build()
def test_balance_lineage_and_single_condition_pairs(data):
 d,a=data;v=validate(d,a);assert v['records']==2304 and v['pairs']==1152
 assert all(len(set(x.values()))==1 for x in v['split_label_counts'].values())
 assert not v['independence_established'] and v['human_reviews']==0
 assert set(TRAIN).isdisjoint(DEV) and set(TRAIN).isdisjoint(REG) and set(DEV).isdisjoint(REG)

def test_quotation_values_balanced_within_every_class(data):
 d,a=data;lookup={x['record_id']:x for x in a['records']}
 for split in ['train','development','regression']:
  for label in ['CHEAP','FAST','ASK','REFUSE']:
   counts=collections.Counter((lookup[r['record_id']]['quote_goal'],lookup[r['record_id']]['quote_consent']) for r in d['records'] if r['split']==split and r['expected_decision']==label)
   assert len(counts)==4 and len(set(counts.values()))==1

@pytest.mark.parametrize('recipe',['active_archived_entries','terms_assignment','priority_register','unfamiliar_glossary'])
def test_order_assignment_negative_control_has_identical_bag_of_words(recipe):
 a,b,_,_=contextual_pair('train',recipe,('CHEAP','FAST'),0)
 def words(s):return collections.Counter(re.findall('[a-z]+|[0-9]+',s.lower()))
 assert words(render(a,recipe,0,'smaller expenditure','active'))==words(render(b,recipe,0,'smaller expenditure','active'))
 assert decide(a)[0]!=decide(b)[0]

@pytest.mark.parametrize('mutation',['label','prompt','state','review','split'])
def test_tamper_and_false_review_cannot_pass(data,mutation):
 d,a=copy.deepcopy(data)
 if mutation=='label':d['records'][0]['expected_decision']='REFUSE'
 elif mutation=='prompt':d['records'][0]['prompt']='replacement'
 elif mutation=='state':a['records'][0]['state']['priority']=None
 elif mutation=='review':d['review_status']='HUMAN_REVIEWED'
 else:d['records'][0]['split']='development'
 with pytest.raises((ValueError,AssertionError)):validate(d,a)

def test_native_data_remains_public_and_unqualified(data):
 d,_=data;validate_dataset(d,'modernbert-seqcls-v1');assert d['independent_population'] is False
 assert d['review_status']=='ORACLE_CHECKED_NOT_INDEPENDENTLY_REVIEWED'
