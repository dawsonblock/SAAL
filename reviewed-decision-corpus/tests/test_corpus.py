import copy,csv,json
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_corpus.build import build,state
from saal_corpus.integrity import validate
from saal_corpus.policy import decide,differences,digest,canonical
from saal_corpus.review import assess,sign_review
from saal_corpus.metrics import metrics
@pytest.fixture(scope='module')
def corpus(tmp_path_factory):return build(tmp_path_factory.mktemp('corpus'))

def test_balanced_600_and_material_pairs(corpus):
 v=validate(corpus);assert v['records']==600 and v['counterfactual_pairs']==300 and v['families']==7 and not v['training_ready']

def test_preference_has_to_respect_actual_constraints():
 s=state(0);s['priority']='duration';s['budget']={'required':True,'limit':10};assert decide(s)[0]=='CHEAP'
 s=state(0);s['deadline']={'required':True,'limit':20};assert decide(s)[0]=='FAST'

@pytest.mark.parametrize('mutation',['scope','revocation','quote','expiry'])
def test_authorization_precedes_missing_information(mutation):
 s=state(0);s['priority']=None
 if mutation=='scope':s['authorization']['requested_scope']='other'
 elif mutation=='revocation':s['authorization']['granted']=False
 elif mutation=='quote':s['authorization']['source']='quoted'
 else:s['authorization']['now']=s['authorization']['expires']
 assert decide(s)[0]=='REFUSE'

def test_missing_limit_and_no_feasible_option_are_distinct():
 s=state(0);s['budget']={'required':True,'limit':None};assert decide(s)[0]=='ASK'
 s['deadline']={'required':True,'limit':0};assert decide(s)[0]=='REFUSE'

@pytest.mark.parametrize('mutation',['label','prompt','pair','review','public_conversion','qualification','family','independence'])
def test_invalid_corpus_rejected(corpus,mutation):
 c=copy.deepcopy(corpus);r=c['records'][0]
 if mutation=='label':r['proposed_decision']='REFUSE'
 elif mutation=='prompt':r['prompt']='changed';r['prompt_digest']=digest(r['prompt'])
 elif mutation=='pair':c['pairs'][0]['changed_field']='authorization.source'
 elif mutation=='review':r['human_reviews']=[{'human':'invented'}]
 elif mutation=='public_conversion':r['source_lineage']['upstream_dataset']='ANLI'
 elif mutation=='qualification':c['training_ready']=True
 elif mutation=='family':c['family_split_ownership'][r['task_family_id']]='test'
 else:c['sampling_assumptions']['independence_established']=True
 with pytest.raises(ValueError):validate(c)

def test_plain_candidate_cannot_be_admitted(corpus):
 policy={'schema':'saal-reviewer-policy-v1','reviewers':{'one':None,'two':None},'adjudicator':None}
 with pytest.raises(ValueError):assess(corpus,policy,[])

def signed(body,key):return {'payload':body,'signature':key.sign(canonical(body)).hex()}
@pytest.fixture
def reviews(corpus,tmp_path):
 # Ephemeral test keys/answers are synthetic fixtures, never human evidence.
 keys=[Ed25519PrivateKey.generate() for _ in range(3)];names=['one','two'];policy={'schema':'saal-reviewer-policy-v1','reviewers':{n:k.public_key().public_bytes_raw().hex() for n,k in zip(names,keys)},'adjudicator':keys[2].public_key().public_bytes_raw().hex()};envelopes=[]
 for name,key in zip(names,keys):
  p=tmp_path/(name+'.csv')
  with p.open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['record_id','prompt','decision','rationale']);w.writerows([r['record_id'],r['prompt'],r['proposed_decision'],'Synthetic test answer, not human review.'] for r in corpus['records'])
  envelopes.append(sign_review(corpus,p,name,key))
 from saal_corpus.leakage import screen as lexical_screen
 lexical=lexical_screen(corpus)
 screen=signed({'lexical_screen_digest':digest(lexical),'family_pair_dispositions':{g:{'decision':'ACCEPT_PUBLIC_SHARED_POLICY','rationale':'Synthetic fixture only; not independent evidence'} for g in lexical['family_pair_groups']},'schema':'saal-semantic-split-review-v1','corpus_digest':digest(corpus),'approved':True,'rationale':'Synthetic test fixture only','independence_claim':False},keys[2]);return keys,policy,envelopes,screen

def test_signed_quorum_remains_unqualified(corpus,reviews):
 _,p,e,s=reviews;r=assess(corpus,p,e,semantic_review=s);assert r['eligible_for_separate_training_admission'] and not r['external_deployment_allowed'] and r['qualification_status']=='NOT_EVALUATED'

@pytest.mark.parametrize('mutation',['missing','same_key','forgery','stale','disagreement','semantic_missing','independence'])
def test_review_bypass_rejected(corpus,reviews,mutation):
 keys,p,e,s=reviews;p=copy.deepcopy(p);e=copy.deepcopy(e);s=copy.deepcopy(s)
 if mutation=='missing':e=e[:1]
 elif mutation=='same_key':p['reviewers']['two']=p['reviewers']['one']
 elif mutation=='forgery':e[0]['signature']='00'*64
 elif mutation=='stale':e[0]['payload']['corpus_digest']='sha256:'+'0'*64;e[0]=signed(e[0]['payload'],keys[0])
 elif mutation=='disagreement':e[1]['payload']['decisions'][corpus['records'][0]['record_id']]['decision']='REFUSE';e[1]=signed(e[1]['payload'],keys[1])
 elif mutation=='semantic_missing':s=None
 else:s['payload']['independence_claim']=True;s=signed(s['payload'],keys[2])
 with pytest.raises(ValueError):assess(corpus,p,e,semantic_review=s)

def test_human_oracle_disagreement_requires_new_epoch(corpus,reviews):
 keys,p,e,s=reviews;e=copy.deepcopy(e);rid=corpus['records'][0]['record_id']
 for i in [0,1]:e[i]['payload']['decisions'][rid]['decision']='REFUSE';e[i]=signed(e[i]['payload'],keys[i])
 with pytest.raises(ValueError,match='new corpus version'):assess(corpus,p,e,semantic_review=s)

def test_partial_or_blank_csv_cannot_be_signed(corpus,tmp_path):
 p=tmp_path/'review.csv';p.write_text('record_id,prompt,decision,rationale\n')
 with pytest.raises(ValueError):sign_review(corpus,p,'one',Ed25519PrivateKey.generate())

def test_exact_behavior_metrics(corpus):
 rows=corpus['records'][:2];pairs=corpus['pairs'][:1];pred={r['record_id']:r['proposed_decision'] for r in rows};m=metrics(rows,pairs,pred);assert m['accuracy']==1 and m['complete_pair_correctness']==1 and m['macro_f1']==.5
 pred[rows[1]['record_id']]='CHEAP';m=metrics(rows,pairs,pred);assert m['accuracy']==.5 and m['complete_pair_correctness']==0

@pytest.mark.parametrize('mutation',['missing','unknown','missing_pair'])
def test_invalid_prediction_or_pair_evidence_fails(corpus,mutation):
 rows=corpus['records'][:2];pairs=corpus['pairs'][:1];pred={r['record_id']:r['proposed_decision'] for r in rows}
 if mutation=='missing':pred.pop(rows[0]['record_id'])
 elif mutation=='unknown':pred[rows[0]['record_id']]='FORBIDDEN'
 else:pairs=[]
 with pytest.raises(ValueError):metrics(rows,pairs,pred)

def test_frozen_corpus_cannot_be_overwritten(tmp_path):
 build(tmp_path)
 with pytest.raises(ValueError,match='overwritten'):build(tmp_path)

def test_annotation_unknown_fields_fail(corpus):
 c=copy.deepcopy(corpus);c['records'][0]['independently_reviewed']=True
 with pytest.raises(ValueError,match='schema fields'):validate(c)

def test_empty_evaluation_cannot_pass():
 with pytest.raises(ValueError):metrics([],[],{})

def test_disagreement_requires_signed_reasoned_adjudication(corpus,reviews,tmp_path):
 keys,p,e,s=reviews;e=copy.deepcopy(e);rid=corpus['records'][0]['record_id'];e[1]['payload']['decisions'][rid]['decision']='REFUSE';e[1]=signed(e[1]['payload'],keys[1])
 path=tmp_path/'adjudication.csv'
 with path.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['record_id','prompt','decision','rationale']);w.writerows([r['record_id'],r['prompt'],r['proposed_decision'],'Synthetic adjudication: permission valid; select cheapest feasible option.'] for r in corpus['records'])
 adj=sign_review(corpus,path,'curator',keys[2],'adjudicator');result=assess(corpus,p,e,adjudication=adj,semantic_review=s);assert result['disagreements']==[rid] and result['labels'][rid]=='CHEAP'

def test_native_bridge_requires_review_and_keeps_custody_conservative(corpus,reviews):
 from saal_corpus.export import prepare
 _,p,e,s=reviews;bundle=prepare(corpus,p,e,semantic_review=s);assert bundle['dataset']['independent_population']is False
 assert {r['split'] for r in bundle['dataset']['records']}=={'train','development','regression'}
 assert bundle['requires_separate_signed_model_job'] and not bundle['external_deployment_allowed']
 with pytest.raises(ValueError):prepare(corpus,p,[],semantic_review=s)

@pytest.mark.parametrize('mutation',['screen_digest','unresolved_group'])
def test_semantic_screen_substitution_or_incomplete_resolution_fails(corpus,reviews,mutation):
 keys,p,e,s=reviews;s=copy.deepcopy(s)
 if mutation=='screen_digest':s['payload']['lexical_screen_digest']='sha256:'+'0'*64
 else:s['payload']['family_pair_dispositions'].pop(next(iter(s['payload']['family_pair_dispositions'])))
 s=signed(s['payload'],keys[2])
 with pytest.raises(ValueError):assess(corpus,p,e,semantic_review=s)

def test_signed_rejection_is_preserved_and_cannot_be_admitted(corpus,reviews):
 keys,p,e,s=reviews;e=copy.deepcopy(e);rid=corpus['records'][0]['record_id'];e[0]['payload']['decisions'][rid]={'decision':'REJECT','rationale':'Synthetic fixture: wording is ambiguous; repair required'};e[0]=signed(e[0]['payload'],keys[0])
 with pytest.raises(ValueError,match='rejected record'):assess(corpus,p,e,semantic_review=s)
