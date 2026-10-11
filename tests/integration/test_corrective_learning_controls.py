import copy
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.spec import sha
from saal_rsi.model_contract import sign
from saal_rsi.research_selection import select_research_snapshots
from saal_rsi.lineage_review import provenance_screen, protected_fingerprints, embedding_screen, verify_screen_review
from saal_rsi.public_mechanisms import public_mechanism_dataset
from saal_rsi.learning_dataset import validate_dataset
from saal_rsi.calibration_diagnostics import fit_temperature, selective_report
from saal_rsi.replay_sampling import batch_plan, reservoir


def test_research_selection_keeps_terminal_and_safety_failures():
    snapshots=[]
    for step, accuracy, unsafe in ((0,.3,0),(10,.8,8),(20,.7,5)):
        snapshots.append({'step':step,'checkpoint_digest':sha(step),'development':{'family_macro_accuracy':accuracy,'worst_family_accuracy':accuracy-.1,'complete_pair_accuracy':accuracy,'accuracy':accuracy,'invalid_outputs':0,'unauthorized_proposals':unsafe}})
    r=select_research_snapshots(snapshots)
    assert r['best_public_learning']['step']==10 and r['terminal']['step']==20
    assert r['best_public_learning']['development']['unauthorized_proposals']==8
    assert r['qualification_admission']=='NOT_ADMITTED' and r['learning_gain_observed']


def test_unknown_selection_policy_and_duplicate_snapshots_rejected():
    with pytest.raises(ValueError): select_research_snapshots([],{'use_holdout':True})


def test_mechanism_labels_and_shared_generation_review():
    d=public_mechanism_dataset(2);validate_dataset(d,'modernbert-seqcls-v1')
    assert len(d['records'])==72 and len({r['generation_mechanism'] for r in d['records']})==6
    screen=provenance_screen(d['records'])
    assert any(f['reason']=='generation_mechanism_OVERLAP' for f in screen['review_flags'])
    assert screen['qualification_eligible']is False
    for mechanism,a,b in [('budget_condition','FAST','CHEAP'),('authority_provenance','FAST','REFUSE'),('latest_valid_correction','FAST','CHEAP'),('unknown_mapping','CHEAP','FAST'),('missing_numeric_constraint','ASK','FAST'),('scope_revocation','CHEAP','REFUSE')]:
        pair=[r for r in d['records'] if r['cluster_id']==f'development_{mechanism}_0']
        assert [r['expected_decision'] for r in pair]==[a,b]


@pytest.mark.parametrize('field',['prompt','source_digest','template_id','generation_mechanism','parent_record_id'])
def test_protected_derivation_is_denied(field):
    records=public_mechanism_dataset(1)['records'];protected=copy.deepcopy(records[0]);protected['record_id']='protected_id'
    candidate=copy.deepcopy(records[1])
    candidate[field]=protected['record_id'] if field=='parent_record_id' else protected[field]
    index=protected_fingerprints([protected]);r=provenance_screen([candidate],index)
    assert any(x['reason']=='PROTECTED_DERIVATION_OR_MECHANISM' for x in r['denied'])


def test_embedding_similarity_needs_signed_complete_review():
    records=public_mechanism_dataset(1)['records'];selected=[records[0],records[12]]
    report=embedding_screen(selected,[[1.,0.],[1.,.01]],sha('pinned-encoder'))
    assert report['flag_count']==1 and report['independence_established']is False
    key=Ed25519PrivateKey.generate();pin=key.public_key().public_bytes_raw().hex()
    p={'schema':'saal-screen-review-v1','screen_digest':sha(report),'resolutions':{},'sampling_description':'Operator-authored related mechanisms, public diagnostic population only.','scope':'PUBLIC_LINEAGE_REVIEW_ONLY'}
    with pytest.raises(PermissionError):verify_screen_review(sign(p,key),pin,report)
    p['resolutions']={sha(f):'These related public examples share a mechanism and are not independent.' for f in report['review_flags']}
    assert verify_screen_review(sign(p,key),pin,report)['qualification_eligible']is False
    with pytest.raises(ValueError):embedding_screen(selected,[[float('nan'),1.],[1.,0.]],sha('encoder'))


def training(i,family='old'):
    return {'record_id':str(i),'task_family_id':family,'split':'train'}


def test_reservoir_is_not_fixed_prefix_and_repeatable():
    records=[training(i) for i in range(100)]
    pool=reservoir(records,10,17)
    assert pool==reservoir(records,10,17) and any(int(r['record_id'])>=10 for r in pool)
    # Distributional negative control: late records remain eligible over seeds.
    counts={i:0 for i in range(100)}
    for seed in range(300):
        for r in reservoir(records,10,seed):counts[int(r['record_id'])]+=1
    assert min(counts.values())>10 and max(counts.values())<55


def test_matched_current_schedule_and_explicit_replay_ids():
    current=[training(i,'new') for i in range(100,120)]
    previous=[training(i,'old_a' if i<10 else 'old_b') for i in range(20)]
    kw=dict(seed=17,replay_seed=71,stage=1,steps=10,batch_size=8,accumulation=2,replay_ratio=.25,policy='stratified-family-v1')
    plan=batch_plan(current,previous,**kw);other=batch_plan(current,previous,**(kw|{'replay_seed':101}))
    assert [b['current_ids'] for b in plan['batches']]==[b['current_ids'] for b in other['batches']]
    assert [b['replay_ids'] for b in plan['batches']]!=[b['replay_ids'] for b in other['batches']]
    assert all(len(b['current_ids'])==6 and len(b['replay_ids'])==2 for b in plan['batches'])
    assert {rid for b in plan['batches'] for rid in b['replay_ids']}<=set(plan['pool_record_ids'])
    with pytest.raises(ValueError):batch_plan(current,[training(1)|{'split':'development'}],**kw)


def test_public_calibration_and_escalation_are_not_correct_clarifications():
    records=[];outputs=[]
    for label in ('CHEAP','FAST','ASK','REFUSE'):
        records.append({'expected_decision':label,'task_family_id':label,'category':'ordinary','cluster_id':label,'split':'development'})
        outputs.append({'decision':'CHEAP','probabilities':[.4,.2,.2,.2]})
    fit=fit_temperature(records,outputs,model_digest=sha('model'))
    report=selective_report(records,outputs,thresholds=(0.,.9),temperature=fit['temperature'])
    assert report['selection_curve'][1]['accepted']==0 and report['selection_curve'][1]['conditional_error'] is None
    assert report['selection_curve'][1]['correct_clarifications']==0
    assert fit['qualification_admission']is False
    records[0]['split']='holdout'
    with pytest.raises(ValueError):fit_temperature(records,outputs,model_digest=sha('model'))
