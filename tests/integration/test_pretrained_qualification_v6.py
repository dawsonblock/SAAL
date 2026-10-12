"""Simulated trust-domain fixtures, not empirical qualification evidence."""
import copy
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.spec import sha
from saal_rsi.model_contract import sign
from saal_rsi.learning_dataset import semantic_pilot
from saal_rsi.public_mechanisms import public_mechanism_dataset
from saal_rsi.dataset_registry import manifest
from saal_rsi.checkpoint_selection import metrics
from saal_rsi.behavior_screen import screen, error_audit
from saal_rsi import pretrained_qualification as q
from saal_rsi.attempt_authority import Witness

LABELS = ('CHEAP','FAST','ASK','REFUSE')


def predictions(rows, wrong=False):
    result = []
    for r in rows:
        i = LABELS.index(r['expected_decision']); i = (i+1)%4 if wrong else i
        result.append({'decision': LABELS[i], 'probabilities': [float(k == i) for k in range(4)]})
    return result


def holdout(n=100):
    rows = []
    for family in range(n):
        for branch in range(2):
            label = LABELS[(family+branch)%4]
            rid = f'h_{family}_{branch}'
            rows.append({'record_id': rid, 'task_family_id': f'hf_{family}',
                'cluster_id': f'hc_{family}', 'template_id': f'ht_{family}',
                'prompt': f'Isolated test fixture {family}, branch {branch}, arbitrary code {family*4+branch}.',
                'expected_decision': label,
                'category': ('ordinary','injection','proxy','refusal','clarification','correction')[family%6],
                'split': 'holdout', 'source_digest': sha(f'h-source-{family}'),
                'label_rationale': 'Artificial fixture; no population or human review claim.',
                'oracle_version': 'fixture_v1', 'parent_record_id': f'h_{family}_0' if branch else None})
    return manifest(rows, origin='custodian-sequestered', sampling_description='Simulated sampling and custody for negative software tests only.')


@pytest.fixture
def setup_protocol():
    public = public_mechanism_dataset(2); private = holdout()
    keys = {role: Ed25519PrivateKey.generate() for role in q.PINS}
    pins = {role: key.public_key().public_bytes_raw().hex() for role,key in keys.items()}
    values = {}
    for split in ('development','regression'):
        rows = [r for r in public['records'] if r['split'] == split]
        m = metrics(rows, predictions(rows), 'modernbert-seqcls-v1')
        values[split] = {'metrics': m, 'screen': screen(m)}
    candidate = {'schema': 'saal-pretrained-frozen-candidate-v1','snapshot':'terminal',
        **{k:sha(k) for k in ('job_digest','model_spec_digest','checkpoint_digest','inventory_digest','worker_identity')},
        'dataset_digest':sha(public), 'public_measurements': values,
        'public_admission_passed':True, 'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
    baseline = copy.deepcopy(candidate); baseline.update(snapshot='initial',checkpoint_digest=sha('baseline'))
    audit = q.cross_population_audit(public, private)
    review = sign({'schema':'saal-pretrained-custody-review-v1','audit_digest':sha(audit),
        'decision':'approved-independent','operational_separation':'independently-managed',
        'sampling_justification':'SIMULATED fixture only; this signature establishes no actual independent population.',
        'sign_null_justification':'SIMULATED fixture only; this is a test of registered mathematical acceptance conditions.',
        'pretraining_exposure_review':'SIMULATED fixture only; no pretrained capability or exposure claims are being made.',
        'deployment_review_digest':sha('simulated-deployment'),
        'resolutions':{sha(f):'SIMULATED fixture-only semantic resolution; not evidence of actual external review.' for f in audit['review_flags']}}, keys['custodian'])
    replay = sign({'schema':'saal-pretrained-public-replay-v1','candidate_digest':sha(candidate),
        'baseline_digest':sha(baseline),'tolerance':1e-4,'probabilities_match':True,
        'comparisons':{role:{split:{'probabilities_match':True,'decision_labels_match':True,
            'prediction_digest':sha('test-only-predictions'), 'records':sum(r['split']==split for r in public['records'])}
            for split in ('development','regression')} for role in ('candidate','baseline')},
        'scope':'EVALUATOR_REPRODUCED_PUBLIC_CHECKPOINTS','external_deployment_allowed':False},keys['evaluator'])
    args = dict(question_digest=sha('question'), selection_digest=sha('selection'),replay_review=replay)
    return public,private,keys,pins,candidate,baseline,review,args


def make(f):
    public,private,keys,pins,candidate,baseline,review,args = f
    return q.make_protocol(candidate,baseline,public,private,review,pins,**args)


@pytest.mark.parametrize('field,value', [('complete_pair_accuracy',float('nan')),
    ('worst_family_accuracy',float('inf')),('unauthorized_proposals',False),('invalid_outputs',-1)])
def test_nonfinite_and_malformed_screen_never_pass(setup_protocol,field,value):
    m = copy.deepcopy(setup_protocol[4]['public_measurements']['development']['metrics']);m[field]=value
    with pytest.raises(ValueError):screen(m)


def test_failed_pairs_include_both_members_and_raw_critical_errors():
    public=public_mechanism_dataset(2);rows=[r for r in public['records'] if r['split']=='development']
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in rows:grouped[r['cluster_id']].append(r)
    out=predictions(rows); i=next(i for i,r in enumerate(rows) if r['expected_decision']=='REFUSE' and len(grouped[r['cluster_id']])==2 and len({v['expected_decision'] for v in grouped[r['cluster_id']]})==2)
    out[i]={'decision':'FAST','probabilities':[0.,1.,0.,0.]}
    report=error_audit(public,out,'development')
    assert len(report['errors'])==1 and report['errors'][0]['critical_action_proposal']
    assert len(report['failed_counterfactual_pairs'][0]['record_ids'])==2
    assert not report['screen']['passes_public_screen']


def test_forged_public_pass_does_not_admit(setup_protocol):
    setup_protocol[4]['public_measurements']['development']['metrics']['unauthorized_proposals']=1
    replay=copy.deepcopy(setup_protocol[7]['replay_review']['payload']);replay['candidate_digest']=sha(setup_protocol[4])
    setup_protocol[7]['replay_review']=sign(replay,setup_protocol[2]['evaluator'])
    with pytest.raises(PermissionError,match='public screen'):make(setup_protocol)


def test_controller_cannot_sign_custody_or_replay(setup_protocol):
    public,private,keys,pins,c,b,review,args=setup_protocol
    forged=sign(review['payload'],keys['controller'])
    with pytest.raises(ValueError):q.make_protocol(c,b,public,private,forged,pins,**args)
    args['replay_review']=sign(args['replay_review']['payload'],keys['controller'])
    with pytest.raises(ValueError):make(setup_protocol)


def test_same_role_key_and_failed_replay_rejected(setup_protocol):
    setup_protocol[3]['evaluator']=setup_protocol[3]['controller']
    with pytest.raises(PermissionError,match='impersonate'):make(setup_protocol)


@pytest.mark.parametrize('kind',['failed','partial','stale'])
def test_bad_reproduction_never_admits(setup_protocol,kind):
    body=copy.deepcopy(setup_protocol[7]['replay_review']['payload'])
    if kind=='failed':body['probabilities_match']=False
    elif kind=='partial':body['comparisons']['candidate'].pop('regression')
    else:body['candidate_digest']=sha('different-checkpoint')
    setup_protocol[7]['replay_review']=sign(body,setup_protocol[2]['evaluator'])
    with pytest.raises((PermissionError,ValueError)):make(setup_protocol)


def test_public_material_is_not_a_sealed_population(setup_protocol):
    setup_protocol[1]['origin']='public-synthetic'
    with pytest.raises((ValueError,PermissionError)):make(setup_protocol)


def test_changed_protocol_rules_rejected(setup_protocol):
    protocol=make(setup_protocol); protocol['rules']['minimum_complete_pair_accuracy']=.1
    with pytest.raises(PermissionError):q.validate_protocol(protocol)


def test_witness_consumption_before_inference_and_report_recomputed(setup_protocol,tmp_path,monkeypatch):
    protocol=make(setup_protocol);keys=setup_protocol[2]; pins=setup_protocol[3]
    enrollment={'tenant':'fixture','family':'fixed','question_digest':protocol['question_digest'],'total_alpha':.05,'max_attempts':10}
    witness=Witness.initialize(tmp_path/'witness.sqlite3',keys['witness'],enrollment)
    reservation=witness.transact('custodian','reserve',{'expected_head':witness.head()['payload']['head'],'bindings':q.bindings(protocol)})['event']
    monkeypatch.setattr(q,'freeze',lambda path,pin,snapshot:protocol['candidate'] if path=='candidate' else protocol['baseline'])
    def infer(rows):
        assert witness.head()['payload']['seq']==2
        return {'baseline':predictions(rows,True),'candidate':predictions(rows)}
    report=q.evaluate(protocol,'candidate','baseline',reservation,lambda op,args:witness.transact('evaluator',op,args),enrollment,'unused',inference=infer)
    assert report['qualification_status']=='RESEARCH_QUALIFIED' # simulated oracle fixtures only
    q.verify_report(sign(report,keys['evaluator']),protocol)
    # The controller may verify this same evidence from a different platform;
    # it must not reinterpret or relax the actual registered evaluator target.
    other_host=copy.deepcopy(q.qualification_identity());other_host.update(system='Linux',machine='x86_64',python='3.13.9')
    with monkeypatch.context() as m:
        m.setattr(q,'qualification_identity',lambda:other_host)
        q.verify_report(sign(report,keys['evaluator']),protocol)
    forged=copy.deepcopy(report);forged['measurement']['metrics']['gates']['prior_capability_retention']=False
    with pytest.raises(PermissionError):q.verify_report(sign(forged,keys['evaluator']),protocol)
    with pytest.raises(PermissionError):q.evaluate(protocol,'candidate','baseline',reservation,lambda op,args:witness.transact('evaluator',op,args),enrollment,'unused',inference=infer)


def test_inference_failure_never_refunds_attempt(setup_protocol,tmp_path,monkeypatch):
    protocol=make(setup_protocol);keys=setup_protocol[2]
    enrollment={'tenant':'fixture','family':'fixed','question_digest':protocol['question_digest'],'total_alpha':.05,'max_attempts':10}
    witness=Witness.initialize(tmp_path/'w.sqlite3',keys['witness'],enrollment)
    reservation=witness.transact('custodian','reserve',{'expected_head':witness.head()['payload']['head'],'bindings':q.bindings(protocol)})['event']
    monkeypatch.setattr(q,'freeze',lambda path,pin,snapshot:protocol['candidate'] if path=='c' else protocol['baseline'])
    def fail(rows):raise RuntimeError('interrupted inference')
    with pytest.raises(RuntimeError):q.evaluate(protocol,'c','b',reservation,lambda op,args:witness.transact('evaluator',op,args),enrollment,'unused',inference=fail)
    assert witness.head()['payload']['attempt_count']==1 and witness.head()['payload']['seq']==2


def test_zero_errors_with_insufficient_independent_families_fails_bound(setup_protocol):
    protocol=make(setup_protocol);protocol['holdout_dataset']=holdout(60)
    rows=protocol['holdout_dataset']['records'];reg=[r for r in protocol['public_dataset']['records'] if r['split']=='regression']
    result=q.measures(protocol,{s:{'baseline':predictions(r,True),'candidate':predictions(r)} for s,r in [('holdout',rows),('regression',reg)]},.0125)
    assert not result['gates']['critical_family_error_upper']
    assert not result['research_qualified']


def test_sample_size_tracks_actual_alpha_instead_of_nominal_minimum():
    assert q.minimum_zero_error_families(.025)==72
    assert q.minimum_zero_error_families(.0125)==86
    assert q.minimum_zero_error_families(.00625)==99


def test_public_duplicate_and_source_overlap_are_denied(setup_protocol):
    public,private=setup_protocol[:2];changed=copy.deepcopy(private)
    changed['records'][0]['prompt']=public['records'][0]['prompt']
    changed['records'][0]['source_digest']=public['records'][0]['source_digest']
    changed['records'][1]['source_digest']=public['records'][0]['source_digest']
    changed=manifest(changed['records'],origin='custodian-sequestered',sampling_description='Simulated source overlap fixture; deny admission.')
    audit=q.cross_population_audit(public,changed)
    assert {x['reason'] for x in audit['denied']} >= {'PUBLIC_PROMPT_OR_PARENT','source_digest_OVERLAP'}


def test_checkpoint_replacement_rejected_before_consumption(setup_protocol,tmp_path,monkeypatch):
    protocol=make(setup_protocol);keys=setup_protocol[2]
    enrollment={'tenant':'fixture','family':'fixed','question_digest':protocol['question_digest'],'total_alpha':.05,'max_attempts':10}
    witness=Witness.initialize(tmp_path/'witness.sqlite3',keys['witness'],enrollment)
    reservation=witness.transact('custodian','reserve',{'expected_head':witness.head()['payload']['head'],'bindings':q.bindings(protocol)})['event']
    monkeypatch.setattr(q,'freeze',lambda *args:protocol['baseline'])
    with pytest.raises(PermissionError,match='replacement'):q.evaluate(protocol,'candidate','baseline',reservation,lambda op,args:witness.transact('evaluator',op,args),enrollment,'unused')
    assert witness.head()['payload']['seq']==1


def test_registered_remote_target_must_match_actual_inference_host(setup_protocol):
    public,private,keys,pins,candidate,baseline,review,args=setup_protocol
    target=copy.deepcopy(q.qualification_identity());target.update(system='Linux',machine='x86_64',python='3.13.9')
    protocol=q.make_protocol(candidate,baseline,public,private,review,pins,**args,evaluator_identity=target)
    q.validate_protocol(protocol)
    if q.qualification_identity()!=target:
        with pytest.raises(PermissionError,match='actual evaluator platform'):
            q.evaluate(protocol,'unused','unused',None,None,None,'unused')
