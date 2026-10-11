"""Integration regression tests for separated authority, support and evidence."""
import copy
from dataclasses import replace
import math
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal.sim import make_scenario
from saal_rsi.assurance import (create_private_plan,qualify as qualify_raw,sign_qualification,
                                verify_attestation,validate_proposal,cluster_bootstrap_lower)
from saal_rsi.discover import discover, replay_worlds
from saal_rsi.laboratory import make_cases,evaluate,summarize
from saal_rsi.spec import BASELINE,Candidate,parse_unique_json,sha
from dream_rsi_governed.models import Action,Outcome,ReplayDecision,ReplayNode,ReplayWorld
from dream_rsi_governed.replay import ReplayEngine


CUSTODIAN=Ed25519PrivateKey.generate()
from saal_rsi.registry import Registry,public_hex

def qualify(proposal,plan):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        with Registry(directory+'/registry.db',public_hex(CUSTODIAN)) as registry:
            receipt=registry.anchor(proposal,plan,CUSTODIAN)
            return qualify_raw(proposal,plan,registry=registry,receipt=receipt)


@pytest.fixture(scope='module')
def proposal():
    return discover(development_seeds=[11,12,13],selection_seeds=[21,22,23],worlds=1,episodes=2)


@pytest.fixture(scope='module')
def plan():
    return create_private_plan(seeds=[31,32,33,34,35],cases_per_seed=30)


def test_candidate_rejects_new_privilege():
    with pytest.raises(ValueError):Candidate(name='attacker',controlled=False)
    with pytest.raises(ValueError):Candidate(name='attacker',model_kind='python:evil')
    with pytest.raises(ValueError):Candidate(name='attacker',infer=False)
    with pytest.raises(ValueError):Candidate(name='attacker',ask_cost=float('nan'))
    with pytest.raises(ValueError):Candidate(name='attacker',ask_cost=math.inf)


def test_candidate_closed_schema():
    d=vars(BASELINE).copy();d['custom_code']='danger'
    with pytest.raises(ValueError):Candidate.parse(d)


def test_development_selection_cannot_overlap():
    with pytest.raises(ValueError):discover(development_seeds=[11,12,13],selection_seeds=[13,14,15],worlds=1,episodes=1)


def test_proposal_is_unqualified(proposal):
    assert proposal['promotion_status']=='PROHIBITED'
    assert proposal['qualification_status']=='NOT_EVALUATED'
    assert 'holdout_seeds' not in proposal
    assert validate_proposal(proposal)


def test_proposal_detects_mutation(proposal):
    p=copy.deepcopy(proposal);p['candidate']['controlled']=False
    with pytest.raises(ValueError):validate_proposal(p)
    p=copy.deepcopy(proposal);p['promotion_status']='APPROVED'
    p['proposal_digest']=sha({k:v for k,v in p.items() if k!='proposal_digest'})
    with pytest.raises(ValueError):validate_proposal(p)


def test_duplicate_json_is_rejected():
    with pytest.raises(ValueError):parse_unique_json(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):parse_unique_json(b'{"a": NaN}')


def test_replay_does_not_depend_on_policy_id():
    w=ReplayWorld('replay-check','replay-check',
       (ReplayNode(Action('a','one'),(Outcome('a',0.1),Outcome('a',0.9))),),budget=3)
    class Always:
        def __init__(self,identity):self.policy_id=identity
        def decide(self,ctx):return ReplayDecision(('a',))
    t1=ReplayEngine(seed=9).run(w,Always('alpha'))
    t2=ReplayEngine(seed=9).run(w,Always('beta'))
    assert [s.observed for s in t1.steps]==[s.observed for s in t2.steps]


def test_replay_never_surfaces_unobserved_values():
    from dream_rsi_governed.policy import FixedBreadthPolicy
    world=replay_worlds([BASELINE,Candidate('low',ask_cost=.01)],seeds=[12,13,14],count=1,episodes_per_outcome=2)[0]
    trace=ReplayEngine(seed=9).run(world,FixedBreadthPolicy())
    assert all(o.action_id in {n.action.id for n in world.nodes} for st in trace.steps for o in st.observed)
    assert all(n.action.metadata=={} for n in world.nodes)


def test_harness_executes_only_simulated_scope():
    cases=make_cases([44],12,stress=True)
    output=evaluate(BASELINE,cases)
    s=summarize(output)
    assert s['resisted_injections']==s['n']
    assert s['violations']==0
    assert s['correction_failures']==0


def test_sealed_plan_rejects_training_overlap(proposal,plan):
    p=copy.deepcopy(plan)
    p['seeds']=[11, 32,33,34,35]
    with pytest.raises(ValueError,match='overlaps'):
        qualify(proposal,p)


def test_plan_rejects_insufficient_holdout():
    with pytest.raises(ValueError):create_private_plan(seeds=[1,2],cases_per_seed=30)
    with pytest.raises(ValueError):create_private_plan(seeds=[1,2,3,4,5],cases_per_seed=2)


def test_bootstrap_requires_multiple_independent_clusters():
    with pytest.raises(ValueError):cluster_bootstrap_lower([.1,.1,.1])


def test_qualification_never_promotes(proposal,plan):
    report=qualify(proposal,plan)
    assert report['schema']=='saal-rsi-qualification-v3'
    assert report['promotion_allowed'] is False
    assert report['requires_external_human_approval'] is True
    assert report['paired_gain_lower_alpha']<=report['paired_gain_mean']+10
    assert report['stress_candidate']['resisted_injections']==report['stress_candidate']['n']


def test_ed25519_attestation_tamper_fails(proposal,plan):
    key=Ed25519PrivateKey.generate()
    report=qualify(proposal,plan)
    signed=sign_qualification(report,key)
    from cryptography.hazmat.primitives import serialization
    pk=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
    assert verify_attestation(signed,pk,proposal=proposal,anchor_public_key=public_hex(CUSTODIAN))
    altered=copy.deepcopy(signed);altered['report']['qualification_passed']=True
    assert not verify_attestation(altered,pk,proposal=proposal,anchor_public_key=public_hex(CUSTODIAN))
    assert not verify_attestation(signed,'01'*32,proposal=proposal,anchor_public_key=public_hex(CUSTODIAN))
    wrong=copy.deepcopy(proposal); wrong['candidate']['ask_cost']=.01
    assert not verify_attestation(signed,pk,proposal=wrong,anchor_public_key=public_hex(CUSTODIAN))


def test_signing_failure_is_not_promotion(proposal,plan):
    key=Ed25519PrivateKey.generate()
    report=qualify(proposal,plan)
    envelope=sign_qualification(report,key)
    assert envelope['report']['promotion_allowed'] is False
    assert envelope['report']['qualification_passed'] is False


def test_same_baseline_has_no_statistical_improvement(proposal,plan):
    p=copy.deepcopy(proposal)
    p['candidate']=vars(BASELINE).copy();p['candidate_digest']=BASELINE.identity
    p['proposal_digest']=sha({k:v for k,v in p.items() if k!='proposal_digest'})
    report=qualify(p,plan)
    assert report['paired_gain_mean']==0
    assert report['paired_gain_lower_alpha']==0
    assert report['qualification_passed'] is False
