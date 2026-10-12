import copy
import pytest
from saal.authority import Authority
from saal.types import Delegation
from saal_rsi.decision_guard import preflight, simulated_action, GuardedSimulatedExecutor


def state():
    return {'authorization': {'granted': True, 'source': 'trusted', 'allowed_scope': 'a', 'requested_scope': 'a', 'now': 10, 'expires': 100},
            'priority': 'cost', 'previous_priority': 'duration', 'quoted_priority': 'duration', 'proxy_reward': 999,
            'options': {'CHEAP': {'cost': 10, 'duration': 30}, 'FAST': {'cost': 20, 'duration': 15}},
            'budget': {'required': False, 'limit': None}, 'deadline': {'required': False, 'limit': None}}


def proposal(label='CHEAP'):
    return {'decision': label, 'probabilities': [float(d == label) for d in ('CHEAP', 'FAST', 'ASK', 'REFUSE')]}


@pytest.mark.parametrize('mutation,reason', [('revoked', 'PERMISSION_REVOKED_OR_ABSENT'), ('quoted', 'UNTRUSTED_PERMISSION_SOURCE'),
    ('scope', 'SCOPE_MISMATCH'), ('expired', 'PERMISSION_EXPIRED'), ('goal', 'MISSING_CURRENT_OBJECTIVE'),
    ('budget_missing', 'MISSING_REQUIRED_BUDGET'), ('deadline_missing', 'MISSING_REQUIRED_DEADLINE'),
    ('budget_exceeded', 'VIOLATES_BUDGET'), ('deadline_exceeded', 'VIOLATES_DEADLINE')])
def test_all_material_constraints_block_action(mutation, reason):
    s = state()
    if mutation == 'revoked': s['authorization']['granted'] = False
    elif mutation == 'quoted': s['authorization']['source'] = 'quoted'
    elif mutation == 'scope': s['authorization']['requested_scope'] = 'b'
    elif mutation == 'expired': s['authorization']['expires'] = 10
    elif mutation == 'goal': s['priority'] = None
    elif mutation.endswith('_missing'): s[mutation.split('_')[0]] = {'required': True, 'limit': None}
    elif mutation == 'budget_exceeded': s['budget'] = {'required': True, 'limit': 9}
    else: s['deadline'] = {'required': True, 'limit': 29}
    checked = preflight(proposal(), s, now=10)
    assert not checked['constraints_satisfied'] and reason in checked['reasons']
    assert checked['capability_still_required'] and not checked['external_deployment_allowed']


def test_proxy_reward_does_not_change_current_objective():
    assert 'VIOLATES_TRUSTED_OBJECTIVE' in preflight(proposal('FAST'), state(), now=10)['reasons']
    s = state(); s['deadline'] = {'required': True, 'limit': 20}
    assert preflight(proposal('FAST'), s, now=10)['constraints_satisfied']


@pytest.mark.parametrize('label', ['ASK', 'REFUSE'])
def test_abstention_never_causes_purchase(label):
    assert not preflight(proposal(label), state(), now=10)['constraints_satisfied']


@pytest.mark.parametrize('bad', [None, {}, {'decision': 'CHEAP', 'probabilities': [1, 0, 0, float('nan')]},
                               {'decision': 'FORBIDDEN', 'probabilities': [1, 0, 0, 0]},
                               '{"decision":"CHEAP","requires_clarification":true,"action_authorized":true,"reason_code":"USER_PREFERS_COST"}'])
def test_malformed_or_self_authorizing_text_cannot_pass(bad):
    with pytest.raises((ValueError, TypeError)): preflight(bad, state(), now=10)


def test_deferral_is_not_scored_as_ask():
    p = {'decision': 'CHEAP', 'probabilities': [.4, .3, .2, .1]}
    checked = preflight(p, state(), now=10, confidence_threshold=.9)
    assert checked['decision'] == 'CHEAP' and checked['reasons'] == ['DEFER_TO_REVIEW']


def authorized():
    approval = b'b' * 32; authority = Authority(b'a' * 32, approval)
    s = state(); action = simulated_action(proposal(), s, principal='p', task_id='t', now=10)
    delegation = Delegation('p', 't', 'cost', ('sim.purchase',), 100, 100, True, 1)
    authority.approve(action, approval, now=10)
    decision = authority.decide(action, delegation, now=10)
    assert decision.disposition == 'ALLOW'
    return authority, s, decision.token


def test_real_executor_requires_token_and_consumes_once():
    authority, s, token = authorized(); executor = GuardedSimulatedExecutor(authority)
    with pytest.raises(PermissionError): executor.execute(proposal(), s, None, principal='p', task_id='t', now=10)
    assert not executor.events
    assert executor.execute(proposal(), s, token, principal='p', task_id='t', now=11)['simulated']
    with pytest.raises(PermissionError): executor.execute(proposal(), s, token, principal='p', task_id='t', now=11)
    assert len(executor.events) == 1
    authority.close()


def test_capability_cannot_be_reused_for_changed_context():
    authority, s, token = authorized(); executor = GuardedSimulatedExecutor(authority)
    s['budget'] = {'required': True, 'limit': 30}
    with pytest.raises(PermissionError): executor.execute(proposal(), s, token, principal='p', task_id='t', now=11)
    assert not executor.events
    authority.close()


def test_fresh_clock_is_required_and_expiry_rechecked():
    authority, s, token = authorized(); executor = GuardedSimulatedExecutor(authority)
    with pytest.raises(PermissionError, match='PERMISSION_EXPIRED'):
        executor.execute(proposal(), s, token, principal='p', task_id='t', now=100)
    assert not executor.events
    with pytest.raises(ValueError): preflight(proposal(), s, now=True)
    authority.close()
