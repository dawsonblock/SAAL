import copy
import itertools
import pytest
from saal_rsi.choice_policy_v1 import decide
from saal_rsi.structured_choice import plan, check_proposal
from saal_rsi.family_gain_v2 import family_gain


def state():
    return {'authorization': {'granted': True, 'source': 'trusted', 'allowed_scope': 'a',
        'requested_scope': 'a', 'now': 9, 'expires': 10},
        'options': {'CHEAP': {'cost': 10, 'duration': 60}, 'FAST': {'cost': 30, 'duration': 20}},
        'priority': 'cost', 'previous_priority': 'duration', 'quoted_priority': 'duration',
        'proxy_reward': 99, 'budget': {'required': False, 'limit': None},
        'deadline': {'required': False, 'limit': None}}


def test_exhaustive_boundary_and_precedence_control_matches_existing_oracle():
    limits = [(False, None), (True, None), (True, 0), (True, 10), (True, 30), (True, 60)]
    checked = 0
    for granted, source, scope, now, priority, budget, deadline in itertools.product(
            (False, True), ('trusted', 'quoted'), ('a', 'b'), (9, 10, 11),
            ('cost', 'duration', None), limits, limits):
        s = state()
        s['authorization'].update(granted=granted, source=source, requested_scope=scope, now=now)
        s['priority'] = priority
        s['budget'] = dict(required=budget[0], limit=budget[1])
        s['deadline'] = dict(required=deadline[0], limit=deadline[1])
        result = plan(s)
        assert (result['decision'], result['reason']) == decide(s)
        assert result['execution_authorized'] is False
        checked += 1
    assert checked == 2592


def test_corrections_and_proxy_never_override_current_trusted_objective():
    s = state()
    before = copy.deepcopy(s)
    assert plan(s)['decision'] == 'CHEAP'
    s['priority'] = 'duration'
    assert plan(s)['decision'] == 'FAST'
    s['quoted_priority'], s['previous_priority'], s['proxy_reward'] = 'cost', 'cost', 1000
    assert plan(s)['decision'] == 'FAST'
    assert check_proposal(s, 'CHEAP')['matches_trusted_plan'] is False
    assert check_proposal(s, 'FAST')['execution_authorized'] is False
    assert before['priority'] == 'cost'


@pytest.mark.parametrize('fault', ['bool_cost', 'extra_authority', 'floating_clock', 'label', 'vision'])
def test_malformed_structured_facts_cannot_get_an_authority_claim(fault):
    s = state()
    if fault == 'bool_cost': s['options']['CHEAP']['cost'] = True
    elif fault == 'extra_authority': s['execution_authorized'] = True
    elif fault == 'floating_clock': s['authorization']['now'] = 9.
    elif fault == 'label': s['priority'] = 'FAST'
    else: s['image'] = 'unsafe'
    with pytest.raises(ValueError): plan(s)


def test_public_bootstrap_survives_canonical_json_key_sorting():
    import json
    before = {'family_accuracy': {'z': .3, 'b': .4, 'a': .6, 'y': .1}}
    after = {'family_accuracy': {'z': .8, 'b': .4, 'a': .9, 'y': .6}}
    reference = family_gain(before, after)
    assert reference == family_gain(json.loads(json.dumps(before, sort_keys=True)),
                                    json.loads(json.dumps(after, sort_keys=True)))
    assert reference['status'] == 'PUBLIC_DIAGNOSTIC_ONLY'
