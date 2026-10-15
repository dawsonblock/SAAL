"""Versioned diagnostics additions; no model or external evaluation needed."""
import copy

import pytest

from saal_rsi.numerical_metrics_v2 import (
    POLICY, POLICY_DIGEST, diagnostics, validate_policy,
)
from saal_rsi.spec import sha


def fixture_data():
    labels = ['CHEAP', 'FAST', 'ASK', 'REFUSE']
    records = [
        {'record_id': 'r0', 'task_family_id': 'f0', 'cluster_id': 'pair0',
         'category': 'safety', 'expected_decision': 'REFUSE'},
        {'record_id': 'r1', 'task_family_id': 'f0', 'cluster_id': 'pair0',
         'category': 'safety', 'expected_decision': 'CHEAP'},
        {'record_id': 'r2', 'task_family_id': 'f1', 'cluster_id': 'pair1',
         'category': 'clarification', 'expected_decision': 'ASK'},
        {'record_id': 'r3', 'task_family_id': 'f1', 'cluster_id': 'pair1',
         'category': 'preference', 'expected_decision': 'FAST'},
    ]
    decisions = ['FAST', 'REFUSE', 'CHEAP', 'FAST']
    predictions = []
    for decision in decisions:
        probabilities = [0.01] * 4
        probabilities[labels.index(decision)] = 0.97
        predictions.append({'decision': decision, 'probabilities': probabilities})
    return records, predictions


def test_v2_support_and_safety_metrics_have_explicit_denominators():
    records, predictions = fixture_data()
    result = diagnostics(records, predictions)
    assert result['numerical_metric_schema'] == 'saal-classification-diagnostics-v2'
    assert result['support']['records'] == 4
    assert result['support']['by_class'] == {
        'CHEAP': 1, 'FAST': 1, 'ASK': 1, 'REFUSE': 1}
    assert result['support']['by_family'] == {'f0': 2, 'f1': 2}
    assert result['support']['counterfactual_pairs']['counterfactual_opposite_label'] == 2
    assert result['counterfactual_pair_correct'] == 0
    assert result['counterfactual_pair_total'] == 2
    assert result['critical_errors']['counts'] == {
        'authorization_violation': 1,
        'unsafe_action_when_clarification_required': 1,
        'clarification_not_returned': 1,
        'false_refusal': 1,
    }
    assert result['critical_errors']['rates']['authorization_violation_per_refuse'] == '1.000000000000000000'
    assert result['critical_errors']['rates']['unsafe_action_per_ask'] == '1.000000000000000000'
    assert result['critical_errors']['rates']['false_refusal_per_permitted'] == '0.500000000000000000'


def test_v2_policy_is_closed_and_digest_bound():
    assert POLICY_DIGEST == sha(POLICY)
    assert validate_policy(copy.deepcopy(POLICY)) == POLICY
    changed = copy.deepcopy(POLICY)
    changed['additions']['authorization_violation'] = 'any-error'
    with pytest.raises(ValueError, match='closed numerical metric v2'):
        validate_policy(changed)


@pytest.mark.parametrize('mutation', ['duplicate', 'invalid-decision', 'missing-probability'])
def test_v2_preserves_v1_fail_closed_input_validation(mutation):
    records, predictions = fixture_data()
    if mutation == 'duplicate':
        records[1]['record_id'] = records[0]['record_id']
    elif mutation == 'invalid-decision':
        predictions[0]['decision'] = 'FORBIDDEN'
    else:
        predictions[0].pop('probabilities')
    with pytest.raises((ValueError, KeyError)):
        diagnostics(records, predictions)
