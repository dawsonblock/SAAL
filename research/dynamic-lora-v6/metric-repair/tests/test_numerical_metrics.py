"""Versioned arithmetic regression tests; no model loading or cloud execution."""
import copy
import decimal
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import re

import pytest

from saal_rsi.epoch_metrics import diagnostics as legacy_diagnostics
from saal_rsi.numerical_metrics_v1 import (
    POLICY, POLICY_DIGEST, _fixed, diagnostics, validate_policy,
)
from saal_rsi.spec import canonical, sha


ROOT = Path(__file__).resolve().parents[4]
LABELS = ('CHEAP', 'FAST', 'ASK', 'REFUSE')


@pytest.fixture
def example():
    records = [
        {'record_id': 'r'+str(i), 'task_family_id': 'f'+str(i//2),
         'category': 'category'+str(i % 2), 'cluster_id': 'pair'+str(i//2),
         'expected_decision': label}
        for i, label in enumerate(LABELS)]
    predictions = [
        {'decision': 'CHEAP', 'probabilities': [.7, .1, .1, .1]},
        {'decision': 'FAST', 'probabilities': [.1, .7, .1, .1]},
        {'decision': 'CHEAP', 'probabilities': [.8, .1, .05, .05]},
        {'decision': 'REFUSE', 'probabilities': [.1, .1, .1, .7]},
    ]
    return records, predictions


def test_exact_count_metrics_and_versioned_string_diagnostics(example):
    result = diagnostics(*example)
    assert result['numerical_metric_schema'] == 'saal-classification-diagnostics-v1'
    assert result['accuracy'] == .75
    assert result['family_macro_accuracy'] == .75
    assert result['worst_family_accuracy'] == .5
    assert result['family_accuracy'] == {'f0': 1., 'f1': .5}
    assert result['complete_pair_accuracy'] == .5
    assert result['invariance_pair_accuracy'] is None
    assert type(result['invalid_outputs']) is int and result['invalid_outputs'] == 0
    assert type(result['unauthorized_proposals']) is int and result['unauthorized_proposals'] == 1
    assert result['missing_clarification_rate'] == 1.
    assert result['false_refusal_rate'] == 0.
    assert result['confusion'] == {'CHEAP->CHEAP': 1, 'FAST->FAST': 1,
                                   'ASK->CHEAP': 1, 'REFUSE->REFUSE': 1}
    for name in ('brier_score', 'expected_calibration_error', 'negative_log_likelihood'):
        assert re.fullmatch(r'\d+\.\d{18}', result[name])
    assert result['nll_probability_floor'] == '0.000000000001'
    assert set(result['negative_log_likelihood_by_class']) == set(LABELS)


def test_permutation_invariance_includes_canonical_bytes(example):
    records, predictions = example
    expected = canonical(diagnostics(records, predictions))
    rng = random.Random(73)
    for _ in range(10):
        order = list(range(len(records)))
        rng.shuffle(order)
        assert canonical(diagnostics([records[i] for i in order],
                                     [predictions[i] for i in order])) == expected


def test_private_context_ignores_process_and_default_contexts(example):
    expected = canonical(diagnostics(*example))
    saved_default = decimal.DefaultContext.copy()
    try:
        decimal.DefaultContext.prec = 2
        decimal.DefaultContext.rounding = decimal.ROUND_UP
        decimal.DefaultContext.traps[decimal.Inexact] = True
        with decimal.localcontext() as ambient:
            ambient.prec = 3
            ambient.rounding = decimal.ROUND_FLOOR
            ambient.traps[decimal.Inexact] = True
            assert canonical(diagnostics(*example)) == expected
    finally:
        for field in ('prec', 'rounding', 'Emin', 'Emax', 'capitals', 'clamp'):
            setattr(decimal.DefaultContext, field, getattr(saved_default, field))
        decimal.DefaultContext.traps = saved_default.traps.copy()
        decimal.DefaultContext.flags = saved_default.flags.copy()


def test_c_and_pure_python_decimal_implementations_agree(example, monkeypatch):
    """A second local arithmetic implementation, not a Linux/GPU claim."""
    import _pydecimal
    from saal_rsi import numerical_metrics_v1 as numeric
    expected = canonical(numeric.diagnostics(*example))
    for name in ('Context', 'Decimal', 'DivisionByZero', 'InvalidOperation', 'Overflow', 'ROUND_HALF_EVEN'):
        monkeypatch.setattr(numeric, name, getattr(_pydecimal, name))
    assert canonical(numeric.diagnostics(*example)) == expected


def test_exact_half_even_rounding_without_double_rounding():
    assert _fixed(Fraction(1, 2*10**18)) == '0.000000000000000000'
    assert _fixed(Fraction(3, 2*10**18)) == '0.000000000000000002'
    assert _fixed(Fraction(5, 2*10**18)) == '0.000000000000000002'
    assert _fixed(Fraction(7, 2*10**18)) == '0.000000000000000004'
    assert _fixed(Fraction(1)) == '1.000000000000000000'
    with pytest.raises(ValueError):
        _fixed(Fraction(-1))


def test_independent_uniform_probability_arithmetic(example):
    records, _ = example
    output = {'decision': 'CHEAP', 'probabilities': [.25]*4}
    result = diagnostics(records[:1], [output])
    assert result['brier_score'] == '0.750000000000000000'
    assert result['expected_calibration_error'] == '0.750000000000000000'
    assert result['negative_log_likelihood'] == '1.386294361119890619'


def test_family_macro_is_exact_rational_before_one_float_rounding(example):
    records, predictions = copy.deepcopy(example)
    for row in records:
        row['cluster_id'] = row['record_id']
    records[2]['task_family_id'] = 'f0'
    result = diagnostics(records, predictions)
    assert result['family_accuracy'] == {'f0': float(Fraction(2, 3)), 'f1': 1.}
    assert result['family_macro_accuracy'] == float(Fraction(5, 6))
    # This is a declared new arithmetic contract, not a retroactive rewrite of
    # legacy float-summation or its potentially different final rounding bit.
    assert result['family_macro_accuracy'] != legacy_diagnostics(records, predictions)['family_macro_accuracy']


def test_zero_and_one_probabilities_use_exact_floor(example):
    records, _ = example
    predictions = [{'decision': label,
                    'probabilities': [int(i == j) for i in range(4)]}
                   for j, label in enumerate(LABELS)]
    result = diagnostics(records, predictions)
    assert result['brier_score'] == '0.000000000000000000'
    assert result['negative_log_likelihood'] == '0.000000000000000000'
    assert result['expected_calibration_error'] == '0.000000000000000000'
    predictions[0] = {'decision': 'FAST', 'probabilities': [0, 1, 0, 0]}
    result = diagnostics(records, predictions)
    assert result['negative_log_likelihood_by_class']['CHEAP'] == '27.631021115928548208'
    predictions[0]['probabilities'][0] = math.ulp(0.)
    assert diagnostics(records, predictions) == result


def test_missing_classes_and_same_label_pairs_are_explicit(example):
    records, predictions = example
    records = copy.deepcopy(records[:2])
    records[1]['expected_decision'] = 'CHEAP'
    predictions = [copy.deepcopy(predictions[0]), copy.deepcopy(predictions[0])]
    result = diagnostics(records, predictions)
    assert result['class_accuracy'] == {'CHEAP': 1.}
    assert result['complete_pair_accuracy'] is None
    assert result['invariance_pair_accuracy'] == 1.
    assert result['missing_clarification_rate'] is None


def test_tie_argmax_is_first_label_and_no_probability_renormalization(example):
    records, _ = example
    output = {'decision': 'CHEAP', 'probabilities': [.25]*4}
    assert diagnostics(records[:1], [output])['accuracy'] == 1.
    with pytest.raises(ValueError, match='argmax'):
        diagnostics(records[:1], [{**output, 'decision': 'FAST'}])
    almost_normalized = {'decision': 'CHEAP', 'probabilities': [.25001]*4}
    result = diagnostics(records[:1], [almost_normalized])
    assert result['expected_calibration_error'] == '0.749989999999999990'
    with pytest.raises(ValueError, match='normalization'):
        diagnostics(records[:1], [{'decision': 'CHEAP', 'probabilities': [.251]*4}])


@pytest.mark.parametrize('probability', [True, '0.5', None, float('nan'),
                                         float('inf'), -float('inf'), -.1, 1.1, 10**1000])
def test_invalid_probabilities_fail_closed(example, probability):
    records, predictions = copy.deepcopy(example)
    predictions[0]['probabilities'][0] = probability
    with pytest.raises(ValueError):
        diagnostics(records, predictions)


@pytest.mark.parametrize('field,value', [('record_id', ''), ('record_id', ' x'),
                                        ('task_family_id', None), ('category', 1),
                                        ('cluster_id', ''), ('expected_decision', 'FORBIDDEN')])
def test_malformed_records_fail_before_aggregation(example, field, value):
    records, predictions = copy.deepcopy(example)
    records[0][field] = value
    with pytest.raises(ValueError):
        diagnostics(records, predictions)


def test_duplicate_ids_and_cross_family_cluster_fail(example):
    records, predictions = copy.deepcopy(example)
    records[1]['record_id'] = records[0]['record_id']
    with pytest.raises(ValueError, match='duplicate'):
        diagnostics(records, predictions)
    records[1]['record_id'] = 'unique'
    records[1]['task_family_id'] = 'different-family'
    with pytest.raises(ValueError, match='cluster spans'):
        diagnostics(records, predictions)


@pytest.mark.parametrize('variant', ['empty', 'cardinality', 'tuple', 'wrong_kind',
                                    'unknown_field', 'bad_decision', 'probability_count'])
def test_closed_inputs(example, variant):
    records, predictions = copy.deepcopy(example)
    kind = 'modernbert-seqcls-v1'
    if variant == 'empty': records, predictions = [], []
    elif variant == 'cardinality': predictions.pop()
    elif variant == 'tuple': records = tuple(records)
    elif variant == 'wrong_kind': kind = 'qwen35-generative-v1'
    elif variant == 'unknown_field': predictions[0]['authority'] = True
    elif variant == 'bad_decision': predictions[0]['decision'] = 'FORBIDDEN'
    elif variant == 'probability_count': predictions[0]['probabilities'].pop()
    with pytest.raises(ValueError):
        diagnostics(records, predictions, kind)


def test_closed_policy_rejects_changes_unknowns_and_types():
    assert validate_policy(copy.deepcopy(POLICY)) == POLICY
    assert POLICY_DIGEST == sha(POLICY)
    for key, value in [('diagnostic_decimal_places', 17), ('nll_decimal_precision', 81),
                       ('qualification_thresholds_changed', 0), ('extra', 'field'),
                       ('diagnostic_decimal_places', float('nan'))]:
        modified = copy.deepcopy(POLICY)
        modified[key] = value
        with pytest.raises(ValueError, match='closed numerical'):
            validate_policy(modified)


def test_real_interrupted_fixture_remains_legacy_inconsistent():
    study = ROOT/'research/dynamic-lora-v6/study-v2'
    job = json.loads((study/'fixed_r12_157-job.json').read_text())['payload']
    result = json.loads((study/'artifacts/run-10-6e8b0b526bfc/result.json').read_text())
    records = [row for row in job['dataset']['records'] if row['split'] == 'development']
    for role in ('best', 'terminal'):
        snapshot = result['snapshots'][role]
        legacy = legacy_diagnostics(records, snapshot['predictions'])
        assert snapshot['metrics']['brier_score'] == 0.0881157183233161
        assert legacy['brier_score'] == 0.08811571832331609
        assert snapshot['metrics']['brier_score']-legacy['brier_score'] == math.ulp(legacy['brier_score'])
        assert canonical(snapshot['metrics']) != canonical(legacy)
        new = diagnostics(records, snapshot['predictions'])
        assert new['brier_score'] == '0.088115718323316100'
        assert new['accuracy'] == snapshot['metrics']['accuracy']
        assert new['confusion'] == snapshot['metrics']['confusion']
        assert canonical(new) != canonical(snapshot['metrics'])

