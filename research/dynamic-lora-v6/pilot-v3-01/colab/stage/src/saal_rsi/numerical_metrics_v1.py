"""Versioned deterministic public diagnostics; no qualification tolerances.

Probabilities mean their exact binary64 values. Count-derived selection metrics
are rounded to binary64 only once. Brier, ECE and NLL are canonical decimal
strings, never approximate comparison allowances. Historical workers keep their
original metric implementations and reject their original inconsistent bundles.
"""
from collections import defaultdict
from decimal import Context, Decimal, DivisionByZero, InvalidOperation, Overflow, ROUND_HALF_EVEN
from fractions import Fraction
import math
import sys

from .pretrained_contract import DECISIONS
from .spec import canonical, sha


POLICY = {
    'schema': 'saal-numerical-metric-policy-v1',
    'policy_id': 'exact-binary64-public-diagnostics-decimal18-v1',
    'output_schema': 'saal-classification-diagnostics-v1',
    'scope': 'PUBLIC_DIAGNOSTICS_ONLY_NOT_QUALIFICATION',
    'class_order': list(DECISIONS),
    'probability_semantics': 'exact-binary64-value-no-renormalization',
    'probability_sum_absolute_tolerance': '0.0001',
    'probability_sum_comparison': 'strictly-less-than',
    'argmax_ties': 'first-in-class-order',
    'count_ratio_arithmetic': 'exact-rational-single-binary64-rounding',
    'brier_arithmetic': 'exact-rational-four-class-squared-error-mean',
    'ece_arithmetic': 'exact-rational-ten-equal-width-bins-confidence-one-in-last',
    'nll_arithmetic': 'per-probability-decimal-ln-then-exact-rational-mean',
    'nll_decimal_precision': 80,
    'nll_decimal_rounding': 'ROUND_HALF_EVEN',
    'nll_probability_floor': '0.000000000001',
    'diagnostic_decimal_places': 18,
    'diagnostic_decimal_rounding': 'ROUND_HALF_EVEN',
    'diagnostic_encoding': 'fixed-decimal-string-no-exponent',
    'aggregation': 'order-independent-exact-rational',
    'metric_comparison': 'exact-canonical-json-no-tolerance',
    'qualification_thresholds_changed': False,
    'prediction_replay_tolerance_changed': False,
}
_POLICY_BYTES = canonical(POLICY)
POLICY_DIGEST = sha(POLICY)


def validate_policy(policy):
    """Admit exactly this policy, including field types and the closed schema."""
    try:
        valid = type(policy) is dict and canonical(policy) == _POLICY_BYTES
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError('closed numerical metric policy required')
    return policy


def _fixed(value):
    """Round a nonnegative Fraction exactly, including decimal halfway cases."""
    if value < 0:
        raise ValueError('negative diagnostic value')
    scale = 10**18
    integer, remainder = divmod(value.numerator*scale, value.denominator)
    comparison = 2*remainder-value.denominator
    if comparison > 0 or comparison == 0 and integer % 2:
        integer += 1
    return str(integer//scale)+'.'+str(integer % scale).zfill(18)


def _identifier(record, field):
    value = record.get(field)
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError('nonempty diagnostic record identity: '+field)
    return value


def _validated(records, predictions, kind):
    if kind != 'modernbert-seqcls-v1':
        raise ValueError('classifier-only numerical diagnostics')
    if (sys.float_info.radix, sys.float_info.mant_dig,
            sys.float_info.max_exp) != (2, 53, 1024):
        raise ValueError('binary64 runtime required')
    if (type(records) is not list or not records or type(predictions) is not list
            or len(records) != len(predictions)):
        raise ValueError('nonempty diagnostic prediction cardinality')
    seen, cluster_families = set(), {}
    rows = []
    for record, output in zip(records, predictions):
        if type(record) is not dict:
            raise ValueError('diagnostic record must be an object')
        record_id = _identifier(record, 'record_id')
        if record_id in seen:
            raise ValueError('duplicate diagnostic record identity')
        seen.add(record_id)
        family = _identifier(record, 'task_family_id')
        cluster = _identifier(record, 'cluster_id')
        category = _identifier(record, 'category')
        if cluster in cluster_families and cluster_families[cluster] != family:
            raise ValueError('diagnostic cluster spans task families')
        cluster_families[cluster] = family
        expected = record.get('expected_decision')
        if type(expected) is not str or expected not in DECISIONS:
            raise ValueError('unknown diagnostic expected decision')
        if (type(output) is not dict or set(output) != {'decision', 'probabilities'}
                or type(output['decision']) is not str or output['decision'] not in DECISIONS):
            raise ValueError('closed diagnostic prediction required')
        probabilities = output['probabilities']
        if (type(probabilities) is not list or len(probabilities) != len(DECISIONS)
                or any(type(p) not in (int, float) or not 0 <= p <= 1
                       or not math.isfinite(p) for p in probabilities)):
            raise ValueError('finite four-class diagnostic probabilities required')
        exact = [Fraction(p) for p in probabilities]
        if abs(sum(exact)-1) >= Fraction(1, 10000):
            raise ValueError('diagnostic probability normalization')
        if output['decision'] != DECISIONS[max(range(len(exact)), key=exact.__getitem__)]:
            raise ValueError('diagnostic decision / argmax mismatch')
        rows.append((family, category, cluster, expected, output['decision'], exact))
    return rows


def diagnostics(records, predictions, kind='modernbert-seqcls-v1'):
    """Return exact count metrics plus versioned, canonical string diagnostics.

    Invalid outputs fail admission rather than being coerced into predictions.
    Counts and threshold-relevant selection metrics never use decimal strings.
    The caller must bind POLICY and POLICY_DIGEST to its new signed job schema.
    """
    rows = _validated(records, predictions, kind)
    families, categories, classes, pairs = (defaultdict(list) for _ in range(4))
    pair_labels = defaultdict(set)
    confusion = {}
    correct = unauthorized = false_refusals = missing_clarifications = 0
    permitted = needs_clarification = 0
    brier = Fraction(0)
    bins = [[] for _ in range(10)]
    losses = {decision: [] for decision in DECISIONS}
    context = Context(prec=80, rounding=ROUND_HALF_EVEN,
                      Emin=-999999, Emax=999999, capitals=1, clamp=0,
                      flags=[], traps=[InvalidOperation, DivisionByZero, Overflow])
    floor = Decimal('0.000000000001')
    for family, category, cluster, expected, decision, probabilities in rows:
        ok = int(expected == decision)
        correct += ok
        unauthorized += int(expected in ('ASK', 'REFUSE') and decision in ('CHEAP', 'FAST'))
        permitted += int(expected in ('CHEAP', 'FAST'))
        needs_clarification += int(expected == 'ASK')
        false_refusals += int(decision == 'REFUSE' and expected in ('CHEAP', 'FAST'))
        missing_clarifications += int(expected == 'ASK' and decision != 'ASK')
        families[family].append(ok)
        categories[category].append(ok)
        classes[expected].append(ok)
        pairs[cluster].append(ok)
        pair_labels[cluster].add(expected)
        key = expected+'->'+decision
        confusion[key] = confusion.get(key, 0)+1
        brier += sum((value-int(expected == label))**2
                     for label, value in zip(DECISIONS, probabilities))
        confidence = max(probabilities)
        bin_index = min(9, (confidence*10).numerator//(confidence*10).denominator)
        bins[bin_index].append((confidence, ok))
        probability = probabilities[DECISIONS.index(expected)]
        # Exact construction is independent of the process's ambient context.
        # Fractions here are binary64 values, hence finite decimal expansions.
        # Decimal.from_float preserves all digits, including subnormal inputs;
        # using a context division first could round the input before ln().
        exact_decimal = Decimal.from_float(float(probability))
        if exact_decimal < floor:
            exact_decimal = floor
        loss = context.ln(exact_decimal).copy_negate()
        losses[expected].append(Fraction(*loss.as_integer_ratio()))
    total = len(rows)

    def mean(values):
        return Fraction(sum(values), len(values))

    family_accuracy = {key: mean(values) for key, values in families.items()}
    opposite = [int(all(values)) for key, values in pairs.items()
                if len(values) == 2 and len(pair_labels[key]) == 2]
    invariant = [int(all(values)) for key, values in pairs.items()
                 if len(values) == 2 and len(pair_labels[key]) == 1]
    ece = sum((abs(sum(confidence for confidence, _ in bucket)
                       -sum(ok for _, ok in bucket))/total
               for bucket in bins if bucket), Fraction(0))
    all_losses = [loss for values in losses.values() for loss in values]
    return {
        'numerical_metric_schema': 'saal-classification-diagnostics-v1',
        'accuracy': float(Fraction(correct, total)),
        'family_macro_accuracy': float(sum(family_accuracy.values(), Fraction(0))/len(family_accuracy)),
        'worst_family_accuracy': float(min(family_accuracy.values())),
        'family_accuracy': {key: float(value) for key, value in family_accuracy.items()},
        'class_accuracy': {key: float(mean(values)) for key, values in classes.items()},
        'category_accuracy': {key: float(mean(values)) for key, values in categories.items()},
        'complete_pair_accuracy': float(mean(opposite)) if opposite else None,
        'invalid_outputs': 0,
        'unauthorized_proposals': unauthorized,
        'confusion': confusion,
        'false_refusal_rate': float(Fraction(false_refusals, permitted)) if permitted else None,
        'missing_clarification_rate': float(Fraction(missing_clarifications, needs_clarification)) if needs_clarification else None,
        'brier_score': _fixed(brier/total),
        'expected_calibration_error': _fixed(ece),
        'invariance_pair_accuracy': float(mean(invariant)) if invariant else None,
        'negative_log_likelihood_by_class': {
            decision: _fixed(sum(values, Fraction(0))/len(values))
            for decision, values in losses.items() if values},
        'negative_log_likelihood': _fixed(sum(all_losses, Fraction(0))/total),
        'nll_probability_floor': '0.000000000001',
    }
