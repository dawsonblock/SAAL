"""Classification fit diagnostics are separate from independent qualification."""
import math
from .checkpoint_selection import metrics
from .pretrained_contract import DECISIONS
def prediction_equal(actual, expected, kind, tolerance):
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1e-4:
        return False
    if type(actual) is not list or type(expected) is not list or len(actual) != len(expected):
        return False
    if kind != 'modernbert-seqcls-v1':
        return actual == expected
    from .pretrained_contract import DECISIONS
    def valid(output):
        if type(output) is not dict or set(output) != {'decision', 'probabilities'}:
            return False
        values = output['probabilities']
        if type(values) is not list or len(values) != len(DECISIONS):
            return False
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            return False
        return (abs(sum(values) - 1) < 1e-4 and
                output['decision'] == DECISIONS[max(range(len(values)), key=lambda i: values[i])])
    for a, b in zip(actual, expected):
        if not valid(a) or not valid(b) or a['decision'] != b['decision']:
            return False
        if any(abs(x - y) > tolerance for x, y in zip(a['probabilities'], b['probabilities'])):
            return False
    return True




def diagnostics(records,predictions,kind='modernbert-seqcls-v1'):
    if kind!='modernbert-seqcls-v1':raise ValueError('classifier-only diagnostics')
    if not prediction_equal(predictions,predictions,'modernbert-seqcls-v1',1e-4):raise ValueError('invalid diagnostic predictions')
    m=metrics(records,predictions,'modernbert-seqcls-v1');losses={d:[] for d in DECISIONS}
    for r,p in zip(records,predictions):losses[r['expected_decision']].append(-math.log(max(p['probabilities'][DECISIONS.index(r['expected_decision'])],1e-12)))
    m['negative_log_likelihood_by_class']={d:sum(v)/len(v) for d,v in losses.items() if v}
    m['negative_log_likelihood']=sum(sum(v) for v in losses.values())/len(records)
    m['nll_probability_floor']=1e-12
    return m
