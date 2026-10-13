import json
from pathlib import Path
from saal_rsi.behavior_screen import error_audit as historical_audit
from saal_rsi.behavior_screen_v2 import error_audit


def test_invalid_probabilities_cannot_hide_behind_correct_pair_labels():
    root = Path(__file__).resolve().parents[3]
    data = json.loads((root/'research/behavior-closure-v4/dataset.json').read_text())
    rows = [r for r in data['records'] if r['split'] == 'development']
    labels = ('CHEAP', 'FAST', 'ASK', 'REFUSE')
    outputs = [{'decision': r['expected_decision'], 'probabilities':
        [float(d == r['expected_decision']) for d in labels]} for r in rows]
    outputs[0]['probabilities'] = [1., 1., 1., 1.]
    old = historical_audit(data, outputs, 'development')
    assert old['metrics']['invalid_outputs'] == 1 and not old['failed_counterfactual_pairs']
    repaired = error_audit(data, outputs, 'development')
    assert repaired['failed_counterfactual_pairs'][0]['cluster_id'] == rows[0]['cluster_id']
    assert repaired['screen']['passes_public_screen'] is False
