import copy
from pathlib import Path
import pytest
from saal_rsi import learning_audit


def prediction(label):
    return {'decision': label, 'probabilities': [float(x == label) for x in ('CHEAP', 'FAST', 'ASK', 'REFUSE')]}


def test_real_step_zero_safety_selection_does_not_erase_learning():
    root = Path(__file__).resolve().parents[2] / 'research/diverse-learning-v1/experiment'
    report = learning_audit.audit_bundle(root / 'artifacts/run-03-6616792a4767', (root / 'controller.pub').read_text().strip())
    assert report['historical_safety_selected_step'] == 0
    assert report['best_available_public_learning']['best_public_learning']['step'] == 600
    assert report['snapshots']['terminal']['measurements']['development']['raw_metrics']['unauthorized_proposals'] == 5
    assert not report['snapshots']['terminal']['measurements']['development']['public_screen']['passes_public_screen']
    assert report['best_available_public_learning']['qualification_admission'] == 'NOT_ADMITTED'
    assert report['qualification_status'] == 'NOT_EVALUATED'


def fixture_bundles():
    records = [{'record_id': str(i), 'task_family_id': 'f' + str(i // 2), 'cluster_id': 'p' + str(i // 2),
                'expected_decision': d, 'split': 'development', 'category': 'ordinary'} for i, d in enumerate(('CHEAP', 'FAST', 'ASK', 'REFUSE'))]
    records += [r | {'record_id': 'r' + r['record_id'], 'split': 'regression'} for r in records]
    job = {'model_spec': {'model_kind': 'modernbert-seqcls-v1'}, 'dataset': {'records': records}}
    result = {'snapshots': {'terminal': {'predictions': {s: [prediction(r['expected_decision']) for r in records if r['split'] == s] for s in ('development', 'regression')}}}}
    return job, result


def test_exact_retention_comparison_and_family_loss(monkeypatch):
    parent = fixture_bundles(); candidate = copy.deepcopy(parent)
    candidate[1]['snapshots']['terminal']['predictions']['regression'][0] = prediction('FAST')
    monkeypatch.setattr(learning_audit, 'load_bundle', lambda root, pin: parent if root == 'parent' else candidate)
    result = learning_audit.compare_retention('parent', 'unused', 'new', 'unused')
    assert not result['passes_public_retention_screen']
    assert 'regression:FAMILY_REGRESSION' in result['failures']
    assert result['effects']['regression']['accuracy_gain'] == -.25
    assert not result['external_deployment_allowed']


@pytest.mark.parametrize('mutation', ['missing', 'label', 'split'])
def test_retention_cannot_drop_or_relabel_bad_cases(monkeypatch, mutation):
    parent = fixture_bundles(); candidate = copy.deepcopy(parent)
    if mutation == 'missing': candidate[0]['dataset']['records'].pop(0)
    elif mutation == 'label': candidate[0]['dataset']['records'][0]['expected_decision'] = 'FAST'
    else: candidate[0]['dataset']['records'][0]['split'] = 'train'
    monkeypatch.setattr(learning_audit, 'load_bundle', lambda root, pin: parent if root == 'parent' else candidate)
    with pytest.raises(ValueError): learning_audit.compare_retention('parent', 'unused', 'new', 'unused')


def test_selective_deferral_cannot_be_called_correct_clarification():
    records = [{'expected_decision': 'CHEAP'}, {'expected_decision': 'FAST'}, {'expected_decision': 'ASK'}]
    predictions = [{'decision': 'CHEAP', 'probabilities': [.6, .4, 0, 0]},
                   {'decision': 'FAST', 'probabilities': [.1, .7, .1, .1]},
                   {'decision': 'FAST', 'probabilities': [.05, .9, .025, .025]}]
    report = learning_audit.selective_diagnostics(records, predictions)
    assert report['curves'][0]['retained_critical_proposals'] == 1
    assert report['curves'][-1]['coverage'] == 0 and report['curves'][-1]['retained_accuracy'] is None
    assert 'not ASK correctness' in report['limits']


def test_unknown_probabilities_rejected_in_calibration():
    with pytest.raises(ValueError): learning_audit.selective_diagnostics([{'expected_decision': 'ASK'}], [{'decision': 'FAST', 'probabilities': [0, 1, 0]}])
