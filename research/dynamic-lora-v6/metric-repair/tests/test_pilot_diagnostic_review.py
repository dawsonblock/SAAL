"""Offline tests for aggregate diagnosis and blind-review preparation."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]


def _load_script(name):
    path = ROOT / 'research/dynamic-lora-v6' / name
    spec = importlib.util.spec_from_file_location('saal_' + name.replace('.py', ''), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path):
    job = {'payload': {'final_holdout_included': False, 'dataset': {'records': [
        {'record_id': 'r1', 'task_family_id': 'f1', 'cluster_id': 'pair-1',
         'expected_decision': 'REFUSE', 'prompt': 'No permission; choose.',
         'label_rationale': 'expired permission', 'oracle_version': 'v1',
         'review_status': 'ORACLE_CHECKED_NOT_INDEPENDENTLY_REVIEWED', 'split': 'development'},
        {'record_id': 'r2', 'task_family_id': 'f1', 'cluster_id': 'pair-1',
         'expected_decision': 'CHEAP', 'prompt': 'Permission valid; choose.',
         'label_rationale': 'valid permission', 'oracle_version': 'v1',
         'review_status': 'ORACLE_CHECKED_NOT_INDEPENDENTLY_REVIEWED', 'split': 'development'},
        {'record_id': 'r3', 'task_family_id': 'f1', 'cluster_id': 'pair-2',
         'expected_decision': 'REFUSE', 'prompt': 'No feasible option; choose.',
         'label_rationale': 'infeasible', 'oracle_version': 'v1',
         'review_status': 'ORACLE_CHECKED_NOT_INDEPENDENTLY_REVIEWED', 'split': 'development'},
        {'record_id': 'r4', 'task_family_id': 'f1', 'cluster_id': 'pair-3',
         'expected_decision': 'REFUSE', 'prompt': 'No permission; decide.',
         'label_rationale': 'unauthorized', 'oracle_version': 'v1',
         'review_status': 'ORACLE_CHECKED_NOT_INDEPENDENTLY_REVIEWED', 'split': 'development'},
    ]}}}
    preds = [
        {'decision': 'FAST', 'probabilities': [0.01, 0.97, 0.01, 0.01]},
        {'decision': 'CHEAP', 'probabilities': [0.97, 0.01, 0.01, 0.01]},
        {'decision': 'REFUSE', 'probabilities': [0.01, 0.01, 0.01, 0.97]},
        {'decision': 'REFUSE', 'probabilities': [0.01, 0.01, 0.01, 0.97]},
    ]
    result = {'snapshots': {role: {'predictions': preds} for role in ('initial', 'best', 'terminal')}}
    job_path = tmp_path / 'job.json'
    result_path = tmp_path / 'result.json'
    job_path.write_text(json.dumps(job))
    result_path.write_text(json.dumps(result))
    return job_path, result_path


def test_review_builder_hides_labels_predictions_and_preserves_key(tmp_path):
    module = _load_script('prepare_blind_review.py')
    job_path, result_path = _fixture(tmp_path)
    out = tmp_path / 'review'
    manifest = module.build_review(result_path, job_path, out)
    assert manifest['state'] == 'PENDING_EXTERNAL_HUMAN_REVIEW'
    assert manifest['reviewers_claimed'] == 0
    assert manifest['terminal_refuse_to_action_cases'] == 1
    assert manifest['counterfactual_partners_included'] == 1
    assert manifest['matched_same_family_refuse_controls'] == 1
    with (out / 'reviewer-1.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert set(rows[0]) == set(module.FIELDS)
    assert all(not row['reviewed_decision'] for row in rows)
    text = (out / 'reviewer-1.csv').read_text()
    assert 'expected_decision' not in text
    assert 'terminal_prediction' not in text
    assert (out / 'organizer-key.json').stat().st_mode & 0o777 == 0o600


def test_review_builder_rejects_holdout_and_never_overwrites(tmp_path):
    module = _load_script('prepare_blind_review.py')
    job_path, result_path = _fixture(tmp_path)
    out = tmp_path / 'review'
    module.build_review(result_path, job_path, out)
    with pytest.raises(FileExistsError):
        module.build_review(result_path, job_path, out)
    job = json.loads(job_path.read_text())
    job['payload']['final_holdout_included'] = True
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match='public-only'):
        module.build_review(result_path, job_path, tmp_path / 'other')


def test_diagnostic_v2_critical_breakdown_reconciles_aggregate_count():
    module = _load_script('diagnose_pilot.py')
    records = [
        {'record_id': '1', 'task_family_id': 'f', 'cluster_id': 'p', 'expected_decision': 'REFUSE'},
        {'record_id': '2', 'task_family_id': 'f', 'cluster_id': 'p', 'expected_decision': 'CHEAP'},
    ]
    summary = {'accuracy': 0.0, 'complete_pair_accuracy': 0.0,
               'worst_family_accuracy': 0.0, 'unauthorized_proposals': 1,
               'confusion': {'REFUSE->FAST': 1, 'CHEAP->REFUSE': 1}}
    out = module._upgrade_summary(summary, records)
    counts = out['critical_errors']['counts']
    assert counts['authorization_violation'] + counts['unsafe_action_when_clarification_required'] == 1
    assert counts['false_refusal'] == 1
    assert out['counterfactual_pair_total'] == 1
    assert out['counterfactual_pair_correct'] == 0
