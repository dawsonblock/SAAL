import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('adaptive_report_tested', ROOT/'tools/adaptive_campaign_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_replay_error_is_reported_even_with_unchanged_decisions():
    a = [{'decision': 'CHEAP', 'probabilities': [.7, .1, .1, .1]}]
    b = [{'decision': 'CHEAP', 'probabilities': [.699, .101, .1, .1]}]
    r = report.replay_differences(a, b)
    assert not r['passed'] and r['changed_decisions'] == 0
    assert r['cases_exceeding_tolerance'] == 1
    assert r['max_probability_delta'] == pytest.approx(.001)


def test_equal_replay_passes_without_changing_tolerance():
    a = [{'decision': 'FAST', 'probabilities': [.1, .7, .1, .1]}]
    assert report.replay_differences(a, a)['passed']


@pytest.mark.parametrize('bad', [[], [{'decision': 'CHEAP', 'probabilities': [float('nan')]*4}],
    [{'decision': 'FORBIDDEN', 'probabilities': [.7, .1, .1, .1]}]])
def test_malformed_replay_cannot_be_reclassified_as_numerical_success(bad):
    with pytest.raises(ValueError):
        report.replay_differences(bad, [{'decision': 'CHEAP', 'probabilities': [.7, .1, .1, .1]}])
