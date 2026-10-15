"""A review document must not become authority or mutate old evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

from saal_rsi.model_contract import verify
from saal_rsi.numerical_metrics_v1 import POLICY, POLICY_DIGEST
from saal_rsi.spec import canonical

BASE = Path(__file__).resolve().parents[1]


def load_draft_module():
    spec = importlib.util.spec_from_file_location('metric_draft_test', BASE/'draft_campaign.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_committed_v3_draft_is_historical_unsigned_and_not_a_v4_admission():
    draft = json.loads((BASE/'CAMPAIGN_DRAFT.json').read_text())
    assert draft['authorization_status'] == 'NOT_AUTHORIZED'
    assert draft['executable_admission'] is False and draft['budget_reserved'] is False
    assert draft['signatures'] == []
    with pytest.raises(ValueError, match='signature envelope'):
        verify(draft, '00'*32)
    assert draft['schema'] == 'saal-adaptive-v3-review-draft-v1'
    assert [len(p['runs']) for p in draft['plans']] == [3, 15, 4]
    assert all(p['authorization_status'] == 'NOT_AUTHORIZED' and p['execution_status'] == 'NOT_RUN' for p in draft['plans'])
    # This preserved proposal binds v1 metrics. It cannot be treated as v4.
    assert draft['numerical_metric_policy']['schema'] == 'saal-numerical-metric-policy-v1'
    assert draft['numerical_metric_policy_digest'] == POLICY_DIGEST
    assert draft['prior_campaign']['state'] == 'FAILED'
    assert draft['prior_campaign']['missing_run_imputation_permitted'] is False
    assert draft['qualification_status'] == 'NOT_EVALUATED'
    assert draft['external_deployment_allowed'] is False
    study = draft['plans'][1]
    assert {(r['method'], r['seed']) for r in study['runs']} == {
        (m, s) for m in ('fixed-r8', 'fixed-r12', 'adalora-12-to-8') for s in (47, 83, 109, 157, 211)}
    assert study['common_training']['sample_exposures'] == 16000
    assert 'src/saal_rsi/numerical_metrics_v1.py' in draft['source_files']
    assert 'src/saal_rsi/adaptive_worker_v4.py' not in draft['worker_source_files']
