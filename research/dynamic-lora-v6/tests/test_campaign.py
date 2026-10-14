import copy
import json
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.adaptive_campaign import stage_worker, unpack
from saal_rsi.adaptive_registry import AdaptiveRegistry
from saal_rsi.adaptive_evaluator import paired_comparison
from saal_rsi.model_contract import sign
from saal_rsi.spec import sha


def admitted_protocol(job, key):
    budget = {k: job['budget'][k] for k in ('campaign_id', 'max_runs', 'total_lifecycle_seconds', 'automatic_training_retries', 'financial_cap')}
    budget['max_provisioning_attempts'] = 1
    return sign({'schema': 'saal-adaptive-campaign-v1', 'budget': budget, 'job_digests': [sha(job)],
                 'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}, key)


def test_durable_budget_has_one_provision_and_no_retry(tmp_path, admitted_job):
    key = Ed25519PrivateKey.generate(); pin = key.public_key().public_bytes_raw().hex()
    p = admitted_protocol(admitted_job, key)
    ledger = AdaptiveRegistry(tmp_path/'budget.sqlite3', pin, p, create=True)
    ledger.reserve(sign(admitted_job, key)); ledger.begin_provisioning()
    with pytest.raises(ValueError): ledger.begin_provisioning()
    with pytest.raises(ValueError): ledger.end(True)
    ledger.start(sha(admitted_job))
    ledger.finish(sha(admitted_job), {'error': 'cloud failure'}, False)
    ledger.end(False)
    with pytest.raises(ValueError): ledger.start(sha(admitted_job))
    assert ledger.records()['attempts'][0]['state'] == 'INTERRUPTED'
    assert ledger.records()['provisioning_attempts'] == 1
    assert ledger.records()['rollback_resistant'] is False
    ledger.close()
    with pytest.raises(FileExistsError): AdaptiveRegistry(tmp_path/'budget.sqlite3', pin, p, create=True)
    again = AdaptiveRegistry(tmp_path/'budget.sqlite3', pin, p)
    with pytest.raises(ValueError): again.begin_provisioning()
    again.close()


def test_cannot_provision_without_complete_reservations(tmp_path, admitted_job):
    key = Ed25519PrivateKey.generate(); pin = key.public_key().public_bytes_raw().hex()
    ledger = AdaptiveRegistry(tmp_path/'budget.sqlite3', pin, admitted_protocol(admitted_job, key), create=True)
    with pytest.raises(ValueError): ledger.begin_provisioning()
    ledger.close()


def test_budget_and_cross_job_substitution_rejected(tmp_path, admitted_job):
    key = Ed25519PrivateKey.generate(); pin = key.public_key().public_bytes_raw().hex()
    ledger = AdaptiveRegistry(tmp_path/'budget.sqlite3', pin, admitted_protocol(admitted_job, key), create=True)
    changed = copy.deepcopy(admitted_job); changed['budget']['campaign_id'] = 'different'
    with pytest.raises(ValueError): ledger.reserve(sign(changed, key))
    ledger.close()


def test_staged_worker_is_closed_and_has_no_evaluation_datasets(tmp_path):
    stage_worker(tmp_path/'worker')
    result = subprocess.run([sys.executable, '-I', str(tmp_path/'worker/worker.py'), '--help'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    paths = [p.relative_to(tmp_path/'worker').as_posix() for p in (tmp_path/'worker').rglob('*') if p.is_file()]
    assert not any(p.endswith(('.key', '.ed25519')) or 'dataset.json' in p or 'custodian' in p for p in paths)
    assert set(p for p in paths if p.endswith('.json')) == {'configs/pretrained/registry.json', 'configs/pretrained/tensor_inventories.json'}


def test_five_seed_comparison_is_descriptive_and_never_promotes(admitted_job):
    rows = copy.deepcopy([r for r in admitted_job['dataset']['records'] if r['split'] == 'development'])
    for index, r in enumerate(rows): r['cluster_id'] = 'pair_'+str(index//2)
    decisions = ('CHEAP', 'FAST', 'ASK', 'REFUSE')
    def output(label): return {'decision': label, 'probabilities': [.7 if d == label else .1 for d in decisions]}
    good = [output(r['expected_decision']) for r in rows]
    bad = [output('FAST' if r['expected_decision'] != 'FAST' else 'CHEAP') for r in rows]
    runs = {s: {'fixed-r8': bad, 'fixed-r12': good, 'adalora-12-to-8': good} for s in (47, 83, 109, 157, 211)}
    result = paired_comparison(rows, runs, iterations=200)
    assert result['pair_gain'] == 1
    assert result['development_screen'] is True
    assert result['qualification_status'] == 'NOT_EVALUATED'
    assert result['make_default_adapter'] is False and result['external_deployment_allowed'] is False
    del runs[211]
    with pytest.raises(ValueError): paired_comparison(rows, runs)
