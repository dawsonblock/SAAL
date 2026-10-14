"""Public-analysis regression tests; stubs are not trained-model evidence."""
import copy
from fractions import Fraction
import importlib.util
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.adaptive_artifacts import file_hash
from saal_rsi.adaptive_contract import adapter_spec
from saal_rsi.adaptive_contract_v3 import new_job, worker_identity
from saal_rsi.adaptive_dataset import training_projection
from saal_rsi.adaptive_evaluator_v3 import (evaluate_frozen, family_pairs,
    paired_comparison, validate_controller_replay, validate_protocol, validate_sources,
    validate_cleanup)
from saal_rsi.learning_dataset import generative_pilot, wrap
from saal_rsi.model_contract import sign
from saal_rsi.numerical_metrics_v1 import POLICY, POLICY_DIGEST, diagnostics
from saal_rsi.pretrained_contract import DECISIONS, model_spec
from saal_rsi.spec import canonical, sha

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT/'research/dynamic-lora-v6'


def load_script(name):
    spec = importlib.util.spec_from_file_location('numerical_'+name, PROJECT/'metric-repair'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction(label):
    return {'decision': label, 'probabilities': [float(label == decision) for decision in DECISIONS]}


@pytest.fixture(scope='module')
def population():
    return wrap([row for row in generative_pilot()['records'] if row['task_type'] == 'decision'])


@pytest.fixture
def job(population):
    spec = model_spec('answerdotai/ModernBERT-base', method='none', backend='local',
        precision='float32', budget='v3_analysis_test')
    budget = {'campaign_id': 'v3_analysis_test', 'run_id': 'analysis_test', 'max_runs': 3,
        'total_lifecycle_seconds': 15000, 'reserved_seconds': 5000,
        'automatic_training_retries': 0, 'financial_cap': False}
    return new_job(spec, training_projection(population),
        adapter_spec('adalora-12-to-8', total_steps=4, tinit=1, tfinal=1, delta=1),
        seed=60606, budget=budget, steps=4, max_seconds=3600,
        campaign_stage='ENGINEERING', checkpoint_interval=2)


@pytest.fixture
def pair_records():
    rows = []
    for family, pairs in [('family_A', 3), ('family_B', 1)]:
        labels = ('CHEAP', 'FAST') if family == 'family_A' else ('ASK', 'REFUSE')
        for index in range(pairs):
            for side, label in enumerate(labels):
                rows.append({'record_id': f'{family}_{index}_{side}', 'task_family_id': family,
                    'cluster_id': f'{family}_{index}', 'category': 'counterfactual',
                    'expected_decision': label})
    return rows


def opposite(label):
    return {'CHEAP': 'FAST', 'FAST': 'CHEAP', 'ASK': 'REFUSE', 'REFUSE': 'ASK'}[label]


def test_primary_effect_uses_exact_pair_counts_and_keeps_qualification_unclaimed(pair_records):
    bad = [prediction(opposite(row['expected_decision'])) for row in pair_records]
    adapted = [prediction(row['expected_decision'] if index < 2 else opposite(row['expected_decision']))
               for index, row in enumerate(pair_records)]
    runs = {seed: {'fixed-r8': bad, 'fixed-r12': bad, 'adalora-12-to-8': adapted}
            for seed in (47, 83, 109, 157, 211)}
    result = paired_comparison(pair_records, runs, iterations=200)
    assert result['pair_gain'] == float(Fraction(1, 6))
    assert result['paired_family_seed_effects'][47]['family_A'] == float(Fraction(1, 3))
    assert result['numerical_metric_policy_digest'] == POLICY_DIGEST
    assert result['guardrails'][0]['methods']['fixed-r8']['brier_score'] == '2.000000000000000000'
    assert 'Exact rational' in result['comparison_arithmetic']
    assert result['qualification_status'] == 'NOT_EVALUATED'
    assert result['independent_generalization'] == 'NOT_ESTABLISHED'
    assert result['make_default_adapter'] is False and result['external_deployment_allowed'] is False
    assert result['retention'].startswith('NOT_MEASURED')
    reordered = {seed: {method: list(reversed(predictions)) for method, predictions in run.items()}
                 for seed, run in runs.items()}
    second = paired_comparison(list(reversed(pair_records)), reordered, iterations=200)
    assert second['pair_gain'] == result['pair_gain']
    assert second['descriptive_95_interval'] == result['descriptive_95_interval']


@pytest.mark.parametrize('fault', ['missing_pair_member', 'same_label_pair', 'duplicate_record'])
def test_primary_endpoint_rejects_incomplete_or_substituted_pairs(pair_records, fault):
    rows = copy.deepcopy(pair_records)
    if fault == 'missing_pair_member': rows.pop()
    elif fault == 'same_label_pair': rows[1]['expected_decision'] = rows[0]['expected_decision']
    else: rows[1]['record_id'] = rows[0]['record_id']
    outputs = [prediction(row['expected_decision']) for row in rows]
    with pytest.raises(ValueError): family_pairs(rows, outputs)


@pytest.mark.parametrize('fault', ['missing_seed', 'missing_method', 'iterations_bool', 'iterations_zero'])
def test_comparison_cannot_fill_missing_runs_or_unregistered_bootstrap(pair_records, fault):
    outputs = [prediction(row['expected_decision']) for row in pair_records]
    runs = {seed: {method: outputs for method in ('fixed-r8', 'fixed-r12', 'adalora-12-to-8')}
            for seed in (47, 83, 109, 157, 211)}
    iterations = 200
    if fault == 'missing_seed': del runs[211]
    elif fault == 'missing_method': del runs[157]['fixed-r12']
    elif fault == 'iterations_bool': iterations = True
    else: iterations = 0
    with pytest.raises(ValueError): paired_comparison(pair_records, runs, iterations=iterations)


def test_frozen_evaluation_uses_v3_metrics_for_all_three_checkpoints(tmp_path, job, population):
    rows = [record for record in population['records'] if record['split'] == 'regression']
    (tmp_path/'artifact_inventory.json').write_text('{}')
    calls = []
    for role in ('initial', 'best', 'terminal'):
        (tmp_path/(role+'_rank_pattern.json')).write_text('{}')

    class StubBackend:
        def restore_checkpoint(self, checkpoint, state): calls.append(checkpoint.stem)
        def evaluate(self, records): return [prediction(row['expected_decision']) for row in records]

    value = evaluate_frozen(StubBackend(), tmp_path, job, rows)
    assert calls == ['initial', 'best', 'terminal']
    assert value['schema'] == 'saal-adaptive-frozen-regression-v3'
    assert value['numerical_metric_policy_digest'] == POLICY_DIGEST
    assert all(snapshot['metrics']['brier_score'] == '0.000000000000000000'
               for snapshot in value['snapshots'].values())
    assert value['checkpoint_selection_permitted'] is False
    assert value['qualification_status'] == 'NOT_EVALUATED'


@pytest.mark.parametrize('fault', ['legacy_job', 'protected', 'unknown_field', 'label', 'source_leak', 'duplicate'])
def test_closed_regression_and_job_admission_precede_any_backend_call(tmp_path, job, population, fault):
    rows = copy.deepcopy([record for record in population['records'] if record['split'] == 'regression'])
    if fault == 'legacy_job': job['schema'] = 'saal-adaptive-job-v2'
    elif fault == 'protected': rows[0]['split'] = 'holdout'
    elif fault == 'unknown_field': rows[0]['protected_labels'] = True
    elif fault == 'label': rows[0]['expected_decision'] = 'REFUSE'
    elif fault == 'source_leak': rows[0]['source_digest'] = job['dataset']['records'][0]['source_digest']
    else: rows[1]['record_id'] = rows[0]['record_id']

    class NoInference:
        def restore_checkpoint(self, *args): pytest.fail('invalid data reached checkpoint restoration')
        def evaluate(self, *args): pytest.fail('invalid data reached inference')

    with pytest.raises(ValueError): evaluate_frozen(NoInference(), tmp_path, job, rows)
    assert not list(tmp_path.iterdir())


def test_controller_receipt_is_v3_bound_and_probability_tolerance_remains_exact(job, pair_records):
    outputs = [prediction(row['expected_decision']) for row in pair_records]
    snapshots = {role: {'predictions': outputs} for role in ('initial', 'best', 'terminal')}
    receipt = {'schema': 'saal-adaptive-controller-replay-v3', 'job_digest': sha(job),
        'worker_identity': worker_identity(), 'numerical_metric_policy': copy.deepcopy(POLICY),
        'numerical_metric_policy_digest': POLICY_DIGEST, 'inventory_sha256': 'test-inventory',
        'verified': True, 'tolerance': 1e-4, 'predictions': {role: outputs for role in snapshots},
        'independent_administration': False, 'qualification_status': 'NOT_EVALUATED'}
    assert validate_controller_replay(receipt, job, 'test-inventory', snapshots) is receipt
    for key, changed in [('schema', 'saal-adaptive-controller-replay-v1'),
                         ('tolerance', .001), ('numerical_metric_policy_digest', 'sha256:'+'0'*64)]:
        broken = copy.deepcopy(receipt); broken[key] = changed
        with pytest.raises(ValueError): validate_controller_replay(broken, job, 'test-inventory', snapshots)


def test_admitted_controller_sources_cannot_be_omitted_or_replaced(tmp_path):
    source = tmp_path/'controller.py'; source.write_text('value = 1\n')
    protocol = {'source_files': {'controller.py': file_hash(source)}}
    validate_sources(protocol, tmp_path, ('controller.py',))
    with pytest.raises(ValueError): validate_sources(protocol, tmp_path, ('evaluator.py',))
    source.write_text('value = 2\n')
    with pytest.raises(ValueError): validate_sources(protocol, tmp_path, ('controller.py',))


def test_new_report_does_not_relabel_the_actual_failed_v2_study():
    report = load_script('report_study_v3.py')
    with pytest.raises(ValueError, match='v3 public campaign'):
        report.summarize(PROJECT/'study-v2')


def test_report_resolves_relative_campaign_path_before_artifact_lookup(monkeypatch):
    report = load_script('report_study_v3.py')
    monkeypatch.chdir(ROOT)
    paths = []

    def stopped_before_model(path):
        paths.append(path)
        assert path.is_absolute()
        raise ValueError('test stops before any model access')

    monkeypatch.setattr(report, 'read_json', stopped_before_model)
    with pytest.raises(ValueError, match='test stops'):
        report.summarize(Path('research/dynamic-lora-v6/study-v2'))
    assert paths == [PROJECT/'study-v2/signed_protocol.json']


def test_new_report_requires_all_fifteen_runs_even_with_a_fresh_v3_policy(tmp_path):
    report = load_script('report_study_v3.py')
    key = Ed25519PrivateKey.generate()
    runs = [{'name': method.replace('-', '_'), 'method': method, 'seed': 60606,
             'job_digest': sha(method)} for method in ('fixed-r8', 'fixed-r12', 'adalora-12-to-8')]
    protocol = {'schema': 'saal-adaptive-campaign-v3', 'stage': 'STUDY', 'runs': runs,
        'job_digests': [run['job_digest'] for run in runs],
        'worker_identities': {'v3': worker_identity()}, 'numerical_metric_policy': POLICY,
        'numerical_metric_policy_digest': POLICY_DIGEST, 'qualification_status': 'NOT_EVALUATED',
        'external_deployment_allowed': False,
        'source_files': {name: file_hash(ROOT/name) for name in report.REQUIRED_SOURCES}}
    campaign = tmp_path/'campaign'; (campaign/'colab').mkdir(parents=True)
    (campaign/'controller.pub').write_text(key.public_key().public_bytes_raw().hex())
    (campaign/'signed_protocol.json').write_bytes(canonical(sign(protocol, key)))
    (campaign/'colab/registry-export.json').write_bytes(canonical({'protocol_digest': sha(protocol),
        'campaign_state': 'FAILED', 'provisioning_attempts': 1,
        'attempts': [{'state': 'INTERRUPTED'} for _ in runs]}))
    session = 'saal-'+'a'*20
    (campaign/'colab/config.json').write_bytes(canonical({'session': session}))
    (campaign/'colab/plan.json').write_bytes(canonical({'session': session, 'job_digest': sha(protocol)}))
    (campaign/'colab/monitor-state.json').write_bytes(canonical({'session': session,
        'job_digest': sha(protocol), 'phase': 'FAILED', 'cleanup_verified': True, 'terminal': True}))
    with pytest.raises(ValueError, match='complete admitted 15-run'):
        report.summarize(campaign)


def test_frozen_regression_rejects_legacy_numeric_result_instead_of_reusing_it(job, population):
    regression = load_script('frozen_regression_v3.py')
    rows = [record for record in population['records'] if record['split'] == 'regression']
    outputs = [prediction(row['expected_decision']) for row in rows]
    binding = {'inventory_sha256': 'test-inventory'}
    plan = {'version': 'test-only'}; identity = {'population': 'test-only'}
    value = {'schema': 'saal-adaptive-frozen-regression-v3', 'job_digest': sha(job),
        'worker_identity': worker_identity(), 'numerical_metric_policy': POLICY,
        'numerical_metric_policy_digest': POLICY_DIGEST, 'analysis_plan_digest': sha(plan),
        'candidate_binding': binding, 'population_identity': identity, 'population_digest': sha(rows),
        'frozen_inventory_sha256': 'test-inventory', 'qualification_status': 'NOT_EVALUATED',
        'external_deployment_allowed': False, 'checkpoint_selection_permitted': False,
        'measured_work': {'checkpoint_evaluations': 3, 'case_predictions': 3*len(rows)},
        'snapshots': {role: {'predictions': outputs, 'metrics': diagnostics(rows, outputs)}
                      for role in ('initial', 'best', 'terminal')}}
    assert regression.validate_analysis_result(value, plan, job, binding, rows, identity) is value
    value['snapshots']['terminal']['metrics']['brier_score'] = 0.0
    with pytest.raises(ValueError, match='metric consistency'):
        regression.validate_analysis_result(value, plan, job, binding, rows, identity)


@pytest.mark.parametrize('fault', ['missing_source_map', 'changed_source_hash',
    'changed_method', 'changed_seed', 'changed_stage'])
def test_campaign_rejects_unbound_controller_source_before_staging_or_spending(
        tmp_path, monkeypatch, job, fault):
    from saal_rsi import adaptive_campaign
    key = Ed25519PrivateKey.generate()
    protocol = {'schema': 'saal-adaptive-campaign-v3', 'stage': 'ENGINEERING',
        'qualification_status': 'NOT_EVALUATED',
        'external_deployment_allowed': False, 'worker_identities': {'v3': worker_identity()},
        'numerical_metric_policy': POLICY, 'numerical_metric_policy_digest': POLICY_DIGEST,
        'runs': [{'name': 'test_run', 'seed': job['seed'], 'method': job['adapter']['method'],
                  'job_digest': sha(job)}], 'job_digests': [sha(job)]}
    if fault == 'changed_source_hash':
        names = list((ROOT/'src/saal_rsi').glob('*.py'))+list((PROJECT/'metric-repair').glob('*.py'))
        protocol['source_files'] = {name.relative_to(ROOT).as_posix(): file_hash(name) for name in names}
        protocol['source_files']['src/saal_rsi/adaptive_campaign.py'] = '0'*64
    elif fault == 'changed_method': protocol['runs'][0]['method'] = 'fixed-r8'
    elif fault == 'changed_seed': protocol['runs'][0]['seed'] += 1
    elif fault == 'changed_stage': protocol['stage'] = 'STUDY'
    job_file = tmp_path/'job.json'; job_file.write_bytes(canonical(sign(job, key)))
    protocol_file = tmp_path/'protocol.json'; protocol_file.write_bytes(canonical(sign(protocol, key)))
    workspace, artifacts, ledger = tmp_path/'workspace', tmp_path/'artifacts', tmp_path/'budget.sqlite3'

    def forbidden(*args, **kwargs):
        pytest.fail('unbound controller source reached staging, budget or cloud execution')

    monkeypatch.setattr(adaptive_campaign, 'stage_worker', forbidden)
    monkeypatch.setattr(adaptive_campaign, 'AdaptiveRegistry', forbidden)
    monkeypatch.setattr(adaptive_campaign, 'execute_colab_plan', forbidden)
    expected = 'source' if fault in ('missing_source_map', 'changed_source_hash') else 'method, seed or stage'
    with pytest.raises(ValueError, match=expected):
        adaptive_campaign.execute(workspace, [job_file], protocol_file, ledger,
            key.public_key().public_bytes_raw().hex(), artifacts, tmp_path/'cache')
    assert not workspace.exists() and not artifacts.exists() and not ledger.exists()


@pytest.mark.parametrize('fault', [None, 'monitor_job', 'plan_job', 'monitor_session',
    'plan_session', 'missing_config', 'not_terminal', 'unconfirmed_cleanup', 'different_export'])
def test_cleanup_binds_exact_campaign_and_session_without_claiming_provider_attestation(tmp_path, fault):
    protocol = {'schema': 'test-only-payload'}
    session = 'saal-'+'a'*20
    config = {'session': session}
    plan = {'session': session, 'job_digest': sha(protocol)}
    monitor = {'session': session, 'job_digest': sha(protocol), 'phase': 'FAILED',
               'cleanup_verified': True, 'terminal': True}
    export = {'protocol_digest': sha(protocol), 'campaign_state': 'FAILED', 'provisioning_attempts': 1}
    if fault == 'monitor_job': monitor['job_digest'] = 'sha256:'+'0'*64
    elif fault == 'plan_job': plan['job_digest'] = 'sha256:'+'0'*64
    elif fault == 'monitor_session': monitor['session'] = 'saal-'+'b'*20
    elif fault == 'plan_session': plan['session'] = 'saal-'+'b'*20
    elif fault == 'not_terminal': monitor['terminal'] = False
    elif fault == 'unconfirmed_cleanup': monitor['cleanup_verified'] = False
    elif fault == 'different_export': export['protocol_digest'] = 'sha256:'+'0'*64
    colab = tmp_path/'colab'; colab.mkdir()
    for name, record in [('plan.json', plan), ('monitor-state.json', monitor)]:
        (colab/name).write_bytes(canonical(record))
    if fault != 'missing_config': (colab/'config.json').write_bytes(canonical(config))
    if fault is None:
        assert validate_cleanup(tmp_path, protocol, export) == monitor
    else:
        with pytest.raises(ValueError): validate_cleanup(tmp_path, protocol, export)
