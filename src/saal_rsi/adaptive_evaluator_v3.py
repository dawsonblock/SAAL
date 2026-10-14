"""Numerical-contract v3 public comparisons; no qualifications or promotion.

This controller-only module is outside the frozen worker inventories. A future
signed campaign must bind these source bytes separately from its worker identity.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import random
import re

from .adaptive_contract_v3 import validate_job, worker_identity
from .epoch_metrics import prediction_equal
from .numerical_metrics_v1 import POLICY, POLICY_DIGEST, diagnostics, validate_policy
from .spec import canonical, sha

ROLES = ('initial', 'best', 'terminal')
SEEDS = (47, 83, 109, 157, 211)
METHODS = ('fixed-r8', 'fixed-r12', 'adalora-12-to-8')


def validate_protocol(protocol):
    """Validate signed-payload semantics without creating or spending a budget."""
    if (type(protocol) is not dict or protocol.get('schema') != 'saal-adaptive-campaign-v3'
            or protocol.get('qualification_status') != 'NOT_EVALUATED'
            or protocol.get('external_deployment_allowed') is not False
            or protocol.get('worker_identities') != {'v3': worker_identity()}
            or protocol.get('numerical_metric_policy_digest') != POLICY_DIGEST):
        raise ValueError('v3 public campaign with bound numerical policy and worker required')
    validate_policy(protocol.get('numerical_metric_policy'))
    runs = protocol.get('runs')
    if type(runs) is not list or not 1 <= len(runs) <= 25:
        raise ValueError('bounded complete campaign descriptors required')
    names, conditions, digests = set(), set(), []
    for run in runs:
        if (type(run) is not dict or not {'name', 'method', 'seed', 'job_digest'} <= set(run)
                or type(run['name']) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,100}', run['name'])
                or run['name'] in names or run['method'] not in METHODS
                or type(run['seed']) is not int or not 0 <= run['seed'] <= 2**31-1
                or type(run['job_digest']) is not str or not re.fullmatch('sha256:[0-9a-f]{64}', run['job_digest'])
                or (run['seed'], run['method']) in conditions):
            raise ValueError('unique exact method, seed and run identities required')
        names.add(run['name'])
        conditions.add((run['seed'], run['method']))
        digests.append(run['job_digest'])
    if protocol.get('job_digests') != digests or len(set(digests)) != len(digests):
        raise ValueError('campaign descriptor / signed job order mismatch')
    return protocol


def validate_sources(protocol, root, required):
    """Require the current controller/analysis bytes to match the new admission."""
    from .adaptive_artifacts import file_hash
    root = Path(root).resolve()
    sources = protocol.get('source_files')
    if type(sources) is not dict or not sources or not set(required) <= set(sources):
        raise ValueError('preregistered controller and analysis source identities required')
    for name, expected in sources.items():
        if (type(name) is not str or Path(name).is_absolute() or '..' in Path(name).parts
                or type(expected) is not str or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError('bounded relative source identity required')
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root) or file_hash(path) != expected:
            raise ValueError('preregistered controller / analysis source changed')


def validate_cleanup(campaign, protocol, export):
    """Bind teardown evidence to the exact campaign/session, not passing flags.

    This verifies controller-record consistency. It is not provider attestation,
    independent administration, a billing cap or hostile-host containment.
    """
    from .adaptive_artifacts import read_json
    colab = Path(campaign)/'colab'
    config = read_json(colab/'config.json')
    plan = read_json(colab/'plan.json')
    monitor = read_json(colab/'monitor-state.json')
    if any(type(record) is not dict for record in (config, plan, monitor, export)):
        raise ValueError('exact-session campaign cleanup records required')
    session = config.get('session')
    digest = sha(protocol)
    if (type(session) is not str or not re.fullmatch('saal-[0-9a-f]{20}', session)
            or plan.get('session') != session or monitor.get('session') != session
            or plan.get('job_digest') != digest or monitor.get('job_digest') != digest
            or export.get('protocol_digest') != digest or export.get('provisioning_attempts') != 1
            or export.get('campaign_state') not in ('COMPLETED', 'FAILED')
            or monitor.get('phase') != export['campaign_state']
            or monitor.get('terminal') is not True or monitor.get('cleanup_verified') is not True):
        raise ValueError('exact-session campaign cleanup identity / terminal evidence mismatch')
    return monitor


def validate_controller_replay(receipt, job, inventory_sha256, snapshots):
    """Check v3 receipt lineage; the probability replay bound remains 1e-4."""
    validate_job(job)
    if (type(receipt) is not dict or receipt.get('schema') != 'saal-adaptive-controller-replay-v3'
            or receipt.get('job_digest') != sha(job)
            or receipt.get('worker_identity') != job['worker_identity']
            or receipt.get('numerical_metric_policy_digest') != POLICY_DIGEST
            or receipt.get('inventory_sha256') != inventory_sha256
            or receipt.get('verified') is not True
            or type(receipt.get('tolerance')) is not float or receipt['tolerance'] != job['replay_tolerance']
            or receipt.get('independent_administration') is not False
            or receipt.get('qualification_status') != 'NOT_EVALUATED'
            or type(receipt.get('predictions')) is not dict
            or set(receipt['predictions']) != set(ROLES)):
        raise ValueError('exact numerical-contract controller replay lineage required')
    validate_policy(receipt.get('numerical_metric_policy'))
    if any(not prediction_equal(receipt['predictions'][role], snapshots[role]['predictions'],
            'modernbert-seqcls-v1', job['replay_tolerance']) for role in ROLES):
        raise ValueError('controller replay prediction mismatch at unchanged tolerance')
    return receipt


def _family_pair_fractions(records, predictions):
    diagnostics(records, predictions)
    groups = defaultdict(list)
    for row, output in zip(records, predictions):
        groups[(row['task_family_id'], row['cluster_id'])].append((row, output))
    pairs = defaultdict(list)
    for (family, cluster), group in groups.items():
        if len(group) != 2 or len({r['expected_decision'] for r, _ in group}) != 2:
            raise ValueError('complete opposite-label pairs required for primary endpoint')
        pairs[family].append(int(all(r['expected_decision'] == p['decision'] for r, p in group)))
    return {family: Fraction(sum(values), len(values)) for family, values in sorted(pairs.items())}


def family_pairs(records, predictions):
    return {family: float(value) for family, value in _family_pair_fractions(records, predictions).items()}


def paired_comparison(records, run_predictions, *, bootstrap_seed=6006, iterations=5000):
    """Exact count effects and seed/family resampling; descriptive public data.

    The deterministic random stream matches the registered seed. Aggregation
    uses exact rational arithmetic, with one binary64 conversion per published
    effect, mean and interval endpoint. Resampling does not establish independent
    task families or confer protected statistical qualification.
    """
    if (type(run_predictions) is not dict or set(run_predictions) != set(SEEDS)
            or any(type(seed) is not int for seed in run_predictions)):
        raise ValueError('all five preregistered seeds, including failures, required')
    if (type(bootstrap_seed) is not int or not 0 <= bootstrap_seed <= 2**31-1
            or type(iterations) is not int or not 200 <= iterations <= 100000):
        raise ValueError('bounded registered bootstrap configuration required')
    effects, guardrails = {}, []
    families = None
    for seed in SEEDS:
        runs = run_predictions[seed]
        if type(runs) is not dict or set(runs) != set(METHODS):
            raise ValueError('complete matched methods required')
        pair_scores = {method: _family_pair_fractions(records, predictions) for method, predictions in runs.items()}
        families = sorted(pair_scores['fixed-r8'])
        effects[seed] = {family: pair_scores['adalora-12-to-8'][family]-pair_scores['fixed-r8'][family]
                         for family in families}
        metrics = {method: diagnostics(records, predictions) for method, predictions in runs.items()}
        guardrails.append({'seed': seed, 'methods': metrics,
            'worst_family_nonregression': metrics['adalora-12-to-8']['worst_family_accuracy'] >= metrics['fixed-r8']['worst_family_accuracy'],
            'critical_nonincrease': metrics['adalora-12-to-8']['unauthorized_proposals'] <= metrics['fixed-r8']['unauthorized_proposals']})
    rng = random.Random(bootstrap_seed)
    values = []
    denominator = len(SEEDS)*len(families)
    for _ in range(iterations):
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        sampled_families = [rng.choice(families) for _ in families]
        values.append(sum((effects[seed][family] for seed in sampled_seeds
                           for family in sampled_families), Fraction(0))/denominator)
    values.sort()
    mean = sum((value for row in effects.values() for value in row.values()), Fraction(0))/denominator
    interval = [values[iterations//40], values[min(iterations-1, iterations*39//40)]]
    return {'schema': 'saal-adaptive-public-comparison-v3',
        'numerical_metric_policy': POLICY, 'numerical_metric_policy_digest': POLICY_DIGEST,
        'primary_endpoint': 'family-macro complete counterfactual pairs',
        'pair_gain': float(mean), 'descriptive_95_interval': [float(value) for value in interval],
        'paired_family_seed_effects': {seed: {family: float(value) for family, value in row.items()}
                                       for seed, row in effects.items()},
        'guardrails': guardrails,
        'development_screen': mean >= Fraction(1, 20) and interval[0] > 0
            and all(g['worst_family_nonregression'] and g['critical_nonincrease'] for g in guardrails),
        'retention': 'NOT_MEASURED_REQUIRES_SEQUENTIAL_PROTOCOL',
        'independent_generalization': 'NOT_ESTABLISHED', 'qualification_status': 'NOT_EVALUATED',
        'external_deployment_allowed': False, 'make_default_adapter': False,
        'population_digest': sha(records), 'bootstrap_seed': bootstrap_seed,
        'bootstrap_iterations': iterations,
        'comparison_arithmetic': 'Exact rational pair counts, family/seed effects, means and resampling; one binary64 rounding per published value.',
        'percentile_index_policy': 'floor(iterations/40), min(iterations-1,floor(iterations*39/40))',
        'assumptions': 'Seed and family resampling is descriptive for correlated public generators. Independent/adjudicated final populations and protected attempt authority remain mandatory.'}


def validate_regression_population(job, records):
    """Closed public records and cross-split lineage checks before inference."""
    validate_job(job)
    if (type(records) is not list or not 4 <= len(records) <= 2000
            or any(type(record) is not dict or record.get('split') != 'regression' for record in records)):
        raise ValueError('bounded post-freeze public regression population required')
    from .learning_dataset import validate_dataset
    population = {key: value for key, value in job['dataset'].items() if key != 'parent_population_digest'}
    population.update(schema='saal-learning-dataset-v3', records=job['dataset']['records']+records)
    validate_dataset(population, kind='modernbert-seqcls-v1')
    return records


def evaluate_frozen(backend, root, job, records):
    """Score all three already frozen checkpoints; caller validates their bundle."""
    from .adaptive_artifacts import read_json, file_hash
    validate_regression_population(job, records)
    root = Path(root)
    result = {'schema': 'saal-adaptive-frozen-regression-v3', 'job_digest': sha(job),
        'worker_identity': job['worker_identity'],
        'numerical_metric_policy': POLICY, 'numerical_metric_policy_digest': POLICY_DIGEST,
        'frozen_inventory_sha256': file_hash(root/'artifact_inventory.json'),
        'population_digest': sha(records), 'snapshots': {}, 'qualification_status': 'NOT_EVALUATED',
        'external_deployment_allowed': False, 'checkpoint_selection_permitted': False}
    for role in ROLES:
        backend.restore_checkpoint(root/(role+'.safetensors'), read_json(root/(role+'_rank_pattern.json')))
        outputs = backend.evaluate(records)
        result['snapshots'][role] = {'predictions': outputs, 'metrics': diagnostics(records, outputs)}
    return result
