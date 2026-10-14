"""Frozen public comparisons. These endpoints never issue qualifications."""
import random
from collections import defaultdict

from .epoch_metrics import diagnostics, prediction_equal
from .spec import sha


def family_pairs(records, predictions):
    if not prediction_equal(predictions, predictions, 'modernbert-seqcls-v1', 1e-4) or len(predictions) != len(records):
        raise ValueError('invalid or incomplete pair predictions')
    groups = defaultdict(list)
    for row, output in zip(records, predictions):
        groups[(row['task_family_id'], row['cluster_id'])].append((row, output))
    pairs = defaultdict(list)
    for (family, cluster), group in groups.items():
        if len(group) != 2 or len({r['expected_decision'] for r, _ in group}) != 2:
            raise ValueError('complete opposite-label pairs required for primary endpoint')
        pairs[family].append(all(r['expected_decision'] == p['decision'] for r, p in group))
    return {family: sum(v)/len(v) for family, v in sorted(pairs.items())}


def paired_comparison(records, run_predictions, *, bootstrap_seed=6006, iterations=5000):
    """Paired seed and family resampling, preserving the comparison pairing.

    Operator-generated public family IDs do not establish independent units.
    This interval is explicitly descriptive, not a protected inference result.
    """
    seeds = sorted(run_predictions)
    if seeds != [47, 83, 109, 157, 211]:
        raise ValueError('all five preregistered seeds, including failures, required')
    effects, guardrails = {}, []
    expected_methods = {'fixed-r8', 'fixed-r12', 'adalora-12-to-8'}
    for seed in seeds:
        runs = run_predictions[seed]
        if set(runs) != expected_methods:
            raise ValueError('complete matched methods required')
        pair_scores = {m: family_pairs(records, p) for m, p in runs.items()}
        families = sorted(pair_scores['fixed-r8'])
        effects[seed] = {f: pair_scores['adalora-12-to-8'][f]-pair_scores['fixed-r8'][f] for f in families}
        metrics = {m: diagnostics(records, p) for m, p in runs.items()}
        guardrails.append({'seed': seed, 'methods': metrics,
                           'worst_family_nonregression': metrics['adalora-12-to-8']['worst_family_accuracy'] >= metrics['fixed-r8']['worst_family_accuracy'],
                           'critical_nonincrease': metrics['adalora-12-to-8']['unauthorized_proposals'] <= metrics['fixed-r8']['unauthorized_proposals']})
    rng = random.Random(bootstrap_seed)
    values = []
    for _ in range(iterations):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        sampled_families = [rng.choice(families) for _ in families]
        values.append(sum(effects[s][f] for s in sampled_seeds for f in sampled_families)/(len(seeds)*len(families)))
    values.sort()
    mean = sum(v for row in effects.values() for v in row.values())/(len(seeds)*len(families))
    interval = [values[int(iterations*.025)], values[min(iterations-1, int(iterations*.975))]]
    return {'schema': 'saal-adaptive-public-comparison-v1', 'primary_endpoint': 'family-macro complete counterfactual pairs',
            'pair_gain': mean, 'descriptive_95_interval': interval, 'paired_family_seed_effects': effects,
            'guardrails': guardrails, 'development_screen': mean >= .05 and interval[0] > 0 and all(g['worst_family_nonregression'] and g['critical_nonincrease'] for g in guardrails),
            'retention': 'NOT_MEASURED_REQUIRES_SEQUENTIAL_PROTOCOL',
            'independent_generalization': 'NOT_ESTABLISHED', 'qualification_status': 'NOT_EVALUATED',
            'external_deployment_allowed': False, 'make_default_adapter': False,
            'population_digest': sha(records), 'bootstrap_seed': bootstrap_seed,
            'assumptions': 'Seed and family resampling is descriptive for correlated public generators. Independent/adjudicated final populations and protected attempt authority remain mandatory.'}


def evaluate_frozen(backend, root, job, records):
    """Regression only after training bundle validation and candidate freeze."""
    from pathlib import Path
    from .adaptive_artifacts import ROLES, read_json, file_hash
    root = Path(root)
    if not records or any(r['split'] != 'regression' for r in records):
        raise ValueError('post-freeze public regression population required')
    result = {'schema': 'saal-adaptive-frozen-regression-v1', 'job_digest': sha(job),
              'frozen_inventory_sha256': file_hash(root/'artifact_inventory.json'),
              'population_digest': sha(records), 'snapshots': {}, 'qualification_status': 'NOT_EVALUATED',
              'external_deployment_allowed': False, 'checkpoint_selection_permitted': False}
    for role in ROLES:
        backend.restore_checkpoint(root/(role+'.safetensors'), read_json(root/(role+'_rank_pattern.json')))
        outputs = backend.evaluate(records)
        result['snapshots'][role] = {'predictions': outputs, 'metrics': diagnostics(records, outputs)}
    return result

