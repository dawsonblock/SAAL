"""Preserve a failed study without repairing, qualifying or retrying its jobs.

Completed candidates retain their strict validators. An interrupted artifact
may be inspected for diagnosis but cannot enter the completed comparison.
"""
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'research/dynamic-lora-v6'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from frozen_regression import frozen_run
from saal_rsi.adaptive_artifacts import ROLES, diagnostics, file_hash, read_json
from saal_rsi.adaptive_dispatch import validate_bundle
from saal_rsi.epoch_metrics import prediction_equal
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical, sha


def differences(recorded, recomputed, path=''):
    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        if set(recorded) != set(recomputed):
            raise ValueError('metric fields differ')
        return [item for key in sorted(recorded)
                for item in differences(recorded[key], recomputed[key], path + '/' + key)]
    if recorded == recomputed:
        return []
    numeric = (type(recorded) in (int, float) and type(recomputed) in (int, float)
               and math.isfinite(recorded) and math.isfinite(recomputed))
    return [{'path': path, 'recorded': recorded, 'recomputed': recomputed,
             'absolute_delta': abs(recorded - recomputed) if numeric else None,
             'ulps_at_recorded': abs(recorded - recomputed) / math.ulp(recorded)
             if numeric and type(recorded) is float else None}]


def inspect_interrupted(bundle, job, pin):
    """Transport/GPU replay checks are diagnostics, never admission overrides."""
    bundle = Path(bundle)
    inventory = read_json(bundle / 'artifact_inventory.json')
    for name, item in inventory['files'].items():
        if Path(name).name != name:
            raise ValueError('unsafe inventory path')
        path = bundle / name
        if path.is_symlink() or path.stat().st_size != item['size'] or file_hash(path) != item['sha256']:
            raise ValueError('recovered physical artifact changed')
    result = read_json(bundle / 'result.json')
    if sha(verify(read_json(bundle / 'signed_training_job.json'), pin)) != sha(job) or result['job_digest'] != sha(job):
        raise ValueError('interrupted job substitution')
    try:
        validate_bundle(bundle, job, pin, require_replay=True)
    except ValueError as exc:
        rejection = str(exc)
    else:
        raise ValueError('the preserved rejection no longer reproduces')
    development = [r for r in job['dataset']['records'] if r['split'] == 'development']
    mismatch = {role: differences(result['snapshots'][role]['metrics'],
                                 diagnostics(development, result['snapshots'][role]['predictions']))
                for role in ROLES}
    gpu = read_json(bundle / 'fresh_process_replay.json')
    gpu_agrees = (gpu['job_digest'] == sha(job)
                  and gpu['inventory_sha256'] == file_hash(bundle / 'artifact_inventory.json')
                  and gpu['tolerance'] == job['replay_tolerance']
                  and gpu['verified'] is True
                  and all(prediction_equal(gpu['checkpoints'][role], result['snapshots'][role]['predictions'],
                                           'modernbert-seqcls-v1', job['replay_tolerance']) for role in ROLES))
    if not gpu_agrees:
        raise ValueError('recovered GPU replay does not agree')
    return {'job_digest': sha(job), 'inventory_sha256': file_hash(bundle / 'artifact_inventory.json'),
            'physical_inventory_files_verified': len(inventory['files']),
            'strict_validation_rejection': rejection, 'metric_differences': mismatch,
            'gpu_replay_predictions_agree': True, 'controller_cpu_replay': 'NOT_COMPLETED',
            'worker_reported_resources': result['resource_accounting'],
            'admitted_as_completed': False, 'original_state_preserved': 'INTERRUPTED'}


def close_failed_study():
    campaign = PROJECT / 'study-v2'
    pin = (campaign / 'controller.pub').read_text().strip()
    protocol = verify(read_json(campaign / 'signed_protocol.json'), pin)
    export = read_json(campaign / 'colab/registry-export.json')
    monitor = read_json(campaign / 'colab/monitor-state.json')
    if (export['protocol_digest'] != sha(protocol) or export['campaign_state'] != 'FAILED'
            or monitor['phase'] != 'FAILED' or monitor['terminal'] is not True
            or monitor['cleanup_verified'] is not True or len(export['attempts']) != 15):
        raise ValueError('exact failed admission and confirmed cleanup required')
    for name, digest in protocol['source_files'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError('admitted source changed: ' + name)
    foundation = read_json(PROJECT / 'FOUNDATION.json')
    for name, digest in foundation['protected_v5_files'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError('protected historical evidence changed: ' + name)
    states, completed, interrupted, attempts = {}, [], [], []
    for descriptor, row in zip(protocol['runs'], export['attempts']):
        job = verify(read_json(campaign / (descriptor['name'] + '-job.json')), pin)
        if sha(job) != row['digest'] or sha(job) != descriptor['job_digest']:
            raise ValueError('attempt identity substitution')
        states[row['state']] = states.get(row['state'], 0) + 1
        attempts.append({'method': descriptor['method'], 'seed': descriptor['seed'],
                         'job_digest': sha(job), 'state': row['state']})
        if row['state'] == 'COMPLETED':
            _, bundle, native, binding = frozen_run(campaign, descriptor, row)
            regression = PROJECT / 'regression-study-v2' / sha(job)[7:19]
            outcome, public = read_json(regression / 'outcome.json'), read_json(regression / 'result.json')
            if (outcome['state'] != 'COMPLETED' or outcome['result_sha256'] != file_hash(regression / 'result.json')
                    or public['candidate_binding'] != binding or outcome['candidate_binding'] != binding
                    or public['checkpoint_selection_permitted'] is not False):
                raise ValueError('completed public regression lineage')
            completed.append({'method': descriptor['method'], 'seed': descriptor['seed'], 'binding': binding,
                'development': {role: {'step': native['snapshots'][role]['step'],
                                      'metrics': native['snapshots'][role]['metrics']} for role in ROLES},
                'public_regression': {role: public['snapshots'][role]['metrics'] for role in ROLES},
                'regression_result_sha256': outcome['result_sha256'],
                'regression_measured_work': public['measured_work'],
                'training_resource_accounting': native['resource_accounting'], 'allocation': native['allocation']})
        elif row['state'] == 'INTERRUPTED':
            interrupted.append(inspect_interrupted(campaign / 'artifacts' / row['outcome']['run'], job, pin))
        elif row['state'] != 'RESERVED':
            raise ValueError('unexpected terminal attempt state')
    with sqlite3.connect('file:' + str(campaign / 'budget.sqlite3') + '?mode=ro', uri=True) as db:
        if dict(db.execute('SELECT state,COUNT(*) FROM attempts GROUP BY state')) != states:
            raise ValueError('SQLite / export mismatch')
        if db.execute('SELECT state,provisions FROM budget').fetchall() != [('FAILED', 1)]:
            raise ValueError('failed consumed budget changed')
    if states != {'COMPLETED': 10, 'INTERRUPTED': 1, 'RESERVED': 4}:
        raise ValueError('unexpected study closure; review instead of changing counts')
    spec = importlib.util.spec_from_file_location('report_study', Path(__file__).with_name('report_study.py'))
    reporting = importlib.util.module_from_spec(spec); spec.loader.exec_module(reporting)
    try:
        reporting.summarize(campaign)
    except ValueError as exc:
        comparison_rejection = str(exc)
    else:
        raise ValueError('incomplete study unexpectedly accepted')
    prior = {name: read_json(PROJECT / name / 'colab/registry-export.json')['campaign_state']
             for name in ('pilot-experiment', 'pilot-repair-02')}
    if prior != {'pilot-experiment': 'FAILED', 'pilot-repair-02': 'FAILED'}:
        raise ValueError('old failed campaigns changed')
    if (PROJECT / 'data-control-v2').exists():
        raise ValueError('parent controls must remain unstarted after failed prerequisite')
    return {'schema': 'saal-adaptive-failed-study-closure-v1', 'version': '0.9.5.dev6',
            'campaign_state': 'FAILED', 'protocol_digest': sha(protocol),
            'source_commit': protocol['source_commit'], 'attempt_counts': states, 'attempts': attempts,
            'completed_runs': completed, 'interrupted_artifacts': interrupted,
            'registered_five_seed_comparison': 'INCOMPLETE_NOT_COMPUTED',
            'comparison_gate_rejection': comparison_rejection, 'new_training_retries': 0,
            'parent_data_controls': 'NOT_STARTED_FAILED_PREREQUISITE',
            'public_regression_completed': len(completed), 'cleanup_verified': True,
            'session': monitor['session'], 'older_failed_campaigns': prior,
            'protected_v5_files_unchanged': True, 'five_original_v5_jobs': 'RESERVED_UNCHANGED',
            'qualification_status': 'NOT_EVALUATED', 'independent_generalization': 'NOT_ESTABLISHED',
            'retention': 'NOT_MEASURED_REQUIRES_SEQUENTIAL_PROTOCOL', 'external_deployment_allowed': False,
            'make_default_adapter': False,
            'scope': 'Descriptive incomplete public research evidence. No registered five-seed inference, independent custody or promotion. Interrupted scores are not treated as accepted evidence.'}


if __name__ == '__main__':
    output = Path(sys.argv[1])
    value = close_failed_study()
    with output.open('xb') as stream:
        stream.write(canonical(value))
    print(json.dumps({k: value[k] for k in ('campaign_state', 'attempt_counts', 'public_regression_completed',
                                          'registered_five_seed_comparison', 'cleanup_verified')}))
