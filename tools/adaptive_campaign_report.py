"""Report trained artifacts separately from admitted, replayed job completion."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.adaptive_artifacts import read_json, file_hash
from saal_rsi.adaptive_dispatch import validate_bundle
from saal_rsi.epoch_metrics import prediction_equal
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical, sha


def replay_differences(actual, expected, tolerance=1e-4):
    if (not prediction_equal(actual, actual, 'modernbert-seqcls-v1', tolerance)
            or not prediction_equal(expected, expected, 'modernbert-seqcls-v1', tolerance)
            or len(actual) != len(expected) or not actual):
        raise ValueError('complete valid replay predictions required')
    deltas = [max(abs(x-y) for x, y in zip(a['probabilities'], b['probabilities']))
              for a, b in zip(actual, expected)]
    return {'cases': len(actual), 'changed_decisions': sum(a['decision'] != b['decision'] for a, b in zip(actual, expected)),
            'max_probability_delta': max(deltas), 'cases_exceeding_tolerance': sum(d > tolerance for d in deltas),
            'tolerance': tolerance,
            'passed': prediction_equal(actual, expected, 'modernbert-seqcls-v1', tolerance)}


def summarize(folder):
    folder = Path(folder).resolve()
    pin = (folder/'controller.pub').read_text().strip()
    protocol = verify(read_json(folder/'signed_protocol.json'), pin)
    expected = set(protocol['job_digests'])
    state = read_json(folder/'colab/monitor-state.json')
    export = read_json(folder/'colab/registry-export.json')
    if export['protocol_digest'] != sha(protocol):
        raise ValueError('campaign identity mismatch')
    completed = {r['digest']: r['state'] for r in export['attempts']}
    runs = []
    artifacts = folder/'artifacts'
    for path in sorted(artifacts.iterdir()) if artifacts.exists() else []:
        if not path.is_dir():
            continue
        envelope = read_json(path/'signed_training_job.json')
        job = verify(envelope, pin)
        digest = sha(job)
        if digest not in expected:
            raise ValueError('cross-campaign report artifact')
        result = validate_bundle(path, job, pin, require_replay=True)
        cpu_path = folder/'colab'/(path.name+'-controller-replay.json')
        cpu = read_json(cpu_path) if cpu_path.exists() else None
        comparisons = {}
        if cpu is not None:
            if cpu['job_digest'] != digest or cpu['inventory_sha256'] != file_hash(path/'artifact_inventory.json'):
                raise ValueError('CPU replay receipt substitution')
            for role in ('initial', 'best', 'terminal'):
                comparisons[role] = replay_differences(cpu['predictions'][role], result['snapshots'][role]['predictions'], job['replay_tolerance'])
        cpu_ok = bool(comparisons) and all(r['passed'] for r in comparisons.values())
        if cpu is not None and cpu['verified'] is not cpu_ok:
            raise ValueError('CPU replay assertion contradicts predictions')
        rows = [json.loads(line) for line in (path/'rank_history.jsonl').read_text().splitlines()]
        rank_samples = {str(r['completed_optimizer_step']): r['active_rank_by_layer'] for r in rows
                        if r['completed_optimizer_step'] in (100, 250, 500, 800, 1000)}
        runs.append({'job_digest': digest, 'method': job['adapter']['method'], 'seed': job['seed'],
            'registry_state': completed[digest], 'trained_optimizer_updates': job['steps'],
            'sample_exposures': result['resource_accounting']['sample_exposures'],
            'artifact_inventory_sha256': file_hash(path/'artifact_inventory.json'),
            'artifact_path': path.relative_to(ROOT).as_posix(),
            'unadapted_model_hash': result['unadapted_model_hash'],
            'allocation': result['allocation'], 'rank_samples': rank_samples,
            'resource_accounting': result['resource_accounting'],
            'fresh_gpu_replay': True, 'fresh_cpu_replay': cpu_ok, 'cpu_replay_differences': comparisons,
            'snapshots': {role: {'step': r['step'], 'metrics': r['metrics']}
                          for role, r in result['snapshots'].items()}})
    gate, reason = False, None
    if protocol['stage'] == 'PILOT':
        source = ROOT/'research/dynamic-lora-v6/prepare.py'
        spec = importlib.util.spec_from_file_location('adaptive_pilot_gate', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            module.pilot_accepted(folder)
            gate = True
        except (OSError, KeyError, TypeError, ValueError) as exc:
            reason = str(exc)
    return {'schema': 'saal-adaptive-campaign-report-v1', 'stage': protocol['stage'],
        'protocol_digest': sha(protocol), 'admission_source_commit': protocol['source_commit'],
        'campaign_state': export['campaign_state'], 'session': state['session'],
        'cleanup_verified': state['cleanup_verified'], 'budget_export': export,
        'runs_with_validated_training_artifacts': len(runs), 'runs': runs,
        'pilot_gate_accepted': gate, 'pilot_block_reason': reason,
        'scientific_comparison_completed': False,
        'scope': 'Pilot seed is excluded from scientific analysis. Training completion does not imply replay admission or reliable behavior.',
        'retention': 'NOT_MEASURED', 'independent_generalization': 'NOT_ESTABLISHED',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False,
        'make_default_adapter': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder'); p.add_argument('output')
    args = p.parse_args()
    result = summarize(args.folder)
    with Path(args.output).open('xb') as stream:
        stream.write(canonical(result))
    print(json.dumps({'runs_with_validated_training_artifacts': len(result['runs']),
        'campaign_state': result['campaign_state'], 'pilot_gate_accepted': result['pilot_gate_accepted'],
        'qualification_status': 'NOT_EVALUATED'}))
