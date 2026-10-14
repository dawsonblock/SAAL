"""Reconstruct the preregistered public comparison from all 15 frozen runs.

No checkpoint selection, training, authority change or protected-data access.
Seed/family intervals describe these correlated public generators only.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.adaptive_artifacts import read_json, file_hash
from saal_rsi.adaptive_dispatch import validate_bundle
from saal_rsi.adaptive_evaluator import paired_comparison
from saal_rsi.epoch_metrics import prediction_equal
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical, sha


def summarize(folder):
    folder = Path(folder)
    pin = (folder/'controller.pub').read_text().strip()
    protocol = verify(read_json(folder/'signed_protocol.json'), pin)
    export = read_json(folder/'colab/registry-export.json')
    monitor = read_json(folder/'colab/monitor-state.json')
    if (protocol['stage'] != 'STUDY' or export['protocol_digest'] != sha(protocol)
            or export['campaign_state'] != 'COMPLETED' or monitor['phase'] != 'COMPLETED'
            or monitor['cleanup_verified'] is not True or len(export['attempts']) != 15
            or len(protocol['runs']) != 15):
        raise ValueError('complete admitted 15-run study and confirmed cleanup required')
    runs, records, results, matching = defaultdict(dict), None, [], defaultdict(list)
    for descriptor, row in zip(protocol['runs'], export['attempts']):
        job = verify(read_json(folder/(descriptor['name']+'-job.json')), pin)
        digest = sha(job)
        outcome = row['outcome']
        if row['state'] != 'COMPLETED' or row['digest'] != digest or descriptor['job_digest'] != digest or outcome['job_digest'] != digest:
            raise ValueError('failed or substituted attempt cannot disappear from study')
        name = outcome['run']
        import re
        if not re.fullmatch(r'run-[0-9]{2}-'+digest[7:19], name):
            raise ValueError('artifact run identity')
        artifact = folder/'artifacts'/name
        value = validate_bundle(artifact, job, pin, require_replay=True)
        receipt = read_json(folder/'colab'/(name+'-controller-replay.json'))
        if (outcome['result_digest'] != sha(value) or outcome['controller_replay_digest'] != sha(receipt)
                or receipt['verified'] is not True or receipt['job_digest'] != digest
                or receipt['inventory_sha256'] != file_hash(artifact/'artifact_inventory.json')
                or any(not prediction_equal(receipt['predictions'][role], value['snapshots'][role]['predictions'], 'modernbert-seqcls-v1', 1e-4)
                       for role in ('initial', 'best', 'terminal'))):
            raise ValueError('study replay lineage mismatch')
        if value['resource_accounting']['sample_exposures'] != 16000 or job['steps'] != 1000:
            raise ValueError('unmatched update / sample exposure budget')
        development = [r for r in job['dataset']['records'] if r['split'] == 'development']
        if records is not None and records != development:
            raise ValueError('development population substitution')
        records = development
        seed, method = job['seed'], job['adapter']['method']
        if method in runs[seed]: raise ValueError('duplicate method / seed')
        runs[seed][method] = value['snapshots']['terminal']['predictions']
        matching[seed].append((value['unadapted_model_hash'], sha(read_json(artifact/'sampler_manifest.json')['record_ids']), job['dataset_digest']))
        results.append({'method': method, 'seed': seed, 'job_digest': digest,
            'artifact_path': artifact.relative_to(ROOT).as_posix(), 'inventory_sha256': file_hash(artifact/'artifact_inventory.json'),
            'snapshots': {role: {'step': s['step'], 'metrics': s['metrics']} for role, s in value['snapshots'].items()},
            'allocation': value['allocation'], 'resource_accounting': value['resource_accounting']})
    if any(len(values) != 3 or len(set(values)) != 1 for values in matching.values()):
        raise ValueError('base/head, sampling or data mismatch within seed')
    comparison = paired_comparison(records, runs)
    if comparison['primary_endpoint'] != 'family-macro complete counterfactual pairs':
        raise ValueError('primary endpoint drift')
    return {'schema': 'saal-adaptive-study-report-v1', 'protocol_digest': sha(protocol),
        'source_commit': protocol['source_commit'], 'terminal_comparison': comparison, 'runs': results,
        'all_15_admitted_runs_accounted': True, 'cleanup_verified': True,
        'engineering_seed_included_in_science': False, 'posthoc_checkpoint_selection': False,
        'retention': 'NOT_MEASURED_REQUIRES_SEPARATE_SEQUENTIAL_PROTOCOL',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False, 'make_default_adapter': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('folder'); parser.add_argument('output')
    args = parser.parse_args(); value = summarize(args.folder)
    with Path(args.output).open('xb') as stream: stream.write(canonical(value))
    print(json.dumps({'all_runs': len(value['runs']), 'comparison': value['terminal_comparison']}))
