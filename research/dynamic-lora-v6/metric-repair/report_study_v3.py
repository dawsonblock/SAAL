"""Reconstruct only a newly admitted, complete 15-run numerical-contract v3 study.

No missing runs, retries, checkpoint selection, training or protected evaluation
can be supplied by this report. Historical incomplete studies remain rejected.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.adaptive_artifacts import read_json, file_hash
from saal_rsi.adaptive_contract_v3 import validate_job
from saal_rsi.adaptive_dispatch import validate_bundle
from saal_rsi.adaptive_evaluator_v3 import (paired_comparison, validate_controller_replay,
    validate_protocol, validate_sources, validate_cleanup)
from saal_rsi.model_contract import verify
from saal_rsi.numerical_metrics_v1 import POLICY, POLICY_DIGEST
from saal_rsi.spec import canonical, sha

REQUIRED_SOURCES = (
    'src/saal_rsi/adaptive_evaluator_v3.py', 'src/saal_rsi/adaptive_replay_v3.py',
    'src/saal_rsi/adaptive_dispatch.py', 'src/saal_rsi/adaptive_campaign.py',
    'src/saal_rsi/adaptive_registry.py',
    'research/dynamic-lora-v6/metric-repair/report_study_v3.py',
)


def summarize(folder):
    folder = Path(folder).resolve()
    pin = (folder/'controller.pub').read_text().strip()
    protocol = validate_protocol(verify(read_json(folder/'signed_protocol.json'), pin))
    validate_sources(protocol, ROOT, REQUIRED_SOURCES)
    export = read_json(folder/'colab/registry-export.json')
    monitor = validate_cleanup(folder, protocol, export)
    if (protocol.get('stage') != 'STUDY' or export.get('protocol_digest') != sha(protocol)
            or export.get('campaign_state') != 'COMPLETED' or monitor.get('phase') != 'COMPLETED'
            or monitor.get('cleanup_verified') is not True or export.get('provisioning_attempts') != 1
            or type(export.get('attempts')) is not list or len(export['attempts']) != 15
            or len(protocol['runs']) != 15):
        raise ValueError('complete admitted 15-run v3 study and confirmed cleanup required')
    runs, records, results, matching = defaultdict(dict), None, [], defaultdict(list)
    for descriptor, row in zip(protocol['runs'], export['attempts']):
        job = validate_job(verify(read_json(folder/(descriptor['name']+'-job.json')), pin))
        digest = sha(job)
        outcome = row.get('outcome')
        if (row.get('state') != 'COMPLETED' or not outcome or outcome.get('state') != 'COMPLETED'
                or row.get('digest') != digest or descriptor['job_digest'] != digest
                or outcome.get('job_digest') != digest or descriptor['seed'] != job['seed']
                or descriptor['method'] != job['adapter']['method']
                or outcome.get('seed') != job['seed'] or outcome.get('method') != job['adapter']['method']):
            raise ValueError('failed, missing or substituted attempt cannot disappear from study')
        name = outcome.get('run')
        if type(name) is not str or not re.fullmatch(r'run-[0-9]{2}-'+digest[7:19], name):
            raise ValueError('artifact run identity')
        artifact = folder/'artifacts'/name
        value = validate_bundle(artifact, job, pin, require_replay=True)
        receipt = read_json(folder/'colab'/(name+'-controller-replay.json'))
        inventory = file_hash(artifact/'artifact_inventory.json')
        validate_controller_replay(receipt, job, inventory, value['snapshots'])
        if outcome.get('result_digest') != sha(value) or outcome.get('controller_replay_digest') != sha(receipt):
            raise ValueError('study replay result / receipt lineage mismatch')
        if value['resource_accounting']['sample_exposures'] != 16000 or job['steps'] != 1000:
            raise ValueError('unmatched update / sample exposure budget')
        development = [record for record in job['dataset']['records'] if record['split'] == 'development']
        if records is not None and records != development:
            raise ValueError('development population substitution')
        records = development
        seed, method = job['seed'], job['adapter']['method']
        if method in runs[seed]:
            raise ValueError('duplicate method / seed')
        runs[seed][method] = value['snapshots']['terminal']['predictions']
        common = ('dataset_digest', 'sampler_seed', 'dropout_seed', 'steps', 'batch_size',
            'accumulation', 'learning_rate', 'head_learning_rate', 'class_balance',
            'checkpoint_interval', 'selection_policy', 'peft_implementation',
            'replay_tolerance', 'numerical_metric_policy_digest')
        matching[seed].append((value['unadapted_model_hash'],
            sha(read_json(artifact/'sampler_manifest.json')['record_ids']),
            sha({key: job[key] for key in common}),
            sha({key: item for key, item in job['model_spec'].items() if key != 'resource_budget_id'})))
        results.append({'method': method, 'seed': seed, 'job_digest': digest,
            'artifact_path': artifact.relative_to(ROOT).as_posix(), 'inventory_sha256': inventory,
            'snapshots': {role: {'step': snapshot['step'], 'metrics': snapshot['metrics']}
                          for role, snapshot in value['snapshots'].items()},
            'allocation': value['allocation'], 'resource_accounting': value['resource_accounting']})
    if any(len(values) != 3 or len(set(values)) != 1 for values in matching.values()):
        raise ValueError('base/head, sampling, data or registered learning condition mismatch within seed')
    comparison = paired_comparison(records, runs)
    return {'schema': 'saal-adaptive-study-report-v3', 'protocol_digest': sha(protocol),
        'numerical_metric_policy': POLICY, 'numerical_metric_policy_digest': POLICY_DIGEST,
        'source_commit': protocol['source_commit'], 'terminal_comparison': comparison, 'runs': results,
        'all_15_admitted_runs_accounted': True, 'cleanup_verified': True,
        'engineering_seed_included_in_science': False, 'posthoc_checkpoint_selection': False,
        'retention': 'NOT_MEASURED_REQUIRES_SEPARATE_SEQUENTIAL_PROTOCOL',
        'mechanism': 'NOT_MEASURED_REQUIRES_SEPARATE_ABLATION_PROTOCOL',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False,
        'make_default_adapter': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder'); parser.add_argument('output')
    args = parser.parse_args()
    value = summarize(args.folder)
    with Path(args.output).open('xb') as stream:
        stream.write(canonical(value))
    print(json.dumps({'all_runs': len(value['runs']), 'comparison': value['terminal_comparison']}))
