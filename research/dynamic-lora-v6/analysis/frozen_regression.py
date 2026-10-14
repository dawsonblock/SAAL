"""Bounded CPU regression of completed, replayed public research candidates.

The training job, development selection and protected evaluations are unchanged.
Each task is frozen before inference, attempted once and preserved on failure.
"""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT/'research/dynamic-lora-v6'
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.adaptive_artifacts import ROLES, file_hash, read_json
from saal_rsi.adaptive_dispatch import validate_bundle
from saal_rsi.adaptive_evaluator import evaluate_frozen
from saal_rsi.epoch_metrics import prediction_equal
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical, sha

MAX_SECONDS = 600


def write_new(path, value):
    with Path(path).open('xb') as stream: stream.write(canonical(value))


def frozen_run(campaign, descriptor, row):
    """Require actual completed/replayed evidence; never trust a passing flag."""
    campaign = Path(campaign)
    pin = (campaign/'controller.pub').read_text().strip()
    job = verify(read_json(campaign/(descriptor['name']+'-job.json')), pin)
    digest = sha(job)
    outcome = row.get('outcome')
    if (row['state'] != 'COMPLETED' or not outcome or row['digest'] != digest
            or descriptor['job_digest'] != digest or outcome['job_digest'] != digest):
        raise ValueError('completed frozen candidate required')
    name = outcome['run']
    if type(name) is not str or not re.fullmatch(r'run-[0-9]{2}-'+digest[7:19], name):
        raise ValueError('candidate artifact identity')
    bundle = campaign/'artifacts'/name
    result = validate_bundle(bundle, job, pin, require_replay=True)
    cpu = read_json(campaign/'colab'/(name+'-controller-replay.json'))
    inventory = file_hash(bundle/'artifact_inventory.json')
    if (outcome['result_digest'] != sha(result) or outcome['controller_replay_digest'] != sha(cpu)
            or cpu['job_digest'] != digest or cpu['inventory_sha256'] != inventory or cpu['verified'] is not True
            or any(not prediction_equal(cpu['predictions'][role], result['snapshots'][role]['predictions'],
                'modernbert-seqcls-v1', job['replay_tolerance']) for role in ROLES)):
        raise ValueError('candidate replay lineage mismatch')
    return job, bundle, result, {'job_digest': digest, 'inventory_sha256': inventory,
                               'result_digest': sha(result), 'cpu_replay_digest': sha(cpu)}


def population():
    source = ROOT/'research/evidence-closure-v5/dataset.json'
    rows = [r for r in read_json(source)['records'] if r['split'] == 'regression']
    if not 1 <= len(rows) <= 2000 or len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('bounded complete public regression population')
    return rows, {'source': source.relative_to(ROOT).as_posix(), 'source_sha256': file_hash(source),
                  'records': len(rows), 'population_digest': sha(rows)}


def worker(task_path):
    started, cpu_started = time.monotonic(), time.process_time()
    task_path = Path(task_path); task = read_json(task_path)
    plan_path = task_path.parent.parent/'plan.json'; plan = read_json(plan_path)
    if task['plan_digest'] != sha(plan) or plan['analysis_source_sha256'] != file_hash(__file__):
        raise ValueError('analysis implementation / task substitution')
    rows, identity = population()
    if plan['population'] != identity:
        raise ValueError('frozen regression population changed')
    campaign = ROOT/plan['campaign_path']
    pin = (campaign/'controller.pub').read_text().strip()
    protocol = verify(read_json(campaign/'signed_protocol.json'), pin)
    if sha(protocol) != plan['protocol_digest']:
        raise ValueError('frozen campaign changed before regression')
    for name, expected in protocol['source_files'].items():
        if name.startswith('src/saal_rsi/') and (Path(name).is_absolute() or '..' in Path(name).parts or file_hash(ROOT/name) != expected):
            raise ValueError('admitted model / replay library changed')
    job, bundle, native, binding = frozen_run(campaign, task['descriptor'], task['completed_attempt'])
    if binding != task['candidate_binding'] or sha(job) not in protocol['job_digests']:
        raise ValueError('frozen candidate changed before regression')
    from saal_rsi.adaptive_backend import verify_peft_implementation
    from saal_rsi.learning_backend import verify_dependencies
    from saal_rsi.pretrained_registry import validate_snapshot
    if job['schema'] == 'saal-adaptive-job-v2':
        from saal_rsi.adaptive_replay_v2 import PortableReplayBackendV2 as Backend
    else:
        from saal_rsi.adaptive_replay import PortableReplayBackend as Backend
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(job['seed']); torch.use_deterministic_algorithms(True)
    versions = verify_dependencies(); verify_peft_implementation()
    spec = job['model_spec']
    snapshot = Path(task['cache'])/spec['model_repository'].split('/')[-1]/spec['model_revision']
    validate_snapshot(snapshot, spec['model_repository'], spec['model_revision'])
    backend = Backend(job, snapshot, bundle/'initial.safetensors', read_json(bundle/'initial_rank_pattern.json')).load_pretrained()
    if (str(next(backend.model.parameters()).device) != 'cpu'
            or backend.unadapted_model_hash != native['unadapted_model_hash']
            or backend.initial_frozen_hash != native['frozen_integrity']['initial']):
        raise ValueError('exact CPU model, head and frozen backbone identity required')
    if time.monotonic()-started >= MAX_SECONDS:
        raise TimeoutError('analysis admission consumed CPU deadline')
    value = evaluate_frozen(backend, bundle, job, rows)
    frozen = backend.verify_frozen_weights()
    if frozen['unchanged'] is not True or file_hash(bundle/'artifact_inventory.json') != binding['inventory_sha256']:
        raise ValueError('frozen evidence changed during analysis')
    work_tokens = 0
    for offset in range(0, len(rows), 16):
        work_tokens += int(backend.tokenize(rows[offset:offset+16])['attention_mask'].sum())
    value.update(analysis_plan_digest=sha(plan), candidate_binding=binding,
        method=job['adapter']['method'], seed=job['seed'], population_identity=identity,
        runtime={'versions': versions, 'device': 'cpu', 'threads': 2, 'platform': sys.platform,
                 'python': sys.version.split()[0],
                 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
        measured_work={'checkpoint_evaluations': 3, 'case_predictions': 3*len(rows),
                       'inference_input_tokens': 3*work_tokens,
                       'wall_seconds': time.monotonic()-started, 'process_cpu_seconds': time.process_time()-cpu_started,
                       'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)},
        frozen_integrity=frozen, independent_administration=False,
        retention='NOT_MEASURED_REQUIRES_SEQUENTIAL_PROTOCOL', independent_generalization='NOT_ESTABLISHED')
    if time.monotonic()-started >= MAX_SECONDS:
        raise TimeoutError('complete regression exceeded CPU deadline')
    write_new(task_path.parent/'result.json', value)
    print(json.dumps({'job_digest': sha(job), 'case_predictions': 3*len(rows), 'completed': True}), flush=True)


def run(campaign, output, cache):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if not campaign.is_relative_to(PROJECT) or not output.is_relative_to(PROJECT):
        raise ValueError('task-owned public campaign and output required')
    pin = (campaign/'controller.pub').read_text().strip()
    protocol = verify(read_json(campaign/'signed_protocol.json'), pin)
    rows, identity = population()
    plan = {'schema': 'saal-adaptive-public-regression-plan-v1', 'campaign_path': campaign.relative_to(ROOT).as_posix(),
        'protocol_digest': sha(protocol), 'job_digests': protocol['job_digests'], 'population': identity,
        'analysis_source_sha256': file_hash(__file__), 'worker_wall_seconds_per_task_limit': MAX_SECONDS,
        'max_tasks': len(protocol['job_digests']), 'max_total_task_seconds': len(protocol['job_digests'])*MAX_SECONDS,
        'automatic_retries': 0, 'selection_permitted': False, 'final_holdout_access': False,
        'scope': 'Exposed public regression, after per-candidate freeze and replay. No training, checkpoint selection, retention experiment or protected qualification.',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    if not output.exists():
        output.mkdir(); write_new(output/'plan.json', plan)
    elif read_json(output/'plan.json') != plan:
        raise ValueError('existing analysis plan cannot be reset or reinterpreted')
    export = read_json(campaign/'colab/registry-export.json')
    if export['protocol_digest'] != sha(protocol) or len(export['attempts']) != len(protocol['runs']):
        raise ValueError('complete admission accounting required')
    completed = 0
    for descriptor, row in zip(protocol['runs'], export['attempts']):
        if row['state'] != 'COMPLETED': continue
        job, bundle, native, binding = frozen_run(campaign, descriptor, row)
        destination = output/(sha(job)[7:19])
        if destination.exists():
            state = read_json(destination/'outcome.json')
            if state['state'] != 'COMPLETED' or state['candidate_binding'] != binding or state['result_sha256'] != file_hash(destination/'result.json'):
                raise ValueError('prior analysis failed / changed; no automatic retry')
            completed += 1
            continue
        destination.mkdir(exist_ok=False)
        task = {'plan_digest': sha(plan), 'descriptor': descriptor, 'completed_attempt': row,
                'candidate_binding': binding, 'cache': str(Path(cache).resolve())}
        write_new(destination/'task.json', task)
        failure = None; returncode = None
        try:
            with (destination/'worker.log').open('xb') as log:
                result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--worker', str(destination/'task.json')],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=MAX_SECONDS)
            returncode = result.returncode
            if returncode: failure = 'bounded CPU regression worker failed'
            elif not (destination/'result.json').is_file(): failure = 'missing regression evidence'
        except subprocess.TimeoutExpired:
            failure = 'bounded CPU regression timed out; no automatic retry'
        except Exception as exc:
            failure = type(exc).__name__+': '+str(exc)
        outcome = {'state': 'FAILED' if failure else 'COMPLETED', 'candidate_binding': binding,
                   'returncode': returncode, 'error': failure,
                   'result_sha256': file_hash(destination/'result.json') if (destination/'result.json').exists() else None,
                   'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
        write_new(destination/'outcome.json', outcome)
        if failure: raise RuntimeError(failure)
        completed += 1
        print(json.dumps({'regression_completed': completed, 'method': job['adapter']['method'], 'seed': job['seed']}), flush=True)
    print(json.dumps({'completed_public_regressions': completed, 'admitted_max': plan['max_tasks'], 'training_changed': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('campaign', nargs='?'); parser.add_argument('output', nargs='?'); parser.add_argument('cache', nargs='?'); parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif None in (args.campaign, args.output, args.cache): parser.error('campaign, output and cache are required')
    else: run(args.campaign, args.output, args.cache)
