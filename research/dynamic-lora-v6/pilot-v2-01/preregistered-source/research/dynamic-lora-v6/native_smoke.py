"""Short official-checkpoint CPU engineering checks, excluded from science."""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.adaptive_contract import adapter_spec, new_job
from saal_rsi.adaptive_dataset import training_projection, validate_training_data
from saal_rsi.adaptive_artifacts import validate_bundle
from saal_rsi.adaptive_campaign import stage_worker
from saal_rsi.attempt_service import private_key
from saal_rsi.model_contract import sign
from saal_rsi.pretrained_contract import model_spec, DECISIONS
from saal_rsi.spec import canonical, sha


def small_population(data):
    """Select whole public counterfactual clusters; no final data are read."""
    data = training_projection(data)
    selected = []
    for split in ('train', 'development'):
        groups = collections.defaultdict(list)
        for row in data['records']:
            if row['split'] == split: groups[row['cluster_id']].append(row)
        counts = collections.Counter()
        for cluster in groups.values():
            additions = collections.Counter(r['expected_decision'] for r in cluster)
            if all(counts[d]+additions[d] <= 8 for d in DECISIONS):
                selected.extend(cluster); counts.update(additions)
            if counts == {d: 8 for d in DECISIONS}: break
        if counts != {d: 8 for d in DECISIONS}: raise ValueError('balanced whole-cluster engineering population unavailable')
    return validate_training_data(dict(data, records=selected))


def run(key_path, cache, attempt):
    out = PROJECT/attempt
    out.mkdir(exist_ok=False)
    key = private_key(key_path)
    pin = key.public_key().public_bytes_raw().hex()
    (out/'controller.pub').write_text(pin+'\n')
    population = json.loads((ROOT/'research/evidence-closure-v5/dataset.json').read_text())
    data = small_population(population)
    stage_worker(out/'worker')
    outcomes, hashes, initial_predictions = [], [], []
    for method in ('fixed-r8', 'fixed-r12', 'adalora-12-to-8'):
        name = method.replace('-', '_')
        spec = model_spec('answerdotai/ModernBERT-base', method='none', backend='local', precision='float32', budget='native_'+name)
        budget = {'campaign_id': attempt.replace('-', '_'), 'run_id': name, 'max_runs': 3,
                  'total_lifecycle_seconds': 7500, 'reserved_seconds': 2300,
                  'automatic_training_retries': 0, 'financial_cap': False}
        adapter = adapter_spec(method, total_steps=4, tinit=1, tfinal=1, delta=1)
        job = new_job(spec, data, adapter, seed=60606, budget=budget, steps=4, max_seconds=900,
                      campaign_stage='ENGINEERING', checkpoint_interval=2)
        job_file = out/(name+'-job.json')
        job_file.write_bytes(canonical(sign(job, key)))
        command = [sys.executable, '-I', str(out/'worker/worker.py'), '--job', str(job_file),
                   '--public-key', pin, '--out', str(out/name), '--cache', str(cache)]
        for phase, suffix in [('training', []), ('replay', ['--replay'])]:
            with (out/(name+'-'+phase+'.log')).open('w') as log:
                status = subprocess.run(command+suffix, stdout=log, stderr=subprocess.STDOUT, timeout=1000).returncode
            if status:
                (out/'failure.json').write_bytes(canonical({'method': method, 'phase': phase, 'returncode': status, 'job_digest': sha(job), 'qualified': False}))
                raise RuntimeError('native engineering check failed: '+method+' / '+phase)
        native = validate_bundle(out/name, job, pin, require_replay=True)
        hashes.append(native['unadapted_model_hash'])
        initial_predictions.append(native['snapshots']['initial']['predictions'])
        outcomes.append({'method': method, 'job_digest': sha(job), 'steps': 4,
                         'real_updates': native['real_parameter_updates'], 'all_snapshots_replayed': True,
                         'allocation': native['allocation'], 'source_worker_identity': job['worker_identity']})
    if len(set(hashes)) != 1 or any(p != initial_predictions[0] for p in initial_predictions[1:]):
        raise ValueError('matched base/head/initial predictions differ across adapters')
    (out/'summary.json').write_bytes(canonical({'schema': 'saal-adaptive-native-engineering-v1',
        'runs': outcomes, 'same_unadapted_model_hash': hashes[0], 'same_initial_predictions': True,
        'scope': 'Official ModernBERT, four CPU optimizer updates and 32 development examples. This is not the mandatory three-run 1000-update T4 pilot or evidence of learning/generalization.',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}))
    print(json.dumps({'native_checks': 3, 'checkpoint_restorations': 9, 'scientific_runs': 0}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--key', required=True); p.add_argument('--cache', required=True)
    p.add_argument('--attempt', required=True)
    a = p.parse_args(); run(a.key, a.cache, a.attempt)
