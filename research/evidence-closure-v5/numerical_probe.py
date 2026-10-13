"""Diagnose two preserved replay failures without relaxing their tolerance."""
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.model_contract import verify
from saal_rsi.epoch_artifacts_v3 import validate_bundle
from saal_rsi.learning_backend import create_backend, verify_dependencies
from saal_rsi.pretrained_registry import validate_snapshot
from saal_rsi.spec import canonical, sha

CASES = [
    ('research/reliable-learning-v3/experiment/artifacts/run-00-390a3134ca07',
     'research/reliable-learning-v3/experiment/controller-replay/run-00-390a3134ca07.json',
     'research/reliable-learning-v3/experiment/controller.pub', 'best'),
    ('research/behavior-closure-v4/experiment/artifacts/run-01-b74e6b14c67a',
     'research/behavior-closure-v4/cpu-replay/continuation03-run-01-b74e6b14c67a.json',
     'research/behavior-closure-v4/experiment/controller.pub', 'terminal')]


def run(cache, destination):
    import torch
    results = []
    started = time.monotonic()
    for bundle_path, replay_path, pin_path, checkpoint in CASES:
        bundle = ROOT/bundle_path
        pin = (ROOT/pin_path).read_text().strip()
        job = verify(json.loads((bundle/'signed_training_job.json').read_text()), pin)
        native = validate_bundle(bundle, job, pin)
        original = native['snapshots'][checkpoint]['predictions']['development']
        previous = next(m['predictions'] for m in json.loads((ROOT/replay_path).read_text())['measurements']
            if m['checkpoint'] == checkpoint and m['split'] == 'development')
        target = max(range(len(original)), key=lambda i: max(abs(a-b) for a,b in
            zip(original[i]['probabilities'], previous[i]['probabilities'])))
        rows = [r for r in job['dataset']['records'] if r['split'] == 'development']
        offset = target//16*16
        snapshot = Path(cache)/job['model_spec']['model_repository'].split('/')[-1]/job['model_spec']['model_revision']
        validate_snapshot(snapshot, job['model_spec']['model_repository'], job['model_spec']['model_revision'])
        torch.set_num_threads(4)
        backend = create_backend(job['model_spec'], snapshot, job['seed']).load_pretrained()
        backend.restore_checkpoint(bundle/(checkpoint+'.safetensors'))
        probes = []
        for threads, batch in [(4, 'registered_batch'), (4, 'repeat'), (1, 'single_thread'), (4, 'single_record')]:
            torch.set_num_threads(threads)
            output = backend.evaluate([rows[target]])[0] if batch == 'single_record' else backend.evaluate(rows[offset:offset+16])[target-offset]
            probes.append({'profile': batch, 'threads': threads, 'output': output,
                'max_probability_difference_from_native_gpu': max(abs(a-b) for a,b in zip(output['probabilities'], original[target]['probabilities'])),
                'max_probability_difference_from_original_cpu': max(abs(a-b) for a,b in zip(output['probabilities'], previous[target]['probabilities'])),
                'label_matches_native': output['decision'] == original[target]['decision']})
        results.append({'job_digest': sha(job), 'checkpoint': checkpoint, 'record_id': rows[target]['record_id'],
            'native_gpu_output': original[target], 'original_cpu_output': previous[target], 'probes': probes})
        del backend
    body = {'schema': 'saal-preserved-replay-failure-probe-v1', 'cases': results,
        'runtime_dependencies': verify_dependencies(), 'device': 'cpu', 'elapsed_seconds': time.monotonic()-started,
        'tolerance_unchanged': 1e-4, 'historical_failures_reclassified': False,
        'limits': 'Selected worst-case exposed diagnostic probes; no statistical claim, no new model training or independent qualification.',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    with Path(destination).open('xb') as out: out.write(canonical(body))
    print(json.dumps({'cases': len(results), 'elapsed_seconds': body['elapsed_seconds']}))


if __name__ == '__main__': run(sys.argv[1], sys.argv[2])
