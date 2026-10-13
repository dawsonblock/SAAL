"""Fresh CPU diagnostics of existing frozen models; no adaptation or qualification."""
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.model_contract import verify
from saal_rsi.learning_backend import create_backend, verify_dependencies
from saal_rsi.epoch_artifacts_v3 import validate_bundle
from saal_rsi.pretrained_registry import validate_snapshot, file_hash
from saal_rsi.checkpoint_selection import metrics
from saal_rsi.behavior_screen import screen
from saal_rsi.behavior_screen_v2 import error_audit
from saal_rsi.spec import canonical, sha


def run(cache):
    import torch
    torch.set_num_threads(4)
    data = json.loads((PROJECT/'dataset.json').read_text())
    parent = json.loads((ROOT/'research/behavior-closure-v4/dataset.json').read_text())
    old_ids = {r['record_id'] for r in parent['records']}
    pin = (ROOT/'research/behavior-closure-v4/experiment/controller.pub').read_text().strip()
    runs = []
    for name, bundle_name in [('full47', 'run-00-7f4563c3d645'), ('lora211', 'run-02-1ead0e5752ba')]:
        started = time.monotonic()
        bundle = ROOT/'research/behavior-closure-v4/experiment/artifacts'/bundle_name
        job = verify(json.loads((bundle/'signed_training_job.json').read_text()), pin)
        native = validate_bundle(bundle, job, pin)
        snapshot = Path(cache)/job['model_spec']['model_repository'].split('/')[-1]/job['model_spec']['model_revision']
        validate_snapshot(snapshot, job['model_spec']['model_repository'], job['model_spec']['model_revision'])
        backend = create_backend(job['model_spec'], snapshot, job['seed']).load_pretrained()
        backend.restore_checkpoint(bundle/'terminal.safetensors')
        groups = {}
        for split in ('development', 'regression'):
            rows = [r for r in data['records'] if r['split'] == split]
            predictions = backend.evaluate(rows)
            audit = error_audit(data, predictions, split)
            with (PROJECT/f'frozen-{name}-{split}-audit.json').open('xb') as out:
                out.write(canonical(audit))
            strata = {}
            for population in ('old_public', 'new_public', 'combined'):
                selected = [(r,p) for r,p in zip(rows, predictions) if population == 'combined'
                    or (r['record_id'] in old_ids) == (population == 'old_public')]
                m = metrics([r for r,_ in selected], [p for _,p in selected], 'modernbert-seqcls-v1')
                strata[population] = {'records': len(selected), 'metrics': m, 'screen': screen(m)}
            groups[split] = {'strata': strata, 'prediction_digest': sha(predictions), 'predictions': predictions}
            print(json.dumps({'model': name, 'split': split, 'new_public': strata['new_public']['metrics']['accuracy'],
                'combined': strata['combined']['metrics']['accuracy'], 'critical': strata['combined']['metrics']['unauthorized_proposals']}), flush=True)
        runs.append({'name': name, 'job_digest': sha(job), 'native_result_digest': sha(native),
            'checkpoint_sha256': file_hash(bundle/'terminal.safetensors'), 'population_results': groups,
            'runtime_dependencies': verify_dependencies(), 'device': 'cpu', 'elapsed_seconds': time.monotonic()-started,
            'additional_training_updates': 0, 'independent_evaluator': False,
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False})
        del backend
    report = {'schema': 'saal-frozen-public-challenge-v1', 'dataset_digest': sha(data),
        'runs': runs, 'limits': 'Public failure-informed challenge and controller CPU inference. Existing weights are unchanged. No independent generalization, learned repair or final qualification claim.',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    with (PROJECT/'FROZEN_CHALLENGE.json').open('xb') as out: out.write(canonical(report))


if __name__ == '__main__': run(sys.argv[1])
