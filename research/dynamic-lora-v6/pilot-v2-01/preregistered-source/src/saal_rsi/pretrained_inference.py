"""Trusted fresh-process v6 replay, with prompts only and no signing credentials.

Process separation under one UID is not a hostile-code sandbox. Deploy this on
the separately managed evaluator with an approved model cache and OS controls.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile


def prediction_envelope(raw):
    from saal_rsi.spec import parse_unique_json
    payload = parse_unique_json(raw)
    if set(payload) != {'predictions'} or type(payload['predictions']) is not list:
        raise ValueError('strict independent prediction envelope required')
    return payload['predictions']


def infer_bundle(bundle, snapshot_name, rows, cache, pin, *, timeout=600):
    from saal_rsi.model_contract import verify
    from saal_rsi.epoch_artifacts import validate_bundle
    from saal_rsi.spec import parse_unique_json, canonical
    root = Path(bundle).resolve()
    job = verify(parse_unique_json((root/'signed_training_job.json').read_bytes()), pin)
    validate_bundle(root, job, pin)
    if snapshot_name not in ('initial', 'best', 'terminal') or not rows or len(rows) > 2000:
        raise ValueError('inference snapshot/population bound')
    request = {'model_spec': job['model_spec'], 'seed': job['seed'],
               'checkpoint': str(root/(snapshot_name+'.safetensors')),
               'cache': str(Path(cache).resolve()),
               'prompts': [r['prompt'] for r in rows]}
    with tempfile.TemporaryDirectory(prefix='saal-pretrained-evaluator-') as temporary:
        work = Path(temporary); data = work/'request.json'; out = work/'predictions.json'
        data.write_bytes(canonical(request))
        env = {'PATH': os.environ.get('PATH', ''), 'HOME': str(work), 'TMPDIR': str(work),
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        # Output is written to a bounded file; stdout is a local diagnostic log,
        # never a source of accepted metrics or qualification assertions.
        with (work/'child.log').open('wb') as log:
            subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()),
                            str(data), str(out)], cwd=work, env=env, check=True,
                           timeout=timeout, stdout=log, stderr=subprocess.STDOUT)
        if out.is_symlink() or not out.is_file() or out.stat().st_size > 2_000_000:
            raise ValueError('inference output budget')
        prediction = prediction_envelope(out.read_bytes())
    from saal_rsi.checkpoint_selection import metrics
    if metrics(rows, prediction, 'modernbert-seqcls-v1')['invalid_outputs']:
        raise ValueError('invalid independent predictions')
    validate_bundle(root, job, pin)
    return prediction


def infer_pair(candidate_bundle, baseline_bundle, protocol, rows, cache):
    return {role: infer_bundle(bundle, protocol[role]['snapshot'], rows, cache,
                              protocol['pins']['controller'])
            for role, bundle in (('baseline', baseline_bundle), ('candidate', candidate_bundle))}


def replay_public(candidate_bundle, baseline_bundle, candidate, baseline, cache, pin):
    from saal_rsi.model_contract import verify
    from saal_rsi.spec import sha, parse_unique_json
    from saal_rsi.epoch_metrics import prediction_equal
    comparisons = {}; probabilities_match = True
    for role, bundle, frozen in (('candidate', candidate_bundle, candidate),
                                 ('baseline', baseline_bundle, baseline)):
        root = Path(bundle)
        job = verify(parse_unique_json((root/'signed_training_job.json').read_bytes()), pin)
        result = parse_unique_json((root/'result.json').read_bytes())
        comparisons[role] = {}
        for split in ('development', 'regression'):
            rows = [r for r in job['dataset']['records'] if r['split'] == split]
            restored = infer_bundle(root, frozen['snapshot'], rows, cache, pin)
            original = result['snapshots'][frozen['snapshot']]['predictions'][split]
            agrees = prediction_equal(restored, original, 'modernbert-seqcls-v1', 1e-4)
            probabilities_match &= agrees
            comparisons[role][split] = {'probabilities_match': agrees,
                'decision_labels_match': all(a['decision'] == b['decision'] for a, b in zip(restored, original)),
                'prediction_digest': sha(restored), 'records': len(rows)}
    return {'schema': 'saal-pretrained-public-replay-v1', 'candidate_digest': sha(candidate),
            'baseline_digest': sha(baseline), 'tolerance': 1e-4,
            'probabilities_match': probabilities_match, 'comparisons': comparisons,
            'scope': 'EVALUATOR_REPRODUCED_PUBLIC_CHECKPOINTS',
            'external_deployment_allowed': False}


def main():
    from saal_rsi.spec import parse_unique_json, canonical
    from saal_rsi.pretrained_registry import validate_snapshot
    from saal_rsi.pretrained_contract import PretrainedModelSpec
    from saal_rsi.learning_backend import verify_dependencies, create_backend
    request = parse_unique_json(Path(sys.argv[1]).read_bytes())
    if set(request) != {'model_spec', 'seed', 'checkpoint', 'cache', 'prompts'}:
        raise ValueError('prompt-only inference request')
    spec = PretrainedModelSpec.parse(request['model_spec']).as_dict()
    if spec['model_kind'] != 'modernbert-seqcls-v1':
        raise ValueError('v6 classifier-only inference')
    verify_dependencies()
    import torch
    torch.set_num_threads(2); torch.manual_seed(request['seed'])
    snapshot = Path(request['cache'])/spec['model_repository'].split('/')[-1]/spec['model_revision']
    validate_snapshot(snapshot, spec['model_repository'], spec['model_revision'])
    backend = create_backend(spec, snapshot, request['seed']).load_pretrained()
    # The qualification protocol fixes CPU inference separately from the
    # admitted training backend, rather than accept whichever GPU is present.
    backend.model.to('cpu')
    backend.restore_checkpoint(request['checkpoint'])
    predictions = backend.evaluate([{'prompt': p} for p in request['prompts']])
    with Path(sys.argv[2]).open('xb') as output:
        output.write(canonical({'predictions': predictions}))


if __name__ == '__main__':
    # A fixed trusted entry point, rather than model-provided Python.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
