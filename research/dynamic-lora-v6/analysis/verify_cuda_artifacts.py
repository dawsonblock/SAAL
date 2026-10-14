"""Post-training negative checks against a real v2 CUDA artifact bundle.

This does not consume a training or protected-evaluation attempt. Disposable
copies are altered; the authentic run and its manifests remain unchanged.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.adaptive_artifacts_v2 import read_json, file_hash, seal, validate_bundle
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical, sha


def run(bundle, pin, output):
    from safetensors.torch import load_file, save_file
    bundle = Path(bundle)
    envelope = read_json(bundle/'signed_training_job.json')
    job = verify(envelope, pin)
    validate_bundle(bundle, job, pin, require_replay=True)
    original = file_hash(bundle/'artifact_inventory.json')
    cases = ('missing_singular_tensor', 'changed_scaling', 'changed_rank_pattern',
             'changed_rank_history', 'changed_worker_identity', 'fixed_tensor_substitution')
    results = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix='saal-v2-negative-') as temp:
            root = Path(temp)/'bundle'
            shutil.copytree(bundle, root)
            # Recompute the untrusted transport inventory after mutation. The
            # validators must reject the underlying semantics, not only a stale
            # transport checksum. Existing fresh-replay binding is not forged.
            (root/'artifact_inventory.json').unlink()
            (root/'fresh_process_replay.json').unlink()
            if case in ('missing_singular_tensor', 'changed_scaling', 'fixed_tensor_substitution'):
                tensors = load_file(str(root/'terminal.safetensors'))
                if case == 'missing_singular_tensor':
                    tensors.pop(next(n for n in tensors if '.lora_E.' in n))
                elif case == 'changed_scaling':
                    tensors[next(n for n in tensors if '.ranknum.' in n)].fill_(11.)
                else:
                    name = next(n for n in tensors if '.lora_A.' in n)
                    tensors[name+'.weight'] = tensors.pop(name)
                save_file(tensors, str(root/'terminal.safetensors'))
            elif case == 'changed_rank_pattern':
                value = read_json(root/'terminal_rank_pattern.json')
                mask = next(iter(value['masks'].values())); mask[0] = not mask[0]
                (root/'terminal_rank_pattern.json').write_bytes(canonical(value))
            elif case == 'changed_rank_history':
                rows = (root/'rank_history.jsonl').read_text().splitlines()
                value = json.loads(rows[1]); value['aggregate_active_rank'] += 1
                rows[1] = canonical(value).decode()
                (root/'rank_history.jsonl').write_text('\n'.join(rows)+'\n')
            else:
                value = read_json(root/'result.json'); value['worker_identity'] = 'sha256:'+'0'*64
                (root/'result.json').write_bytes(canonical(value))
            seal(root)
            try:
                validate_bundle(root, job, pin)
            except ValueError as exc:
                results.append({'case': case, 'rejected': True, 'reason': str(exc)})
            else:
                raise AssertionError('corruption was admitted: '+case)
    if file_hash(bundle/'artifact_inventory.json') != original:
        raise ValueError('original artifact changed during negative checks')
    report = {'schema': 'saal-adaptive-cuda-artifact-negative-checks-v1', 'job_digest': sha(job),
              'inventory_sha256': original, 'results': results, 'original_bundle_unchanged': True,
              'scope': 'Semantic rejection on copies of a real CUDA bundle after resealing untrusted transport inventory; not hostile-worker attestation.',
              'qualification_status': 'NOT_EVALUATED'}
    with Path(output).open('xb') as stream: stream.write(canonical(report))
    print(json.dumps({'negative_cases_rejected': len(results), 'original_bundle_unchanged': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('bundle'); parser.add_argument('pin'); parser.add_argument('output')
    args = parser.parse_args(); run(args.bundle, args.pin, args.output)
