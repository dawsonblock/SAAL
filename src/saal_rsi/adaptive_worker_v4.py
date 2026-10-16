"""V4 deterministic numerical contract; only freshly signed v4 jobs execute."""
"""V4 public-data-only worker bound to numerical metrics v2; no private authority credentials."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

from .adaptive_contract_v4 import validate_job, RANKNUM_POLICY
from .numerical_metrics_v2 import POLICY, POLICY_DIGEST
from .adaptive_backend_v2_v4 import CudaSafeAdaLoRABackend as AdaptiveModernBERTBackend
from .adaptive_backend import verify_peft_implementation
from .adaptive_artifacts_v4 import ROLES, seal, validate_bundle, read_json, file_hash
from .adaptive_training_v4 import train
from .epoch_metrics import prediction_equal
from .learning_backend import verify_dependencies
from .model_contract import verify
from .pretrained_contract import job_bytes
from .pretrained_registry import fetch_snapshot, validate_snapshot
from .spec import canonical, parse_unique_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--public-key', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    envelope = parse_unique_json(job_bytes(args.job))
    job = validate_job(verify(envelope, args.public_key))
    started = time.monotonic()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    versions = verify_dependencies()
    implementation = verify_peft_implementation()
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(job['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    free = total = 0
    spec = job['model_spec']
    if spec['compute_backend'] == 'colab':
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():
            raise ValueError('exact admitted T4 is unavailable')
        free, total = torch.cuda.mem_get_info()
        if free < 6*1024**3 or os.environ.get('SAAL_HASH_LOCK_ENFORCED') != '1':
            raise ValueError('GPU memory / hash lock admission failed')
        torch.cuda.reset_peak_memory_stats()
    snapshot = fetch_snapshot(args.cache, spec['model_repository'], spec['model_revision'],
                              lambda v: print(json.dumps(v), flush=True))
    manifest = validate_snapshot(snapshot, spec['model_repository'], spec['model_revision'])
    backend = AdaptiveModernBERTBackend(job, snapshot).load_pretrained()
    for offset in range(0, len(job['dataset']['records']), 16):
        backend.tokenize(job['dataset']['records'][offset:offset+16])
    out = Path(args.out)
    if args.replay:
        result = validate_bundle(out, job, args.public_key)
        if backend.unadapted_model_hash != result['unadapted_model_hash'] or backend.initial_frozen_hash != result['frozen_integrity']['initial']:
            raise ValueError('fresh baseline / classifier / frozen backbone identity mismatch')
        rows = [r for r in job['dataset']['records'] if r['split'] == 'development']
        checkpoints = {}
        for role in ROLES:
            backend.restore_checkpoint(out/(role+'.safetensors'), read_json(out/(role+'_rank_pattern.json')))
            fresh = backend.evaluate(rows)
            if not prediction_equal(fresh, result['snapshots'][role]['predictions'], 'modernbert-seqcls-v1', job['replay_tolerance']):
                raise ValueError('fresh-process checkpoint restoration failed: '+role)
            checkpoints[role] = fresh
        receipt = {'schema': 'saal-adaptive-fresh-replay-v4', 'job_digest': sha(job),
                   'inventory_sha256': file_hash(out/'artifact_inventory.json'),
                   'numerical_metric_policy_digest': POLICY_DIGEST,
                   'checkpoints': checkpoints, 'tolerance': job['replay_tolerance'], 'verified': True,
                   'runtime': {'versions': versions, 'python': platform.python_version(), 'device': str(next(backend.model.parameters()).device)}}
        with (out/'fresh_process_replay.json').open('xb') as f:
            f.write(canonical(receipt))
        validate_bundle(out, job, args.public_key, require_replay=True)
        print(json.dumps({'monitor': 'fresh_process_replay', 'verified': True, 'checkpoints': list(ROLES), 'job_digest': sha(job)}), flush=True)
        return
    if time.monotonic()-started > job['max_seconds']-30:
        raise TimeoutError('preflight consumed training budget')
    from .epoch_sampler import sample_ids
    import contextlib
    training = [r for r in job['dataset']['records'] if r['split'] == 'train']
    indexed = {r['record_id']: r for r in training}
    probe_ids = sample_ids(training, 'complete-balanced-epochs-v1', job['sampler_seed'], job['batch_size'])
    probe_started = time.monotonic()
    backend.model.train()
    amp = torch.autocast('cuda', dtype=torch.float16) if spec['precision'] == 'float16' else contextlib.nullcontext()
    with amp:
        probe_loss, probe_tokens, _ = backend.batch_objective([indexed[i] for i in probe_ids])
    if not torch.isfinite(probe_loss):
        raise ValueError('forward preflight is nonfinite')
    probe_loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in backend.trainable.values()):
        raise ValueError('backward preflight is missing / nonfinite')
    backend.model.zero_grad(set_to_none=True)
    if backend.trainable_hash() != backend.initial_trainable_hash:
        raise ValueError('admission probe altered weights')
    torch.manual_seed(job['dropout_seed'])
    preflight = {'forward_backward_completed': True, 'optimizer_updates': 0,
                 'sample_exposures_without_updates': len(probe_ids), 'tokens': probe_tokens,
                 'weights_unchanged': True, 'elapsed_seconds': time.monotonic()-probe_started,
                 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                 'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
    print(json.dumps({'monitor': 'gpu_admission_probe', **preflight}), flush=True)
    result = train(backend, job, out, deadline=started+job['max_seconds'])
    result.update(schema='saal-adaptive-result-v4', ranknum_policy=RANKNUM_POLICY, ranknum_normalized=backend.ranknum_normalized,
                  numerical_metric_policy=POLICY, numerical_metric_policy_digest=POLICY_DIGEST)
    result.update(runtime={'versions': versions, 'peft_implementation': implementation,
                           'python': platform.python_version(), 'platform': platform.platform(),
                           'device': str(next(backend.model.parameters()).device),
                           'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                           'free_gpu_bytes_preflight': free, 'total_gpu_bytes': total,
                           'attention': spec['attention_implementation'],
                           'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                           'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                           'cuda_version': torch.version.cuda,
                           'cudnn_version': torch.backends.cudnn.version(),
                           'dependency_hash_lock_enforced': os.environ.get('SAAL_HASH_LOCK_ENFORCED') == '1'},
                  preflight=preflight, total_worker_seconds=time.monotonic()-started)
    values = {'signed_training_job.json': envelope, 'model_spec.json': spec,
              'dataset_manifest.json': job['dataset'], 'snapshot_manifest.json': manifest,
              'adapter_config.json': job['adapter'],
              'rank_schedule.json': {'schema': 'saal-adaptive-rank-schedule-v1', 'schedule': job['adapter']['schedule']},
              'resource_accounting.json': result['resource_accounting'], 'result.json': result}
    for name, value in values.items():
        (out/name).write_bytes(canonical(value))
    seal(out)
    validate_bundle(out, job, args.public_key)
    print(json.dumps({'monitor': 'training_complete', 'job_digest': sha(job),
                      'selected_step': result['snapshots']['best']['step'], 'qualification_status': 'NOT_EVALUATED'}), flush=True)
