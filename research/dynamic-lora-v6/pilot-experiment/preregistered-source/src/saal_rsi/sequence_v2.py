"""New sequential protocol: identifiable replay, matched current order, full predictions.

Historical sequence-v1 stays byte-identical. This version has its own signed
source identity and cannot reinterpret an old run or its accuracy-only replay.
"""
import hashlib
import json
import random
import time
from pathlib import Path

from .pretrained_contract import DECISIONS, worker_identity
from .pretrained_continual import validate_sequence, sequence_worker_identity
from .learning_backend import create_backend
from .checkpoint_selection import metrics
from .continual_learning import retention_metrics
from .model_contract import verify
from .spec import sha, canonical
from .replay_sampling import batch_plan


def identity():
    return sha({'base_worker_identity': worker_identity(), 'sources': {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('sequence_v2.py', 'replay_sampling.py')}})


def validate_protocol(protocol, job):
    fields = {'worker_identity', 'schema', 'job_digest', 'groups', 'method', 'replay_ratio',
              'steps_per_stage', 'max_seconds', 'positive_learning_report_digest',
              'qualification_status', 'external_deployment_allowed', 'replay_policy',
              'replay_seed', 'reservoir_capacity', 'dropout_control'}
    if type(protocol) is not dict or set(protocol) != fields or protocol['schema'] != 'saal-pretrained-sequence-v2' or protocol['worker_identity'] != identity():
        raise ValueError('sequence-v2 signed source/protocol required')
    if protocol['replay_policy'] not in ('none', 'reservoir-v1', 'stratified-family-v1') or (protocol['replay_policy'] == 'none') != (protocol['replay_ratio'] == 0.) or protocol['dropout_control'] != 'seeded-per-update-v1':
        raise ValueError('versioned replay and stochastic controls')
    if type(protocol['replay_seed']) is not int or not 0 <= protocol['replay_seed'] < 2**30 or type(protocol['reservoir_capacity']) is not int or not 1 <= protocol['reservoir_capacity'] <= 1024:
        raise ValueError('replay storage and RNG budget')
    if protocol['replay_ratio'] and job['batch_size'] < 4:
        raise ValueError('replay requires four admitted batch slots')
    legacy = {k: v for k, v in protocol.items() if k not in {'replay_policy', 'replay_seed', 'reservoir_capacity', 'dropout_control'}}
    legacy.update(schema='saal-pretrained-sequence-v1', worker_identity=sequence_worker_identity())
    validate_sequence(legacy, job)
    return protocol


def run_sequence_v2(job_envelope, protocol_envelope, pin, snapshot, out, learning_report):
    import torch
    job = verify(job_envelope, pin); protocol = verify(protocol_envelope, pin)
    validate_protocol(protocol, job)
    if job['model_spec']['compute_backend'] != 'local' or job['model_spec']['precision'] != 'float32':
        raise ValueError('sequence-v2 currently admitted for local FP32 only; no cloud-to-CPU fallback')
    if sha(learning_report) != protocol['positive_learning_report_digest'] or learning_report.get('adaptation_gain_terminal', {}).get('mean_gain', 0) <= 0 or learning_report.get('real_parameter_updates') is not True:
        raise ValueError('positive ordinary learning evidence required')
    root = Path(out); root.mkdir(exist_ok=False); started = time.monotonic()
    spec = job['model_spec']; records = job['dataset']['records']
    groups = [{split: [r for r in records if r['split'] == split and r['category'] in categories]
               for split in ('train', 'development')} for categories in protocol['groups']]
    lookup = {r['record_id']: r for r in records}
    shared = create_backend(spec, snapshot, job['seed']).load_pretrained()
    # Admit every prompt/completion before any update; no silent truncation.
    if spec['model_kind'] == 'modernbert-seqcls-v1':
        for i in range(0, len(records), 16): shared.tokenize(records[i:i + 16])
    else:
        from .qwen35_backend import completion_tokens
        for r in records: completion_tokens(shared.tokenizer, r['prompt'], r['expected_output'], spec['max_input_tokens'])

    def predictions(backend, population):
        return backend.evaluate(population, job['max_new_tokens'])

    def accuracy(population, outputs):
        return metrics(population, outputs, spec['model_kind'])['accuracy']

    initial_predictions = [predictions(shared, g['development']) for g in groups]
    initial = [accuracy(g['development'], p) for g, p in zip(groups, initial_predictions)]
    matrix = []; fresh = []; accounting = []; seen = []; isolated = []; evidence = []
    completed = 0

    def adapt(backend, plan, stage, branch):
        nonlocal completed
        before_hash = backend.trainable_hash()
        device = next(backend.model.parameters()).device
        optimizer = torch.optim.AdamW(backend.trainable.values(), lr=job['learning_rate'])
        enabled = device.type == 'cuda' and spec['precision'] == 'float16'
        scaler = torch.amp.GradScaler('cuda', enabled=enabled, init_scale=job['initial_loss_scale'], growth_interval=2000)
        stream = plan['batches']; tokens = 0; used = []; elapsed_start = time.monotonic()
        for step in range(protocol['steps_per_stage']):
            if time.monotonic() - started > protocol['max_seconds']:
                raise TimeoutError('cumulative sequence-v2 deadline')
            torch.manual_seed(job['seed'] + stage * 1009 + step * 9176)
            backend.model.train(); optimizer.zero_grad(set_to_none=True)
            for accumulation in range(job['accumulation']):
                batch = stream[step * job['accumulation'] + accumulation]
                ids = batch['current_ids'] + (batch['replay_ids'] if branch == 'experimental' else [])
                selected = [lookup[rid] for rid in ids]
                weights = None
                if job['class_balance']:
                    # Same registered weighting rule on each actual minibatch.
                    counts = [sum(r['expected_decision'] == d for r in selected) for d in DECISIONS]
                    represented = sum(n > 0 for n in counts)
                    weights = torch.tensor([len(selected) / (represented * n) if n else 0. for n in counts], device=device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled):
                    loss, count = backend.batch_loss(selected, weights)
                if not torch.isfinite(loss): raise ValueError('nonfinite sequence loss')
                scaler.scale(loss / job['accumulation']).backward(); tokens += count
                used.append({'step': step, 'accumulation': accumulation, 'record_ids': ids})
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(backend.trainable.values(), 1.)
            if not torch.isfinite(norm): raise ValueError('nonfinite sequence gradient')
            scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < scale: raise ValueError('skipped sequence update')
            completed += 1
            print(json.dumps({'monitor': 'training_progress', 'stage': stage, 'branch': branch,
                              'completed_steps': completed, 'objective_loss': float(loss.detach()),
                              'elapsed_seconds': time.monotonic() - started}), flush=True)
        backend.verify_frozen_weights()
        path = root / f'{stage}-{branch}.safetensors'; backend.export_checkpoint(path)
        return {'tokens': tokens, 'used_batches': used, 'elapsed_seconds': time.monotonic() - elapsed_start,
                'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'initial_trainable_hash': before_hash, 'terminal_trainable_hash': backend.trainable_hash(),
                'real_parameter_changes': before_hash != backend.trainable_hash()}

    try:
        for stage, group in enumerate(groups):
            if protocol['method'] == 'isolated':
                shared = create_backend(spec, snapshot, job['seed']).load_pretrained()
            plan = batch_plan(group['train'], seen, seed=job['seed'], replay_seed=protocol['replay_seed'], stage=stage,
                              steps=protocol['steps_per_stage'], batch_size=job['batch_size'], accumulation=job['accumulation'],
                              replay_ratio=protocol['replay_ratio'], policy=protocol['replay_policy'], capacity=protocol['reservoir_capacity'])
            (root / f'{stage}-replay-plan.json').write_bytes(canonical(plan))
            experimental = adapt(shared, plan, stage, 'experimental')
            if protocol['method'] == 'isolated': isolated.append(shared)
            output = [predictions(isolated[k] if protocol['method'] == 'isolated' and k <= stage else shared, other['development']) for k, other in enumerate(groups)]
            row = [accuracy(other['development'], p) for other, p in zip(groups, output)]; matrix.append(row)
            control = create_backend(spec, snapshot, job['seed']).load_pretrained()
            control_work = adapt(control, plan, stage, 'fresh_control')
            control_output = predictions(control, group['development']); fresh.append(accuracy(group['development'], control_output))
            evidence.append({'stage': stage, 'task_record_ids': [[r['record_id'] for r in g['development']] for g in groups],
                             'experimental_predictions': output, 'fresh_control_record_ids': [r['record_id'] for r in group['development']],
                             'fresh_control_predictions': control_output})
            accounting.append({'stage': stage, 'steps_each_branch': protocol['steps_per_stage'],
                               'plan_digest': sha(plan), 'experimental': experimental, 'fresh_control': control_work,
                               'trainable_parameters': shared.report_resources()['trainable_parameters']})
            seen.extend(group['train']); del control
    except Exception as exc:
        (root / 'interrupted.json').write_bytes(canonical({'schema': 'saal-sequence-interrupted-v2',
                'protocol_digest': sha(protocol), 'completed_updates': completed, 'error_type': type(exc).__name__,
                'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}))
        raise
    result = retention_metrics(matrix, initial)
    result.update(schema='saal-sequence-result-v2', protocol_digest=sha(protocol), worker_identity=identity(),
                  initial_predictions=initial_predictions, stage_predictions=evidence,
                  fresh_control_accuracy=fresh, matched_adaptation_gain=[matrix[k][k] - fresh[k] for k in range(len(groups))],
                  resource_accounting=accounting, elapsed_seconds=time.monotonic() - started,
                  completed_updates=completed, qualification_status='NOT_EVALUATED', external_deployment_allowed=False,
                  limits='Public, correlated recipes. Current examples and per-update RNG initialization are matched. Replay changes exposure, batch size and actual work, all recorded; identical steps do not establish equal compute or independent transfer.')
    (root / 'result.json').write_bytes(canonical(result))
    (root / 'signed_job.json').write_bytes(canonical(job_envelope))
    (root / 'signed_protocol.json').write_bytes(canonical(protocol_envelope))
    files = {p.name: {'size': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.iterdir() if p.is_file()}
    (root / 'inventory.json').write_bytes(canonical({'schema': 'saal-sequence-inventory-v2', 'files': files}))
    return result
