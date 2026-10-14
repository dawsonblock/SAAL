"""V3 deterministic diagnostics; legacy training semantics remain frozen."""
import contextlib
import copy
import json
import math
import time
from pathlib import Path

from .adaptive_rank_controller import RankController, initial_state
from .epoch_sampler import sample_ids
from .numerical_metrics_v1 import diagnostics
from .epoch_metrics import prediction_equal
from .pretrained_contract import DECISIONS
from .spec import canonical, sha


def selection_key(metrics):
    """New disclosed research selector, separate from qualification."""
    return (-metrics['invalid_outputs'], -metrics['unauthorized_proposals'],
            sum(v > 0 for v in metrics['class_accuracy'].values()),
            metrics['complete_pair_accuracy'] or 0., metrics['worst_family_accuracy'],
            metrics['family_macro_accuracy'], metrics['accuracy'])


def optimizer_update(optimizer, parameters, scaler=None):
    """A post-step hook distinguishes GradScaler overflow from an update.

    Nonfinite gradients interrupt this attempt; no silent skipped-step retry or
    changed schedule is admitted. Allocation retains gradients until afterward.
    """
    import torch
    if scaler is not None:
        scaler.unscale_(optimizer)
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        return False, {'finite': False, 'norm': 0.0, 'gradient_tensors': len(gradients)}
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm):
        return False, {'finite': False, 'norm': 0.0, 'gradient_tensors': len(gradients)}
    called = []
    hook = optimizer.register_step_post_hook(lambda *_: called.append(True))
    try:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
    finally:
        hook.remove()
    return len(called) == 1, {'finite': True, 'norm': float(norm), 'gradient_tensors': len(gradients)}


def train(backend, job, out, *, deadline=None):
    import torch
    out = Path(out)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = deadline if deadline is not None else started+job['max_seconds']
    rows = job['dataset']['records']
    if any(r['split'] not in ('train', 'development') for r in rows):
        raise ValueError('training worker must not receive regression or protected records')
    training = [r for r in rows if r['split'] == 'train']
    development = [r for r in rows if r['split'] == 'development']
    by_id = {r['record_id']: r for r in training}
    stream = sample_ids(training, 'complete-balanced-epochs-v1', job['sampler_seed'],
                        job['steps']*job['batch_size']*job['accumulation'])
    controller = RankController(backend.model, job['adapter'])
    head = [p for n, p in backend.trainable.items() if '.modules_to_save.default.' in n]
    adapters = [p for n, p in backend.trainable.items() if '.modules_to_save.default.' not in n]
    optimizer = torch.optim.AdamW([{'params': head, 'lr': job['head_learning_rate']},
                                  {'params': adapters, 'lr': job['learning_rate']}])
    device = next(backend.model.parameters()).device
    half = job['model_spec']['precision'] == 'float16'
    if half and device.type != 'cuda':
        raise ValueError('FP16 training requires an admitted CUDA runtime')
    scaler = torch.amp.GradScaler('cuda') if half else None
    weights = None
    if job['class_balance']:
        counts = [sum(r['expected_decision'] == d for r in training) for d in DECISIONS]
        weights = torch.tensor([len(training)/(4*n) for n in counts], device=device)
    history, snapshots = [], {}
    offset = tokens = 0

    def check():
        if time.monotonic() > deadline:
            raise TimeoutError('admitted cumulative training/evaluation deadline')

    def emit(row):
        print(json.dumps(row, allow_nan=False), flush=True)
        with (out/'training_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')

    def measure(step):
        check()
        predictions = backend.evaluate(development)
        check()
        m = diagnostics(development, predictions)
        history.append({'step': step, 'metrics': m})
        emit({'monitor': 'checkpoint_metrics', 'step': step, 'development': m})
        return {'step': step, 'predictions': predictions, 'metrics': m}

    def save(role, evidence):
        backend.export_checkpoint(out/(role+'.safetensors'))
        state = copy.deepcopy(controller.state)
        (out/(role+'_rank_pattern.json')).write_bytes(canonical(state))
        snapshots[role] = copy.deepcopy(evidence)

    initial = measure(0)
    save('initial', initial)
    save('best', initial)
    for step in range(1, job['steps']+1):
        check()
        backend.model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = ce = penalty = 0.0
        for _ in range(job['accumulation']):
            ids = stream[offset:offset+job['batch_size']]
            offset += len(ids)
            batch = [by_id[i] for i in ids]
            amp = torch.autocast('cuda', dtype=torch.float16) if half else contextlib.nullcontext()
            with amp:
                loss, work, terms = backend.batch_objective(batch, weights)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss; attempt interrupted without retries')
            objective += float(loss.detach())
            ce += terms['cross_entropy']
            penalty += terms['orthogonal_penalty']
            tokens += work
            accumulated = loss/job['accumulation']
            (scaler.scale(accumulated) if scaler is not None else accumulated).backward()
            check()
        updated, gradients = optimizer_update(optimizer, list(backend.trainable.values()), scaler)
        if not updated:
            # No schedule advancement, altered precision, changed minibatch or retry.
            emit({'monitor': 'optimizer_interrupted', 'attempted_step': step, 'completed_steps': step-1, 'gradients': gradients})
            raise ValueError('optimizer did not complete; rank schedule not advanced')
        memory = {'allocated_bytes': torch.cuda.memory_allocated() if device.type == 'cuda' else 0,
                  'reserved_bytes': torch.cuda.memory_reserved() if device.type == 'cuda' else 0}
        event = controller.complete_update(step, optimizer_updated=True, gradient_evidence=gradients,
                                          elapsed_seconds=time.monotonic()-started, gpu_memory=memory)
        with (out/'rank_history.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False)+'\n')
        # PEFT importance updates consume gradients above. Only now clear them.
        optimizer.zero_grad(set_to_none=True)
        emit({'monitor': 'training_progress', 'completed_steps': step, 'total_steps': job['steps'],
              'objective_loss': objective/job['accumulation'], 'cross_entropy': ce/job['accumulation'],
              'orthogonal_penalty': penalty/job['accumulation'], 'gradient_norm': gradients['norm'],
              'tokens': tokens, 'elapsed_seconds': time.monotonic()-started,
              'active_rank': event['aggregate_active_rank'], 'gpu_allocated_bytes': memory['allocated_bytes']})
        if step % job['checkpoint_interval'] == 0 or step == job['steps']:
            evidence = measure(step)
            if selection_key(evidence['metrics']) > selection_key(snapshots['best']['metrics']):
                save('best', evidence)
    save('terminal', evidence)
    allocation = controller.require_completion(job['steps'])
    if offset != len(stream):
        raise ValueError('sample / optimizer accounting mismatch')
    updated = backend.trainable_hash() != backend.initial_trainable_hash
    if not updated:
        raise ValueError('no real parameter updates')
    frozen = backend.verify_frozen_weights()
    reloads = {}
    for role in ('initial', 'best', 'terminal'):
        state = json.loads((out/(role+'_rank_pattern.json')).read_text())
        backend.restore_checkpoint(out/(role+'.safetensors'), state)
        actual = backend.evaluate(development)
        reloads[role] = prediction_equal(actual, snapshots[role]['predictions'], 'modernbert-seqcls-v1', job['replay_tolerance'])
        if not reloads[role]:
            raise ValueError('same-runtime checkpoint replay failed: '+role)
    sample = {'schema': 'saal-adaptive-sample-stream-v1', 'record_ids': stream,
              'seed': job['sampler_seed'], 'sample_exposures': len(stream), 'job_digest': sha(job)}
    (out/'sampler_manifest.json').write_bytes(canonical(sample))
    (out/'checkpoint_selection.json').write_bytes(canonical({'policy': job['selection_policy'], 'history': history, 'best_step': snapshots['best']['step'], 'data_scope': 'PUBLIC_DEVELOPMENT_ONLY'}))
    return {'schema': 'saal-adaptive-result-v3', 'job_digest': sha(job), 'worker_identity': job['worker_identity'],
            'adapter_method': job['adapter']['method'], 'unadapted_model_hash': backend.unadapted_model_hash,
            'snapshots': snapshots, 'real_parameter_updates': updated, 'frozen_integrity': frozen,
            'same_runtime_replay': reloads, 'allocation': allocation,
            'resource_accounting': {'completed_optimizer_updates': job['steps'], 'microbatches': job['steps']*job['accumulation'],
                'sample_exposures': len(stream), 'tokens': tokens, 'elapsed_seconds': time.monotonic()-started,
                **backend.report_resources()},
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False,
            'trust': 'Worker-reported training evidence, validated/replayed separately; no hostile-worker attestation.'}
