"""Bounded, non-resumable static-adapter diagnostic loop for protocol v5."""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import torch

from .adaptive_contract_v5 import validate_pilot_config
from .adaptive_dataset import validate_training_data
from .adaptive_rank_controller_v4 import initial_state
from .adaptive_training_v4 import eligible_trained_checkpoint
from .dynamic_lora_v5.svd_reference import (
    capture_parameter_snapshot,
    effective_update_change_frobenius_norm,
    effective_update_frobenius_norm,
    gradient_contribution_diagnostics,
    trainable_parameter_diagnostics,
)
from .epoch_metrics import prediction_equal
from .epoch_sampler import sample_ids
from .numerical_metrics_v2 import diagnostics
from .pretrained_contract import DECISIONS
from .spec import canonical, sha

CLIP_THRESHOLD = 1.0
REPLAY_TOLERANCE = 1e-4


def _checkpoint_selection_key(metrics):
    """Order feasible candidates by pair behavior before aggregate accuracy."""
    pair_accuracy = metrics.get('complete_pair_accuracy')
    worst_family = metrics.get('worst_family_accuracy')
    family_macro = metrics.get('family_macro_accuracy')
    accuracy = metrics.get('accuracy')
    values = (pair_accuracy, worst_family, family_macro, accuracy)
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in values):
        raise ValueError('finite behavioral checkpoint metrics required')
    return tuple(float(value) for value in values)


def _write_jsonl(path, value, *, maximum_bytes=64_000_000):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        if Path(path).stat().st_size > maximum_bytes:
            raise ValueError('bounded pilot log exceeded')


def _optimizer_completed(optimizer, scaler):
    """Detect GradScaler skips using the optimizer post-step hook."""
    completed = []
    handle = optimizer.register_step_post_hook(lambda *_: completed.append(True))
    try:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
    finally:
        handle.remove()
    return len(completed) == 1


def _nonfinite_gradient_summary(named_parameters):
    """Return bounded per-tensor evidence without copying raw gradient data."""
    import torch
    failures = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        finite_mask = torch.isfinite(gradient)
        if not bool(finite_mask.all()):
            finite_values = gradient[finite_mask]
            failures.append({
                'name': name,
                'nonfinite_elements': int((~finite_mask).sum().item()),
                'total_elements': gradient.numel(),
                'finite_max_abs': (float(finite_values.abs().max().item())
                                   if finite_values.numel() else None),
            })
    return failures


def _evaluate(backend, rows):
    outputs = backend.evaluate(rows)
    return diagnostics(rows, outputs, kind='modernbert-seqcls-v1'), outputs


def _train_static_pilot(config, dataset, backend, output_dir, *,
                        execution_permit, deadline_seconds=None):
    """Internal training body; callers must use the signed v5 controller path."""
    from .adaptive_static_pilot_v5 import _ExecutionPermit
    if type(execution_permit) is not _ExecutionPermit:
        raise PermissionError('v5 controller-issued execution permit required')
    validate_pilot_config(config)
    validate_training_data(dataset)
    if (sha(dataset) != config['identity']['dataset_digest']
            or sha(config) != execution_permit.config_digest
            or sha(dataset) != execution_permit.dataset_digest):
        raise ValueError('dataset digest differs from scientific configuration')
    out = Path(output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError('pilot output must be a new directory')
    out.mkdir(parents=True, mode=0o700)
    identity = config['identity']
    training = [row for row in dataset['records'] if row['split'] == 'train']
    development = [row for row in dataset['records'] if row['split'] == 'development']
    by_id = {row['record_id']: row for row in training}
    count = config['steps'] * identity['batch_size'] * identity['accumulation']
    stream = sample_ids(training, 'complete-balanced-epochs-v1',
                        identity['sampler_seed'], count)
    if sha(stream) != identity['sample_order_digest']:
        raise ValueError('deterministic sample stream differs from admission identity')
    if (backend.spec.as_dict() != identity['model_spec']
            or backend.seed != identity['seed']):
        raise ValueError('backend model or seed differs from scientific configuration')
    device = next(backend.model.parameters()).device
    precision = identity['precision']
    if precision != 'float32' and device.type != 'cuda':
        raise ValueError('mixed-precision pilot requires the admitted CUDA runtime')
    if precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('admitted runtime does not support CUDA bfloat16')
    autocast_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler('cuda') if precision == 'float16' else None
    # Keep stochastic training draws (including dropout) on the contract's
    # separate dropout stream; the model/head seed was already applied during
    # pretrained model construction.
    torch.manual_seed(identity['dropout_seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(identity['dropout_seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    head = [parameter for name, parameter in backend.trainable.items()
            if '.modules_to_save.default.' in name]
    adapter = [parameter for name, parameter in backend.trainable.items()
               if '.modules_to_save.default.' not in name]
    parameter_groups = []
    if head:
        parameter_groups.append({'params': head, 'lr': identity['head_learning_rate'], 'name': 'head'})
    if adapter:
        parameter_groups.append({'params': adapter, 'lr': identity['adapter_learning_rate'], 'name': 'adapter'})
    if not parameter_groups:
        raise ValueError('no trainable pilot parameters')
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=identity['weight_decay'])
    weights = None
    if identity['class_balance']:
        counts = [sum(row['expected_decision'] == label for row in training)
                  for label in DECISIONS]
        if any(value == 0 for value in counts):
            raise ValueError('class-balanced objective requires all four labels')
        weights = torch.tensor([len(training) / (4 * n) for n in counts], device=device)

    start = time.monotonic()
    if type(deadline_seconds) not in (int, float) or deadline_seconds <= 0:
        raise ValueError('remaining signed wall-clock budget required')
    deadline = min(start + deadline_seconds, execution_permit.deadline)

    def check_output_budget():
        total = 0
        for path in out.rglob('*'):
            if path.is_symlink():
                raise ValueError('symlink in pilot output')
            if path.is_file():
                total += path.stat().st_size
                if total > execution_permit.output_limit:
                    raise ValueError('pilot artifact output budget exceeded')

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError('static pilot deadline exceeded; no automatic retry')

    def evaluate_step(step):
        check_deadline()
        backend.model.eval()
        metrics, predictions = _evaluate(backend, development)
        _write_jsonl(out / 'development_metrics.jsonl', {
            'schema': 'saal-static-pilot-development-v5', 'step': step,
            'metrics': metrics,
        }, maximum_bytes=execution_permit.output_limit)
        check_output_budget()
        return {'step': step, 'metrics': metrics, 'predictions': predictions}

    def save_checkpoint(name, state):
        check_deadline()
        backend.export_checkpoint(out / f'{name}.safetensors')
        check_deadline()
        if (out / f'{name}.safetensors').stat().st_size > execution_permit.output_limit:
            raise ValueError('checkpoint exceeds signed output budget')
        rank_state = initial_state(backend.adapter)
        rank_state['step'] = state['step']
        (out / f'{name}_rank_state.json').write_bytes(canonical(rank_state))
        return {
            'step': state['step'], 'metrics': state['metrics'],
            'predictions_digest': sha(state['predictions']),
            'checkpoint_sha256': __import__('hashlib').sha256(
                (out / f'{name}.safetensors').read_bytes()).hexdigest(),
        }

    initial = evaluate_step(0)
    initial_record = save_checkpoint('initial', initial)
    (out / 'initial.json').write_bytes(canonical(initial_record))
    best = None
    best_record = None
    stream_offset = 0
    total_tokens = 0
    previous_parameters = capture_parameter_snapshot(backend.trainable.items())
    previous_updates = backend.effective_update_records()
    previous_effective = {
        target: effective_update_frobenius_norm(
            value['A'], value['E'], value['B'], scaling=value['scaling'],
            ranknum=value['ranknum'])
        for target, value in previous_updates.items()
    }
    progress_path = out / 'training_diagnostics.jsonl'

    for step in range(1, config['steps'] + 1):
        check_deadline()
        backend.model.train()
        optimizer.zero_grad(set_to_none=True)
        objective_terms = []
        diagnostic_terms = None
        for microstep in range(identity['accumulation']):
            ids = stream[stream_offset:stream_offset + identity['batch_size']]
            stream_offset += len(ids)
            batch = [by_id[value] for value in ids]
            autocast = (torch.autocast('cuda', dtype=autocast_dtype)
                        if autocast_dtype is not None else contextlib.nullcontext())
            with autocast:
                loss, token_count, terms = backend.batch_objective(batch, weights)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss; attempt terminates without retry')
            total_tokens += token_count
            objective_terms.append(terms)
            if (diagnostic_terms is None
                    and (step <= 10 or step % config['evaluation_interval'] == 0)):
                diagnostic_terms = gradient_contribution_diagnostics(
                    backend._cross_entropy_for_diagnostics,
                    backend.orthogonal_penalty(),
                    backend.trainable.values(),
                    config['orthogonal_coefficient'],
                )
            scaled_loss = loss / identity['accumulation']
            (scaler.scale(scaled_loss) if scaler is not None else scaled_loss).backward()
            check_deadline()

        if scaler is not None:
            scaler.unscale_(optimizer)
        if any(value is None for value in
               (parameter.grad for parameter in backend.trainable.values())):
            _write_jsonl(progress_path, {
                'schema': 'saal-static-pilot-failed-step-v5',
                'attempted_optimizer_step': step,
                'completed_optimizer_steps': step - 1,
                'precision': precision,
                'reason': 'missing trainable gradient',
                'parameters': sorted(name for name, parameter in backend.trainable.items()
                                     if parameter.grad is None),
            }, maximum_bytes=execution_permit.output_limit)
            raise ValueError('missing trainable gradient')
        nonfinite = _nonfinite_gradient_summary(backend.trainable.items())
        if nonfinite:
            _write_jsonl(progress_path, {
                'schema': 'saal-static-pilot-failed-step-v5',
                'attempted_optimizer_step': step,
                'completed_optimizer_steps': step - 1,
                'precision': precision,
                'grad_scaler_scale': (float(scaler.get_scale())
                                      if scaler is not None else None),
                'reason': 'nonfinite trainable gradient',
                'parameters': nonfinite,
            }, maximum_bytes=execution_permit.output_limit)
            raise ValueError('nonfinite gradient; attempt terminates without retry')
        preclip = trainable_parameter_diagnostics(
            backend.trainable.items(), previous=previous_parameters,
            optimizer=optimizer, clip_threshold=identity['gradient_clip'])
        unclipped_norm = torch.nn.utils.clip_grad_norm_(
            list(backend.trainable.values()), identity['gradient_clip'])
        if not torch.isfinite(unclipped_norm):
            raise ValueError('nonfinite gradient norm')
        if not _optimizer_completed(optimizer, scaler):
            _write_jsonl(progress_path, {
                'schema': 'saal-static-pilot-interrupted-v5',
                'attempted_step': step, 'completed_steps': step - 1,
                'reason': 'optimizer step skipped or not observed',
            }, maximum_bytes=execution_permit.output_limit)
            raise RuntimeError('optimizer update did not complete; no retry')

        after_parameters = trainable_parameter_diagnostics(
            backend.trainable.items(), previous=previous_parameters,
            optimizer=optimizer, clip_threshold=identity['gradient_clip'])
        update_norm = math.sqrt(sum(
            value['parameter_change_norm'] ** 2
            for value in after_parameters['parameters'].values()
            if value['parameter_change_norm'] is not None
        ))
        current_updates = backend.effective_update_records()
        effective = {}
        for target, value in current_updates.items():
            norm = effective_update_frobenius_norm(
                value['A'], value['E'], value['B'], scaling=value['scaling'],
                ranknum=value['ranknum'])
            change = effective_update_change_frobenius_norm(
                previous_updates[target], value)
            effective[target] = {
                'frobenius_norm': norm,
                'change_norm': change,
                'change_from_previous_norm': norm - previous_effective[target],
            }
            previous_effective[target] = norm
        row = {
            'schema': 'saal-static-pilot-step-v5', 'completed_optimizer_step': step,
            'sample_offset': stream_offset, 'sample_exposures': stream_offset,
            'tokens': total_tokens,
            'cross_entropy': sum(v['cross_entropy'] for v in objective_terms) / len(objective_terms),
            'orthogonal_penalty': sum(v['orthogonal_penalty'] for v in objective_terms) / len(objective_terms),
            'weighted_orthogonal_penalty': sum(v['weighted_orthogonal_penalty'] for v in objective_terms) / len(objective_terms),
            'gradient_contributions': diagnostic_terms,
            'gradient_before_clip': preclip,
            'optimizer_update_observation': after_parameters,
            'optimizer_parameter_update_norm': update_norm,
            'optimizer_updated': True,
            'optimizer_step': step,
            'precision': precision,
            'grad_scaler_scale': (float(scaler.get_scale())
                                  if scaler is not None else None),
            'learning_rates': {group['name']: group['lr'] for group in optimizer.param_groups},
            'effective_update_by_target': effective,
            'active_rank_by_target': {target: backend.adapter['init_r']
                                      for target in backend.adapter['target_modules']},
            'elapsed_seconds': time.monotonic() - start,
            'gpu_memory': {
                'allocated_bytes': torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0,
                'reserved_bytes': torch.cuda.memory_reserved(device) if device.type == 'cuda' else 0,
            },
        }
        _write_jsonl(progress_path, row, maximum_bytes=execution_permit.output_limit)
        check_output_budget()
        optimizer.zero_grad(set_to_none=True)
        previous_parameters = capture_parameter_snapshot(backend.trainable.items())
        previous_updates = current_updates

        if step % config['evaluation_interval'] == 0 or step == config['steps']:
            state = evaluate_step(step)
            if eligible_trained_checkpoint(state['metrics'], initial['metrics']):
                candidate_key = _checkpoint_selection_key(state['metrics'])
                best_key = None if best is None else _checkpoint_selection_key(best['metrics'])
                if best_key is None or candidate_key > best_key:
                    best = state
                    best_record = save_checkpoint('best', best)
                    (out / 'best.json').write_bytes(canonical(best_record))

    terminal = evaluate_step(config['steps'])
    terminal_record = save_checkpoint('terminal', terminal)
    (out / 'terminal.json').write_bytes(canonical(terminal_record))
    if stream_offset != len(stream):
        raise ValueError('sample exposure / optimizer step mismatch')
    frozen = backend.verify_frozen_weights()
    check_deadline()
    replays = {}
    for name, state in (('initial', initial), ('best', best), ('terminal', terminal)):
        if name == 'best' and best is None:
            continue
        rank_state = json.loads((out / f'{name}_rank_state.json').read_text())
        check_deadline()
        backend.restore_checkpoint(out / f'{name}.safetensors', rank_state)
        check_deadline()
        replay_metrics, replay = _evaluate(backend, development)
        check_deadline()
        replays[name] = {
            'predictions_match': prediction_equal(
                replay, state['predictions'], 'modernbert-seqcls-v1', REPLAY_TOLERANCE),
            'metrics_match': canonical(replay_metrics) == canonical(state['metrics']),
        }
        if not all(replays[name].values()):
            raise ValueError(f'checkpoint replay failed: {name}')
    result = {
        'schema': 'saal-static-adapter-pilot-result-v5',
        'configuration_digest': sha(config), 'dataset_digest': sha(dataset),
        'training_seed': identity['seed'], 'dropout_seed': identity['dropout_seed'],
        'metric_policy_digest': identity['metric_policy_digest'],
        'sample_order_digest': sha(stream), 'sample_exposures': len(stream),
        'completed_optimizer_updates': config['steps'],
        'selected_checkpoint_status': 'SELECTED_TRAINED_CHECKPOINT' if best else 'NO_QUALIFYING_CHECKPOINT',
        'selected_checkpoint_step': best['step'] if best else None,
        'initial': initial_record, 'best': best_record, 'terminal': terminal_record,
        'frozen_weights': frozen, 'replay': replays,
        'resource_accounting': {
            'elapsed_seconds': time.monotonic() - start,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
            'trainable_parameters': sum(p.numel() for p in backend.trainable.values()),
            'active_rank_total': backend.adapter['init_r'] * len(backend.adapter['target_modules']),
        },
        'qualification_status': 'NOT_EVALUATED',
        'external_deployment_allowed': False,
    }
    check_deadline()
    (out / 'result.json').write_bytes(canonical(result))
    check_deadline()
    return result
