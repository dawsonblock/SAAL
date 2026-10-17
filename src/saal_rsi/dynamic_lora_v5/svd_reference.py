"""Independent PEFT AdaLoRA SVD equations and bounded diagnostics.

For PEFT's Linear adapter, the low-rank contribution is
``(x @ (A * E).T @ B.T) * scaling / (ranknum + 1e-5)``.  These routines do
not modify parameter gradients or training state.
"""
from __future__ import annotations

import math

PEFT_RANKNUM_EPSILON = 1e-5


def _check_factors(adapter_a, singular_values, adapter_b, *, scaling, ranknum, mask=None):
    import torch
    if any(not isinstance(value, torch.Tensor) for value in (adapter_a, singular_values, adapter_b)):
        raise TypeError('SVD adapter factors must be tensors')
    if adapter_a.ndim != 2 or adapter_b.ndim != 2:
        raise ValueError('A and B must be rank-two matrices')
    rank = adapter_a.shape[0]
    if singular_values.numel() != rank or adapter_b.shape[1] != rank:
        raise ValueError('SVD factor rank mismatch')
    if (adapter_a.device != adapter_b.device or adapter_a.device != singular_values.device
            or adapter_a.dtype != adapter_b.dtype or adapter_a.dtype != singular_values.dtype):
        raise ValueError('SVD factors must share device and dtype')
    if (type(scaling) not in (int, float) or isinstance(scaling, bool)
            or type(ranknum) not in (int, float) or isinstance(ranknum, bool)
            or not math.isfinite(float(scaling)) or not math.isfinite(float(ranknum)) or ranknum <= 0):
        raise ValueError('finite scaling and positive ranknum required')
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.numel() != rank or mask.dtype != torch.bool:
            raise ValueError('Boolean rank mask with exact allocated rank required')
        if mask.device != adapter_a.device:
            raise ValueError('rank mask must share the adapter device')
    for tensor in (adapter_a, singular_values, adapter_b):
        if not torch.isfinite(tensor).all():
            raise ValueError('nonfinite SVD factor')


def effective_delta_weight(adapter_a, singular_values, adapter_b, *, scaling=1.0,
                           ranknum=1.0, mask=None):
    """Return ``scaling/(ranknum+1e-5) * B diag(E * mask) A``."""
    _check_factors(adapter_a, singular_values, adapter_b, scaling=scaling,
                   ranknum=ranknum, mask=mask)
    values = singular_values.reshape(-1)
    if mask is not None:
        values = values * mask.to(dtype=values.dtype)
    values = values * (float(scaling) / (float(ranknum) + PEFT_RANKNUM_EPSILON))
    return (adapter_b * values.unsqueeze(0)) @ adapter_a


def svd_linear_reference(inputs, base_weight, base_bias, adapter_a, singular_values,
                         adapter_b, *, scaling=1.0, ranknum=1.0, mask=None,
                         dropout=None):
    """Independent Linear forward reference, including optional fixed dropout."""
    import torch.nn.functional as F
    _check_factors(adapter_a, singular_values, adapter_b, scaling=scaling,
                   ranknum=ranknum, mask=mask)
    adapted_input = dropout(inputs) if dropout is not None else inputs
    a_scaled = adapter_a * singular_values.reshape(-1, 1)
    if mask is not None:
        a_scaled = a_scaled * mask.to(dtype=a_scaled.dtype).reshape(-1, 1)
    base = F.linear(inputs, base_weight, base_bias)
    update = F.linear(F.linear(adapted_input, a_scaled), adapter_b)
    return base + update * (float(scaling) / (float(ranknum) + PEFT_RANKNUM_EPSILON))


def effective_update_frobenius_norm(adapter_a, singular_values, adapter_b, *,
                                   scaling=1.0, ranknum=1.0, mask=None):
    """Compute ||B diag(E) A||_F through rank-sized Gram matrices."""
    import torch
    _check_factors(adapter_a, singular_values, adapter_b, scaling=scaling,
                   ranknum=ranknum, mask=mask)
    # Accumulate diagnostics in float64 without materializing the out x in
    # update matrix.  The cost is quadratic in the small adapter rank.
    a = adapter_a.detach().to(dtype=torch.float64)
    b = adapter_b.detach().to(dtype=torch.float64)
    e = singular_values.detach().reshape(-1).to(dtype=torch.float64)
    if mask is not None:
        e = e * mask.to(dtype=torch.float64)
    e = e * (float(scaling) / (float(ranknum) + PEFT_RANKNUM_EPSILON))
    scaled_b = b * e.unsqueeze(0)
    gram_b = scaled_b.T @ scaled_b
    gram_a = a @ a.T
    squared = (gram_b * gram_a.T).sum()
    # Roundoff can make a mathematically zero norm slightly negative.
    return float(torch.sqrt(torch.clamp(squared, min=0)).item())


def effective_update_change_frobenius_norm(previous, current):
    """Compute ||DeltaW_current - DeltaW_previous||_F from low-rank factors.

    Each mapping contains A, E, B and optional scaling, ranknum, and mask.
    This permits comparing updates from different parameter values/ranks.
    """
    import torch

    def factors(record):
        required = {'A', 'E', 'B'}
        if type(record) is not dict or not required.issubset(record):
            raise ValueError('update record requires A, E, and B')
        scale, ranknum, mask = record.get('scaling', 1.0), record.get('ranknum', 1.0), record.get('mask')
        _check_factors(record['A'], record['E'], record['B'], scaling=scale,
                       ranknum=ranknum, mask=mask)
        a = record['A'].detach().to(dtype=torch.float64)
        b = record['B'].detach().to(dtype=torch.float64)
        e = record['E'].detach().reshape(-1).to(dtype=torch.float64)
        if mask is not None:
            e = e * mask.to(dtype=torch.float64)
        return a, b * (e * (float(scale) / (float(ranknum) + PEFT_RANKNUM_EPSILON))).unsqueeze(0)

    a0, b0 = factors(previous)
    a1, b1 = factors(current)
    if a0.shape[1] != a1.shape[1] or b0.shape[0] != b1.shape[0]:
        raise ValueError('effective updates must have matching input/output dimensions')

    def squared_norm(a, b):
        return ((b.T @ b) * (a @ a.T).T).sum()

    cross_b = b0.T @ b1
    cross_a = a0 @ a1.T
    inner = (cross_b * cross_a).sum()
    squared = squared_norm(a0, b0) + squared_norm(a1, b1) - 2 * inner
    return float(torch.sqrt(torch.clamp(squared, min=0)).item())


def gradient_contribution_diagnostics(cross_entropy, orthogonal_penalty, parameters,
                                      coefficient, *, epsilon=1e-12):
    """Measure CE and weighted-orthogonality gradients without touching .grad."""
    import torch
    if type(coefficient) is not float or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('nonnegative finite regularization coefficient required')
    if type(epsilon) is not float or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('positive finite epsilon required')
    params = tuple(p for p in parameters if p.requires_grad)
    if not params or not isinstance(cross_entropy, torch.Tensor) or not cross_entropy.requires_grad:
        raise ValueError('differentiable CE and trainable parameters required')
    ce_grads = torch.autograd.grad(cross_entropy, params, retain_graph=True, allow_unused=True)
    if coefficient and not isinstance(orthogonal_penalty, torch.Tensor):
        raise ValueError('orthogonal penalty tensor required for nonzero coefficient')
    if coefficient and isinstance(orthogonal_penalty, torch.Tensor) and orthogonal_penalty.requires_grad:
        reg_grads = torch.autograd.grad(orthogonal_penalty, params, retain_graph=True, allow_unused=True)
    else:
        reg_grads = (None,) * len(params)

    # Keep aligned zero blocks for parameters unused by one objective.
    def aligned(gradients, multiplier=1.0):
        return torch.cat([
            (torch.zeros_like(param, dtype=torch.float64).reshape(-1) if grad is None
             else (grad.detach() * multiplier).reshape(-1).to(dtype=torch.float64))
            for param, grad in zip(params, gradients)
        ])
    ce_vec = aligned(ce_grads)
    reg_vec = aligned(reg_grads, coefficient)
    if not torch.isfinite(ce_vec).all() or not torch.isfinite(reg_vec).all():
        raise ValueError('nonfinite objective gradient')
    ce_norm = torch.linalg.vector_norm(ce_vec)
    reg_norm = torch.linalg.vector_norm(reg_vec)
    cosine = float((torch.dot(ce_vec, reg_vec) /
                    (ce_norm * reg_norm).clamp_min(epsilon)).item()) if reg_norm.item() else 0.0
    return {
        'cross_entropy_gradient_norm': float(ce_norm.item()),
        'weighted_orthogonal_gradient_norm': float(reg_norm.item()),
        'orthogonal_to_ce_norm_ratio': float((reg_norm / ce_norm.clamp_min(epsilon)).item()),
        'gradient_cosine_similarity': cosine,
        'trainable_parameter_count': len(params),
        'parameter_elements': sum(p.numel() for p in params),
        'coefficient': coefficient,
        'finite': True,
    }


def orthogonal_penalty_reference(adapter_pairs):
    """PEFT 0.21.2 orthogonal penalty before its configured coefficient.

    ``adapter_pairs`` contains ``(A, B)`` matrices for every adapted target.
    PEFT averages one Frobenius deviation per A/B matrix.
    """
    import torch
    penalties = []
    for adapter_a, adapter_b in adapter_pairs:
        if (not isinstance(adapter_a, torch.Tensor) or not isinstance(adapter_b, torch.Tensor)
                or adapter_a.ndim != 2 or adapter_b.ndim != 2
                or adapter_a.shape[0] != adapter_b.shape[1]):
            raise ValueError('paired AdaLoRA matrices with matching rank required')
        if adapter_a.device != adapter_b.device or adapter_a.dtype != adapter_b.dtype:
            raise ValueError('A/B device and dtype must match')
        for matrix, covariance in ((adapter_a, adapter_a @ adapter_a.T),
                                   (adapter_b, adapter_b.T @ adapter_b)):
            identity = torch.eye(covariance.shape[0], device=covariance.device,
                                 dtype=covariance.dtype)
            penalties.append(torch.linalg.matrix_norm(covariance - identity, ord='fro'))
    if not penalties:
        raise ValueError('at least one AdaLoRA target is required')
    return torch.stack(penalties).mean()


def trainable_parameter_diagnostics(named_parameters, *, previous=None, optimizer=None,
                                    clip_threshold=1.0):
    """Summarize per-tensor updates, gradients, Adam moments, and clip pressure.

    The caller should invoke this before clipping and before ``optimizer.step``
    for gradient evidence. ``previous`` is a snapshot from the preceding step.
    No gradients or optimizer state are modified.
    """
    import torch
    if type(clip_threshold) not in (int, float) or isinstance(clip_threshold, bool):
        raise ValueError('finite positive gradient clipping threshold required')
    if not math.isfinite(float(clip_threshold)) or clip_threshold <= 0:
        raise ValueError('finite positive gradient clipping threshold required')
    entries = {}
    gradient_square_sum = 0.0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if type(name) is not str or not name:
            raise ValueError('named trainable parameter required')
        value = parameter.detach().to(dtype=torch.float64)
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite trainable parameter')
        gradient = parameter.grad
        if gradient is None:
            gradient_norm = None
        else:
            gradient64 = gradient.detach().to(dtype=torch.float64)
            if not torch.isfinite(gradient64).all():
                raise ValueError('nonfinite trainable gradient')
            gradient_norm = float(torch.linalg.vector_norm(gradient64).item())
            gradient_square_sum += gradient_norm * gradient_norm
        prior = None if previous is None else previous.get(name)
        change_norm = None
        if prior is not None:
            if prior.shape != parameter.shape:
                raise ValueError('previous parameter snapshot shape changed')
            change_norm = float(torch.linalg.vector_norm(value-prior.detach().to(dtype=torch.float64)).item())
        moment1 = moment2 = None
        step = None
        if optimizer is not None:
            state = optimizer.state.get(parameter, {})
            if 'exp_avg' in state:
                moment1 = float(torch.linalg.vector_norm(state['exp_avg'].detach().to(dtype=torch.float64)).item())
            if 'exp_avg_sq' in state:
                moment2 = float(torch.linalg.vector_norm(state['exp_avg_sq'].detach().to(dtype=torch.float64)).item())
            if 'step' in state:
                raw_step = state['step']
                step = int(raw_step.item()) if isinstance(raw_step, torch.Tensor) else int(raw_step)
        entries[name] = {
            'parameter_norm': float(torch.linalg.vector_norm(value).item()),
            'parameter_change_norm': change_norm,
            'gradient_norm': gradient_norm,
            'adam_exp_avg_norm': moment1,
            'adam_exp_avg_sq_norm': moment2,
            'optimizer_step': step,
        }
    if not entries:
        raise ValueError('trainable parameters required')
    pre_clip_norm = math.sqrt(gradient_square_sum)
    clip_scale = min(1.0, float(clip_threshold)/(pre_clip_norm+1e-6))
    return {
        'parameters': entries,
        'pre_clip_global_gradient_norm': pre_clip_norm,
        'clip_threshold': float(clip_threshold),
        'predicted_clip_scale': clip_scale,
        'would_clip': clip_scale < 1.0,
        'trainable_tensor_count': len(entries),
    }


def capture_parameter_snapshot(named_parameters):
    """Capture only trainable tensors for bounded step-to-step diagnostics."""
    snapshot = {}
    for name, parameter in named_parameters:
        if parameter.requires_grad:
            snapshot[name] = parameter.detach().clone()
    if not snapshot:
        raise ValueError('trainable parameters required for snapshot')
    return snapshot


def validate_rank_mask(mask, singular_values, aggregate_budget):
    """Fail closed when mask state, masked singular values, or budget disagree."""
    import torch
    if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool or mask.ndim != 1:
        raise ValueError('one-dimensional Boolean rank mask required')
    if not isinstance(singular_values, torch.Tensor) or singular_values.numel() != mask.numel():
        raise ValueError('singular-value inventory does not match allocated rank')
    if type(aggregate_budget) is not int or aggregate_budget < 0:
        raise ValueError('nonnegative integer aggregate budget required')
    if not torch.isfinite(singular_values).all():
        raise ValueError('nonfinite singular values')
    if torch.any(singular_values.reshape(-1)[~mask] != 0):
        raise ValueError('masked singular component is nonzero')
    active = int(mask.sum().item())
    if active > aggregate_budget:
        raise ValueError('active rank exceeds aggregate allocation budget')
    return active
