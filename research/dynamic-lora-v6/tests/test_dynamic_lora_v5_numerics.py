import copy

import pytest
import torch
from peft import AdaLoraConfig, get_peft_model

from saal_rsi.dynamic_lora_v5.svd_reference import (
    capture_parameter_snapshot,
    effective_delta_weight,
    effective_update_change_frobenius_norm,
    effective_update_frobenius_norm,
    gradient_contribution_diagnostics,
    orthogonal_penalty_reference,
    svd_linear_reference,
    trainable_parameter_diagnostics,
    validate_rank_mask,
)


class LinearFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 7)

    def forward(self, inputs, labels=None):
        logits = self.linear(inputs)
        if labels is None:
            return logits
        from types import SimpleNamespace
        return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits, labels), logits=logits)


def make_static_model(seed=91):
    torch.manual_seed(seed)
    config = AdaLoraConfig(
        init_r=3,
        target_r=3,
        lora_alpha=6,
        lora_dropout=0.0,
        tinit=1,
        tfinal=1,
        deltaT=1,
        total_step=10,
        orth_reg_weight=0.0,
        target_modules=['linear'],
        bias='none',
    )
    return get_peft_model(LinearFixture(), config)


def test_peft_svd_forward_matches_independent_reference_and_zero_initialization():
    torch.manual_seed(17)
    model = make_static_model()
    layer = model.base_model.model.linear
    inputs = torch.randn(4, 5)
    with torch.no_grad():
        assert torch.count_nonzero(layer.lora_E['default']) == 0
    expected_initial = layer.base_layer(inputs)
    torch.testing.assert_close(model(inputs), expected_initial, rtol=1e-6, atol=1e-6)

    with torch.no_grad():
        layer.lora_A['default'].copy_(torch.randn_like(layer.lora_A['default']))
        layer.lora_B['default'].copy_(torch.randn_like(layer.lora_B['default']))
        layer.lora_E['default'].copy_(torch.tensor([[0.2], [-0.4], [0.7]]))
    ranknum = float(layer.ranknum['default'].item())
    expected = svd_linear_reference(
        inputs,
        layer.base_layer.weight,
        layer.base_layer.bias,
        layer.lora_A['default'],
        layer.lora_E['default'],
        layer.lora_B['default'],
        scaling=layer.scaling['default'],
        ranknum=ranknum,
    )
    torch.testing.assert_close(model(inputs), expected, rtol=1e-6, atol=1e-6)

    disabled = layer.base_layer(inputs)
    with model.disable_adapter():
        torch.testing.assert_close(model(inputs), disabled, rtol=1e-6, atol=1e-6)


def test_peft_svd_backward_matches_independent_reference():
    torch.manual_seed(59)
    model = make_static_model()
    layer = model.base_model.model.linear
    with torch.no_grad():
        layer.lora_A['default'].copy_(torch.randn_like(layer.lora_A['default']))
        layer.lora_B['default'].copy_(torch.randn_like(layer.lora_B['default']))
        layer.lora_E['default'].copy_(torch.randn_like(layer.lora_E['default']))
    x = torch.randn(3, 5, requires_grad=True)
    coefficients = torch.randn(3, 7)
    actual = model(x)
    actual_grads = torch.autograd.grad(
        (actual * coefficients).sum(),
        [x, layer.lora_A['default'], layer.lora_E['default'], layer.lora_B['default']],
    )

    x_ref = x.detach().clone().requires_grad_(True)
    reference = svd_linear_reference(
        x_ref, layer.base_layer.weight, layer.base_layer.bias,
        layer.lora_A['default'], layer.lora_E['default'], layer.lora_B['default'],
        scaling=layer.scaling['default'], ranknum=float(layer.ranknum['default'].item()),
    )
    reference_grads = torch.autograd.grad(
        (reference * coefficients).sum(),
        [x_ref, layer.lora_A['default'], layer.lora_E['default'], layer.lora_B['default']],
    )
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)


def test_peft_orthogonal_objective_matches_reference_coefficient():
    torch.manual_seed(73)
    model = make_static_model()
    model.base_model.peft_config['default'].orth_reg_weight = 0.1
    layer = model.base_model.model.linear
    with torch.no_grad():
        layer.lora_A['default'].copy_(torch.randn_like(layer.lora_A['default']))
        layer.lora_B['default'].copy_(torch.randn_like(layer.lora_B['default']))
        layer.lora_E['default'].copy_(torch.randn_like(layer.lora_E['default']))
    x, labels = torch.randn(4, 5), torch.tensor([0, 1, 2, 3])
    # The pinned AdaLoraModel implements the PEFT regularizer. The generic
    # PeftModel wrapper bypasses that loss augmentation for this task_type=None
    # configuration, which is why SAAL computes the term explicitly.
    output = model.base_model(x, labels=labels)
    outer_output = model(x, labels=labels)
    logits = model(x)
    ce = torch.nn.functional.cross_entropy(logits, labels)
    expected = ce + 0.1 * orthogonal_penalty_reference(
        [(layer.lora_A['default'], layer.lora_B['default'])]
    )
    torch.testing.assert_close(output.loss, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outer_output.loss, ce, rtol=1e-6, atol=1e-6)
    raw_penalty = orthogonal_penalty_reference(
        [(layer.lora_A['default'], layer.lora_B['default'])]
    )
    gradients = gradient_contribution_diagnostics(
        ce, raw_penalty,
        [layer.lora_A['default'], layer.lora_B['default'], layer.lora_E['default']],
        0.1,
    )
    assert gradients['finite'] is True
    assert gradients['cross_entropy_gradient_norm'] > 0
    assert gradients['weighted_orthogonal_gradient_norm'] > 0
    assert all(parameter.grad is None for parameter in
               (layer.lora_A['default'], layer.lora_B['default'], layer.lora_E['default']))


def test_masked_rank_reference_is_zeroed_and_budget_checked():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    e = torch.tensor([[2.0], [3.0], [4.0]])
    b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    mask = torch.tensor([True, False, True])
    masked_e = e.reshape(-1).clone()
    masked_e[~mask] = 0
    expected = (b * (masked_e * (2.0 / (3.0 + 1e-5))).unsqueeze(0)) @ a
    torch.testing.assert_close(
        effective_delta_weight(a, e, b, scaling=2.0, ranknum=3.0, mask=mask),
        expected,
    )
    assert validate_rank_mask(mask, torch.tensor([[2.0], [0.0], [4.0]]), 2) == 2
    with pytest.raises(ValueError, match='masked singular component'):
        validate_rank_mask(mask, e, 3)
    with pytest.raises(ValueError, match='exceeds'):
        validate_rank_mask(mask, torch.tensor([[2.0], [0.0], [4.0]]), 1)


def test_low_rank_effective_update_norms_match_materialized_matrices():
    torch.manual_seed(31)
    a0, b0, e0 = torch.randn(3, 5), torch.randn(7, 3), torch.randn(3)
    a1, b1, e1 = torch.randn(3, 5), torch.randn(7, 3), torch.randn(3)
    old = {'A': a0, 'E': e0, 'B': b0, 'scaling': 2.0, 'ranknum': 3.0}
    new = {'A': a1, 'E': e1, 'B': b1, 'scaling': 2.0, 'ranknum': 3.0}
    delta0 = effective_delta_weight(a0, e0, b0, scaling=2.0, ranknum=3.0)
    delta1 = effective_delta_weight(a1, e1, b1, scaling=2.0, ranknum=3.0)
    assert effective_update_frobenius_norm(
        a1, e1, b1, scaling=2.0, ranknum=3.0
    ) == pytest.approx(float(torch.linalg.matrix_norm(delta1, ord='fro')), rel=1e-6)
    assert effective_update_change_frobenius_norm(old, new) == pytest.approx(
        float(torch.linalg.matrix_norm(delta1 - delta0, ord='fro')), rel=1e-6
    )


def test_gradient_diagnostics_separate_ce_and_regularizer_without_mutating_gradients():
    value = torch.nn.Parameter(torch.tensor(0.0))
    ce = (value - 2.0).square()
    orthogonal = (value + 1.0).square()
    result = gradient_contribution_diagnostics(ce, orthogonal, [value], 0.5)
    assert result['cross_entropy_gradient_norm'] == pytest.approx(4.0)
    assert result['weighted_orthogonal_gradient_norm'] == pytest.approx(1.0)
    assert result['orthogonal_to_ce_norm_ratio'] == pytest.approx(0.25)
    assert result['gradient_cosine_similarity'] == pytest.approx(-1.0)
    assert result['finite'] is True
    assert value.grad is None


def test_zero_regularization_reports_zero_pressure():
    value = torch.nn.Parameter(torch.tensor(0.0))
    result = gradient_contribution_diagnostics(
        value.square(), (value - 1.0).square(), [value], 0.0
    )
    assert result['weighted_orthogonal_gradient_norm'] == 0.0
    assert result['orthogonal_to_ce_norm_ratio'] == 0.0
    assert result['gradient_cosine_similarity'] == 0.0


def test_orthogonal_penalty_uses_peft_mean_over_A_and_B_matrices():
    torch.manual_seed(115)
    pairs = [(torch.randn(3, 5), torch.randn(7, 3)),
             (torch.randn(3, 4), torch.randn(6, 3))]
    expected = []
    for a, b in pairs:
        expected.extend([
            torch.linalg.matrix_norm(a @ a.T - torch.eye(3), ord='fro'),
            torch.linalg.matrix_norm(b.T @ b - torch.eye(3), ord='fro'),
        ])
    torch.testing.assert_close(orthogonal_penalty_reference(pairs), torch.stack(expected).mean())


def test_parameter_diagnostics_report_clip_pressure_update_and_adam_moments():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    before = capture_parameter_snapshot([('adapter.lora_E', parameter)])
    parameter.grad = torch.tensor([3.0, 4.0])
    pre = trainable_parameter_diagnostics([('adapter.lora_E', parameter)], clip_threshold=1.0)
    assert pre['pre_clip_global_gradient_norm'] == pytest.approx(5.0)
    assert pre['predicted_clip_scale'] == pytest.approx(0.2, abs=1e-6)
    assert pre['would_clip'] is True
    optimizer.step()
    post = trainable_parameter_diagnostics([('adapter.lora_E', parameter)],
                                           previous=before, optimizer=optimizer)
    item = post['parameters']['adapter.lora_E']
    assert item['parameter_change_norm'] > 0
    assert item['adam_exp_avg_norm'] > 0
    assert item['adam_exp_avg_sq_norm'] > 0
    assert item['optimizer_step'] == 1


def test_initial_singular_values_unlock_factors_after_first_update():
    torch.manual_seed(71)
    model = make_static_model()
    params = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert any('lora_E' in name for name in params)
    optimizer = torch.optim.AdamW(list(params.values()), lr=1e-2)
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 20
    labels = torch.tensor([0, 1, 2, 3])
    changed = set()
    for step in range(10):
        before = {name: value.detach().clone() for name, value in params.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), labels)
        loss.backward()
        for name, value in params.items():
            if value.grad is None or not torch.isfinite(value.grad).all():
                raise AssertionError(f'missing/nonfinite gradient at step {step}: {name}')
        optimizer.step()
        changed.update(name for name, value in params.items() if not torch.equal(before[name], value))
    assert any('lora_E' in name for name in changed)
    assert any('lora_A' in name for name in changed)
    assert any('lora_B' in name for name in changed)


def test_stock_peft_masked_singular_value_regrows_and_keeps_adam_moments_until_remasked():
    """Characterize PEFT behavior; SAAL's current controller reapplies masks post-step."""
    torch.manual_seed(143)
    model = make_static_model()
    layer = model.base_model.model.linear
    with torch.no_grad():
        layer.lora_A['default'].copy_(torch.randn_like(layer.lora_A['default']))
        layer.lora_B['default'].copy_(torch.randn_like(layer.lora_B['default']))
        layer.lora_E['default'].copy_(torch.tensor([[0.0], [0.5], [1.0]]))
    pattern = {'linear.lora_E.default': [False, True, True]}
    model.base_model.peft_config['default'].rank_pattern = pattern
    model.base_model.rankallocator.mask_using_rank_pattern(model.base_model.model, pattern)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    inputs = torch.randn(8, 5)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(inputs), labels).backward()
    optimizer.step()
    # PEFT's masking operation zeros the parameter but does not mask its
    # gradient or Adam moments. A normal optimizer update can regrow it.
    assert float(layer.lora_E['default'][0].detach()) != 0.0
    state = optimizer.state[layer.lora_E['default']]
    assert float(state['exp_avg'][0].detach()) != 0.0

    model.base_model.rankallocator.mask_using_rank_pattern(model.base_model.model, pattern)
    assert float(layer.lora_E['default'][0].detach()) == 0.0
    assert float(state['exp_avg'][0].detach()) != 0.0
    assert validate_rank_mask(
        torch.tensor([False, True, True]), layer.lora_E['default'], 2
    ) == 2


def test_static_checkpoint_reload_reproduces_outputs_and_rank_state():
    model = make_static_model(seed=111)
    layer = model.base_model.model.linear
    with torch.no_grad():
        layer.lora_E['default'].copy_(torch.tensor([[0.1], [-0.2], [0.3]]))
    inputs = torch.randn(6, 5)
    expected = model(inputs).detach().clone()
    state = copy.deepcopy(model.state_dict())
    restored = make_static_model(seed=222)
    restored.load_state_dict(state, strict=True)
    actual = restored(inputs).detach()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored_layer = restored.base_model.model.linear
    assert torch.equal(restored_layer.ranknum['default'], layer.ranknum['default'])
