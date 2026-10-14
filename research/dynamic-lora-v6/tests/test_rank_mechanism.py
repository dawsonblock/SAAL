import pytest
import torch
from peft import AdaLoraConfig, get_peft_model

from saal_rsi.adaptive_backend import verify_peft_implementation
from saal_rsi.adaptive_contract import adapter_spec
from saal_rsi.adaptive_rank_controller import RankController
from saal_rsi.adaptive_training import optimizer_update


class EngineeringFixture(torch.nn.Module):
    """Artificial two-projection mechanism fixture, never a research model."""
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.left = torch.nn.Linear(8, 8)
        self.model.right = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 8)
        self.classifier = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.classifier(self.head(torch.tanh(self.model.right(torch.tanh(self.model.left(x))))))


def mechanism():
    torch.manual_seed(60606)
    torch.set_num_threads(1)
    config = AdaLoraConfig(init_r=12, target_r=8, lora_alpha=24, lora_dropout=.05,
        tinit=100, tfinal=200, deltaT=10, beta1=.85, beta2=.85, orth_reg_weight=.5,
        total_step=1000, target_modules=['model.left', 'model.right'], modules_to_save=['head', 'classifier'], bias='none')
    model = get_peft_model(EngineeringFixture(), config)
    adapter = adapter_spec('adalora-12-to-8')
    adapter['target_modules'] = ['model.left', 'model.right']
    return model, RankController(model, adapter), adapter


def test_real_peft_full_1000_update_schedule_and_determinism():
    verify_peft_implementation()
    def run():
        model, controller, a = mechanism()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=.001)
        x = torch.randn(8, 8); labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        for step in range(1, 1001):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), labels)
            regularizers = []
            for n, p in model.named_parameters():
                if p.requires_grad and ('.lora_A.' in n or '.lora_B.' in n):
                    cov = p@p.T if '.lora_A.' in n else p.T@p
                    regularizers.append(torch.linalg.matrix_norm(cov-torch.eye(12), ord='fro'))
            (loss+.5*torch.stack(regularizers).mean()).backward()
            updated, grads = optimizer_update(optimizer, trainable)
            event = controller.complete_update(step, optimizer_updated=updated, gradient_evidence=grads,
                                                elapsed_seconds=float(step)/100, gpu_memory={'allocated_bytes': 0, 'reserved_bytes': 0})
            assert event['scheduled_budget'] == model.base_model.rankallocator.budget_schedule(step)[0]
            assert event['aggregate_active_rank'] <= event['applied_budget']
        result = controller.require_completion(1000)
        assert result['allocation_events'] == 70
        assert 0 < result['active_rank_total'] <= 16
        assert all(controller.history[i]['state']['masks'] == controller.history[799]['state']['masks'] for i in range(800, 1000))
        assert controller.history[100]['applied_budget'] == 24
        return controller.state, {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    state, values = run(); again, repeated = run()
    assert state == again
    assert all(torch.equal(t, repeated[n]) for n, t in values.items())


def test_skipped_optimizer_and_microbatches_do_not_advance_rank():
    model, controller, a = mechanism()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters)
    for _ in range(3):
        torch.nn.functional.cross_entropy(model(torch.ones(4, 8)), torch.tensor([0, 1, 2, 3])).div(3).backward()
    assert controller.state['step'] == 0
    updated, gradients = optimizer_update(optimizer, parameters)
    controller.complete_update(1, optimizer_updated=updated, gradient_evidence=gradients,
                               elapsed_seconds=.1, gpu_memory={'allocated_bytes': 0, 'reserved_bytes': 0})
    assert controller.state['step'] == 1
    assert controller.complete_update(2, optimizer_updated=False, gradient_evidence={}, elapsed_seconds=.2, gpu_memory={}) is None
    assert controller.state['step'] == 1
    with pytest.raises(ValueError):
        controller.complete_update(1, optimizer_updated=True, gradient_evidence=gradients,
                                   elapsed_seconds=.2, gpu_memory={'allocated_bytes': 0, 'reserved_bytes': 0})


def test_amp_nonfinite_update_is_not_counted():
    p = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([p])
    scaler = torch.amp.GradScaler('cpu')
    scaler.scale(p.square().sum()).backward()
    p.grad.fill_(float('inf'))
    updated, gradients = optimizer_update(optimizer, [p], scaler)
    assert updated is False and gradients['finite'] is False and p.item() == 1


def test_optimizer_hook_detects_skipped_scaler_step_even_with_finite_gradients():
    p = torch.nn.Parameter(torch.ones(1)); p.square().sum().backward()
    optimizer = torch.optim.AdamW([p])
    class Skipped:
        def unscale_(self, optimizer): pass
        def step(self, optimizer): pass
        def update(self): pass
    updated, gradients = optimizer_update(optimizer, [p], Skipped())
    assert updated is False and gradients['finite'] is True and p.item() == 1


def test_completion_without_allocation_is_rejected():
    _, controller, _ = mechanism()
    with pytest.raises(ValueError): controller.require_completion(1000)
