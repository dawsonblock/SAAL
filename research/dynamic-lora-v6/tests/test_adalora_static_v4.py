import copy
import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model

from saal_rsi.adaptive_contract_v4 import adapter_spec
from saal_rsi.adaptive_backend_v4 import build_adapter_config
from saal_rsi.adaptive_rank_controller_v4 import RankController
from saal_rsi.adaptive_training_v4 import optimizer_update


class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.left = torch.nn.Linear(16, 16)
        self.model.right = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, 16)
        self.classifier = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.classifier(self.head(torch.tanh(self.model.right(torch.tanh(self.model.left(x))))))


def test_static_adalora_updates_without_rank_allocator(monkeypatch):
    torch.manual_seed(4711)
    torch.set_num_threads(1)
    targets = ['model.left', 'model.right']
    adapter = adapter_spec('adalora-static-r12', total_steps=4,
                           tinit=1, tfinal=1, delta=1)
    config = build_adapter_config(adapter)
    assert isinstance(config, AdaLoraConfig)
    assert config.init_r == config.target_r == 12
    config.target_modules = set(targets)
    model = get_peft_model(Fixture(), config)
    test_adapter = copy.deepcopy(adapter)
    test_adapter['target_modules'] = targets
    controller = RankController(model, test_adapter)
    monkeypatch.setattr(model.base_model, 'update_and_allocate',
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('static arm invoked allocator')))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.001)
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    for step in range(1, 5):
        optimizer.zero_grad(set_to_none=True)
        x = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 64
        loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        updated, gradients = optimizer_update(optimizer, trainable)
        event = controller.complete_update(step, optimizer_updated=updated,
            gradient_evidence=gradients, elapsed_seconds=float(step),
            gpu_memory={'allocated_bytes': 0, 'reserved_bytes': 0})
        assert event['allocated_this_update'] is False
        assert event['aggregate_active_rank'] == 24
        assert event['importance_scores'] == {}
        optimizer.zero_grad(set_to_none=True)
    result = controller.require_completion(4)
    after = {name: value.detach() for name, value in model.named_parameters() if value.requires_grad}
    assert result['allocation_events'] == 0
    assert result['active_rank_total'] == 24
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_fixed_lora_configs_without_rank_schedule():
    for method, expected_rank in [('fixed-r8',8),('fixed-r12',12)]:
        config=build_adapter_config(adapter_spec(method))
        assert isinstance(config,LoraConfig)
        assert config.r==expected_rank
