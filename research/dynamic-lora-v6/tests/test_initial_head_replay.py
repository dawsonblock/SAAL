import copy

import pytest
import torch

from saal_rsi.adaptive_replay import restore_initial_head


class HeadFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Module()
        self.head.dense = torch.nn.Linear(4, 4, bias=False)
        self.head.norm = torch.nn.LayerNorm(4, bias=False)
        self.classifier = torch.nn.Linear(4, 4)


def checkpoint(model):
    result = {}
    for name, p in model.named_parameters():
        module, suffix = name.split('.', 1)
        result['base_model.model.'+module+'.modules_to_save.default.'+suffix] = p.detach().clone()
    return result


def test_admitted_head_bytes_replace_nonportable_seed_initialization_exactly():
    torch.manual_seed(1); admitted = HeadFixture(); values = checkpoint(admitted)
    torch.manual_seed(2); replay = HeadFixture()
    before = replay.classifier.weight.detach().clone()
    assert not torch.equal(before, admitted.classifier.weight)
    restore_initial_head(replay, values)
    assert all(torch.equal(p, dict(admitted.named_parameters())[n]) for n, p in replay.named_parameters())


@pytest.mark.parametrize('fault', ['missing', 'nonfinite', 'shape', 'dtype'])
def test_invalid_initial_head_rejected_before_any_tensor_mutation(fault):
    model = HeadFixture(); original = copy.deepcopy(model.state_dict()); values = checkpoint(model)
    key = 'base_model.model.classifier.modules_to_save.default.bias'
    if fault == 'missing':
        del values[key]
    elif fault == 'nonfinite':
        values[key][0] = float('nan')
    elif fault == 'shape':
        values[key] = torch.ones(5)
    else:
        values[key] = values[key].double()
    values['base_model.model.head.modules_to_save.default.dense.weight'] += 1
    with pytest.raises(ValueError):
        restore_initial_head(model, values)
    assert all(torch.equal(v, original[n]) for n, v in model.state_dict().items())
