import pytest
import torch
from safetensors.torch import save_file

from saal_rsi.adaptive_backend import tensor_inventory
from saal_rsi.adaptive_contract import adapter_spec
from saal_rsi.adaptive_rank_controller import initial_state
from saal_rsi.adaptive_artifacts import validate_checkpoint


@pytest.mark.parametrize('method', ['fixed-r8', 'fixed-r12', 'adalora-12-to-8'])
def test_exact_versioned_tensors_and_scaling(tmp_path, method):
    a = adapter_spec(method); state = initial_state(a)
    tensors = {n: torch.zeros(shape) for n, shape in tensor_inventory(a).items()}
    for n in tensors:
        if '.ranknum.' in n: tensors[n].fill_(12)
    file = tmp_path/'checkpoint.safetensors'; save_file(tensors, str(file))
    assert set(validate_checkpoint(file, a, state)) == set(tensors)


@pytest.mark.parametrize('fault', ['singular_missing', 'fixed_substitution', 'ranknum', 'nan', 'dtype',
                                  'shape', 'unsupported', 'mask', 'masked_nonzero'])
def test_bad_dynamic_checkpoint_rejected(tmp_path, fault):
    a = adapter_spec('adalora-12-to-8'); state = initial_state(a)
    tensors = {n: torch.zeros(shape) for n, shape in tensor_inventory(a).items()}
    for n in tensors:
        if '.ranknum.' in n: tensors[n].fill_(12)
    e = next(n for n in tensors if '.lora_E.' in n)
    if fault == 'singular_missing': del tensors[e]
    elif fault == 'fixed_substitution': tensors = {n: torch.zeros(shape) for n, shape in tensor_inventory(adapter_spec('fixed-r8')).items()}
    elif fault == 'ranknum': tensors[next(n for n in tensors if '.ranknum.' in n)].fill_(8)
    elif fault == 'nan': tensors[e].fill_(float('nan'))
    elif fault == 'dtype': tensors[e] = tensors[e].half()
    elif fault == 'shape': tensors[e] = tensors[e][:8]
    elif fault == 'unsupported': tensors['evaluator.private_key'] = torch.zeros(1)
    elif fault == 'mask': state['masks'][a['target_modules'][0]][0] = 1
    else:
        state['masks'][a['target_modules'][0]][0] = False
        tensors[e][0] = 1
    file = tmp_path/'checkpoint.safetensors'; save_file(tensors, str(file))
    with pytest.raises(ValueError): validate_checkpoint(file, a, state)
