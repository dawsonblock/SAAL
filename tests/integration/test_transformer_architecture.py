import torch
import pytest
from saal_rsi.neural_contract import ModelSpec
from saal_rsi.model_architectures import initialized,tokenize,configure_training,adapter_name

@pytest.fixture
def spec():return ModelSpec(hidden_dimension=32,encoder_layers=1,feedforward_dimension=128,dropout=0.0).as_dict()

def test_order_and_padding(spec):
    torch.set_num_threads(2);model=initialized(spec).eval();prompts=['Choose fast, not cheap.','Choose cheap, not fast.']
    with torch.no_grad():
        ordinary=model(*tokenize(prompts,spec));padded=model(*tokenize(prompts,spec,pad_to=100))
        tokens,mask=tokenize(prompts,spec,pad_to=100);tokens[~mask]=123;attacked=model(tokens,mask)
    assert not torch.allclose(ordinary[0],ordinary[1],atol=1e-5)
    assert torch.allclose(ordinary,padded,atol=1e-6,rtol=1e-5)
    assert torch.allclose(ordinary,attacked,atol=1e-6,rtol=1e-5)

@pytest.mark.parametrize('prompt',['','x'*257,'é'*129])
def test_explicit_input_bounds(spec,prompt):
    with pytest.raises(ValueError):tokenize([prompt],spec)

def test_byte_preservation_and_eval_dropout(spec):
    spec['dropout']=.1;model=initialized(spec).eval();tokens,mask=tokenize(['éX'],spec)
    assert tokens[0].tolist()==[257,195,169,88]
    with torch.no_grad():assert torch.equal(model(tokens,mask),model(tokens,mask))

def test_lora_freeze(spec):
    model=initialized(spec);before={n:p.detach().clone() for n,p in model.named_parameters()}
    parameters=configure_training(model,'lora');opt=torch.optim.AdamW(parameters,lr=.01)
    x=tokenize(['Choose cheap','Choose fast'],spec);opt.zero_grad();torch.nn.functional.cross_entropy(model(*x),torch.tensor([0,1])).backward();opt.step()
    changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
    assert changed and all(adapter_name(n) for n in changed)


def test_supervised_order_counterfactual_learning(spec):
    # Engineering capability test, not a generalization estimate: fit four known prompts.
    torch.set_num_threads(2);torch.manual_seed(17);model=initialized(spec);opt=torch.optim.AdamW(configure_training(model,'full'),lr=.006)
    prompts=['Choose fast, not cheap.','Choose cheap, not fast.','Goal missing: ask.','Permission revoked: refuse.']
    tokens,mask=tokenize(prompts,spec);target=torch.tensor([1,0,2,3])
    for step in range(150):
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(tokens,mask),target);loss.backward();opt.step()
    model.eval()
    with torch.no_grad():assert model(tokens,mask).argmax(-1).tolist()==target.tolist()
