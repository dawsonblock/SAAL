import torch
from peft import AdaLoraConfig, get_peft_model
from transformers.modeling_outputs import SequenceClassifierOutput

from saal_rsi.adaptive_backend import AdaptiveModernBERTBackend
from saal_rsi.adaptive_contract import adapter_spec


def test_explicit_regularizer_matches_pinned_peft_loss():
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.left = torch.nn.Linear(8, 8); self.classifier = torch.nn.Linear(8, 4)
        def forward(self, x, labels=None):
            logits = self.classifier(self.left(x))
            loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)
    torch.manual_seed(60606)
    model = get_peft_model(Fixture(), AdaLoraConfig(init_r=12, target_r=8, total_step=1000,
        tinit=100, tfinal=200, target_modules=['left'], modules_to_save=['classifier'], orth_reg_weight=.5))
    model.eval(); x = torch.randn(4, 8); labels = torch.tensor([0, 1, 2, 3])
    backend = object.__new__(AdaptiveModernBERTBackend)
    backend.adapter = adapter_spec('adalora-12-to-8'); backend.model = model
    backend.trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    # PeftModel may bypass the tuner's forward. Compare directly with the
    # pinned AdaLoraModel routine that implements the orthogonal penalty.
    actual = model.base_model(x, labels=labels).loss
    explicit = torch.nn.functional.cross_entropy(model(x).logits, labels)+backend.orthogonal_regularizer()
    assert torch.allclose(actual, explicit, atol=1e-7, rtol=1e-7)
