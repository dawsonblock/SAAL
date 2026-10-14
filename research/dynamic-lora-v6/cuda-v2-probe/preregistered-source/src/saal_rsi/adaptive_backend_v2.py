"""Explicit normalization of PEFT 0.21.2 CUDA ranknum registration.

ParameterDict can wrap a moved, non-trainable Tensor as a trainable Parameter.
Normalize only the declared immutable scaling constants; never train them or
change their values. The original v1 worker keeps rejecting that condition.
"""
from .adaptive_backend import AdaptiveModernBERTBackend, tensor_inventory
from .adaptive_contract_v2 import RANKNUM_POLICY


def freeze_ranknum(parameters, expected_rank=12):
    import torch
    values = list(parameters.values())
    if not values or any(list(p.shape) != [1] or p.dtype != torch.float32
                         or not torch.isfinite(p).all() or p.item() != float(expected_rank)
                         for p in values):
        raise ValueError('exact immutable ranknum scaling values required')
    changed = sum(p.requires_grad for p in values)
    for p in values:
        p.requires_grad_(False)
    return changed


class CudaSafeAdaLoRABackend(AdaptiveModernBERTBackend):
    def initialize_experiment(self):
        import torch
        self.ranknum_normalized = 0
        try:
            value = super().initialize_experiment()
        except ValueError as exc:
            if str(exc) != 'ranknum must be immutable' or self.adapter['method'] != 'adalora-12-to-8':
                raise
            # All preceding native target/inventory checks have already passed.
            self.ranknum_normalized = freeze_ranknum(self.ranknum)
            self.trainable = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            if {n: list(p.shape) for n, p in (self.trainable | self.ranknum).items()} != tensor_inventory(self.adapter):
                raise ValueError('v2 exact trainable/scaling inventory')
            self.initial_frozen_hash = self.frozen_hash()
            self.initial_trainable_hash = self.trainable_hash()
            torch.manual_seed(self.job['dropout_seed'])
            value = self.parameter_inventory()
        if any(p.requires_grad or p.grad is not None for p in self.ranknum.values()):
            raise ValueError('immutable ranknum remains trainable')
        self.ranknum_policy = RANKNUM_POLICY
        return value
