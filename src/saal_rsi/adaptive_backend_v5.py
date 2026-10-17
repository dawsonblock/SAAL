"""Static LoRA/AdaLoRA ModernBERT backend for the v5 diagnostic pilot.

This is an additive research backend. It reuses the pinned v4 PEFT tensor
layout, but the v5 scientific configuration controls the orthogonal-loss
coefficient. No dynamic rank allocator is invoked here.
"""
from __future__ import annotations

import torch

from .adaptive_backend_v4 import AdaptiveModernBERTBackend
from .adaptive_contract_v4 import adapter_spec
from .dynamic_lora_v5.svd_reference import orthogonal_penalty_reference
from .pretrained_contract import DECISIONS


class StaticPilotBackendV5(AdaptiveModernBERTBackend):
    """PEFT 0.21.2 backend with a frozen rank pattern for the short pilot."""

    def __init__(self, config, snapshot):
        self.scientific_config = config
        method = config['method']
        if method == 'fixed-r8':
            adapter = adapter_spec('fixed-r8', total_steps=config['steps'])
        elif method == 'adalora-static-r12':
            # Valid PEFT schedule metadata is required by AdaLoraConfig, but
            # the static treatment deliberately never calls update_and_allocate.
            adapter = adapter_spec('adalora-static-r12', total_steps=config['steps'],
                                   tinit=10, tfinal=10, delta=10)
        else:
            raise ValueError('v5 static pilot backend method')
        job = {
            'adapter': adapter,
            'steps': config['steps'],
            'model_spec': config['identity']['model_spec'],
            'seed': config['identity']['seed'],
            'dropout_seed': config['identity']['dropout_seed'],
        }
        super().__init__(job, snapshot)

    def initialize_experiment(self):
        inventory = super().initialize_experiment()
        if self.adapter['method'] == 'adalora-static-r12':
            # Persist the actual v5 coefficient in the PEFT config metadata.
            self.model.base_model.peft_config['default'].orth_reg_weight = (
                self.scientific_config['orthogonal_coefficient'])
        return inventory

    def orthogonal_penalty(self):
        """Return PEFT's unweighted SVD orthogonal penalty."""
        if self.adapter['method'] != 'adalora-static-r12':
            return torch.zeros((), device=next(self.model.parameters()).device)
        pairs = []
        for name, parameter in self.trainable.items():
            if name.endswith('.lora_A.default'):
                target = name.removesuffix('.lora_A.default')
                pairs.append((parameter, self.trainable[target + '.lora_B.default']))
        if len(pairs) != len(self.adapter['target_modules']):
            raise ValueError('incomplete AdaLoRA A/B target inventory')
        return orthogonal_penalty_reference(pairs)

    def orthogonal_regularizer(self):
        """Match the inherited backend hook with the v5 admitted coefficient."""
        return self.orthogonal_penalty() * self.scientific_config['orthogonal_coefficient']

    def batch_objective(self, records, class_weights=None):
        tokens = self.tokenize(records)
        logits = self.model(**tokens).logits.float()
        labels = torch.tensor(
            [DECISIONS.index(record['expected_decision']) for record in records],
            device=logits.device,
        )
        cross_entropy = torch.nn.functional.cross_entropy(
            logits, labels, weight=class_weights,
        )
        penalty = self.orthogonal_penalty()
        coefficient = self.scientific_config['orthogonal_coefficient']
        self._cross_entropy_for_diagnostics = cross_entropy
        loss = cross_entropy + coefficient * penalty
        if not torch.isfinite(loss):
            raise ValueError('nonfinite static pilot objective')
        return loss, int(tokens['attention_mask'].sum()), {
            'cross_entropy': float(cross_entropy.detach()),
            'orthogonal_penalty': float(penalty.detach()),
            'weighted_orthogonal_penalty': float((coefficient * penalty).detach()),
        }

    def effective_update_records(self):
        """Capture exact low-rank factors and scaling for each target layer."""
        records = {}
        root = self.model.base_model.model
        for target in self.adapter['target_modules']:
            layer = root.get_submodule(target)
            a_module = layer.lora_A['default']
            b_module = layer.lora_B['default']
            a = a_module.weight if hasattr(a_module, 'weight') else a_module
            b = b_module.weight if hasattr(b_module, 'weight') else b_module
            if hasattr(layer, 'lora_E'):
                e = layer.lora_E['default']
                ranknum = float(layer.ranknum['default'].detach().item())
                scaling = float(layer.scaling['default'])
            else:
                e = torch.ones((a.shape[0], 1), dtype=a.dtype, device=a.device)
                ranknum = 1.0
                # Compensate for the AdaLoRA reference epsilon so this helper
                # reports the exact ordinary-LoRA alpha/r scaling.
                scaling = float(layer.scaling['default']) * (1.0 + 1e-5)
            records[target] = {
                'A': a.detach().clone(), 'E': e.detach().clone(), 'B': b.detach().clone(),
                'scaling': scaling, 'ranknum': ranknum,
            }
        return records
