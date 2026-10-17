"""PEFT 0.21.2 ModernBERT adapters with exact tensor inventories.

Keep full allocated SVD tensors and ranknum. Stock compact AdaLoRA exports
are deliberately not used: masking is not physical tensor-size reduction.
"""
import hashlib
from pathlib import Path

from .modernbert_backend import ModernBERTBackend
from .adaptive_contract import PEFT_IMPLEMENTATION
from .adaptive_contract_v4 import validate_adapter


def verify_peft_implementation():
    import importlib.metadata
    import peft
    if importlib.metadata.version('peft') != '0.21.2':
        raise ValueError('AdaLoRA requires exact PEFT 0.21.2')
    root = Path(peft.__file__).parent/'tuners/adalora'
    actual = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in PEFT_IMPLEMENTATION}
    if actual != PEFT_IMPLEMENTATION:
        raise ValueError('PEFT AdaLoRA implementation bytes differ')
    return actual


def tensor_inventory(adapter):
    dynamic = adapter['method'] in ('adalora-12-to-8', 'adalora-static-r12')
    rank = adapter['init_r']
    result = {}
    for target in adapter['target_modules']:
        prefix = 'base_model.model.'+target
        if dynamic:
            result[prefix+'.lora_A.default'] = [rank, 768]
            result[prefix+'.lora_B.default'] = [2304, rank]
            result[prefix+'.lora_E.default'] = [rank, 1]
            result[prefix+'.ranknum.default'] = [1]
        else:
            result[prefix+'.lora_A.default.weight'] = [rank, 768]
            result[prefix+'.lora_B.default.weight'] = [2304, rank]
    for n, shape in {'head.dense.weight': [768, 768], 'head.norm.weight': [768],
                     'classifier.weight': [4, 768], 'classifier.bias': [4]}.items():
        module, suffix = n.split('.', 1)
        result['base_model.model.'+module+'.modules_to_save.default.'+suffix] = shape
    return result


class AdaptiveModernBERTBackend(ModernBERTBackend):
    def __init__(self, job, snapshot):
        self.job = job
        self.adapter = validate_adapter(job['adapter'], job['steps'])
        super().__init__(job['model_spec'], snapshot, job['seed'])

    def initialize_experiment(self):
        import torch
        from peft import get_peft_model
        verify_peft_implementation()
        self.unadapted_model_hash = self.tensor_hash(self.model.named_parameters())
        modules = dict(self.model.named_modules())
        for target in self.adapter['target_modules']:
            layer = modules.get(target)
            if not isinstance(layer, torch.nn.Linear) or (layer.in_features, layer.out_features) != (768, 2304):
                raise ValueError('ModernBERT target dimensions / identity')
        for p in self.model.parameters():
            p.requires_grad_(False)
        config = build_adapter_config(self.adapter)
        self.model = get_peft_model(self.model, config, autocast_adapter_dtype=True)
        self.trainable = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        for p in self.trainable.values():
            p.data = p.data.float()
        self.ranknum = {n: p for n, p in self.model.named_parameters() if '.ranknum.default' in n}
        expected = tensor_inventory(self.adapter)
        if {n: list(p.shape) for n, p in (self.trainable | self.ranknum).items()} != expected:
            raise ValueError('exact native trainable and scaling tensor inventory')
        if any(p.requires_grad for p in self.ranknum.values()):
            raise ValueError('ranknum must be immutable')
        self.initial_frozen_hash = self.frozen_hash()
        self.initial_trainable_hash = self.trainable_hash()
        # Wrapping ranks consume different random draws. Reset the training
        # stream, independently of the Python sampler and head initialization.
        torch.manual_seed(self.job['dropout_seed'])
        return self.parameter_inventory()

    def orthogonal_regularizer(self):
        import torch
        if self.adapter['method'] not in ('adalora-12-to-8', 'adalora-static-r12'):
            return torch.zeros((), device=next(self.model.parameters()).device)
        penalties = []
        for n, p in self.trainable.items():
            if n.endswith('.lora_A.default') or n.endswith('.lora_B.default'):
                cov = p @ p.T if '.lora_A.' in n else p.T @ p
                penalties.append(torch.linalg.matrix_norm(cov-torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype), ord='fro'))
        if not penalties:
            raise ValueError('missing orthogonal adapter parameters')
        return torch.stack(penalties).mean()*self.adapter['schedule']['orth_reg_weight']

    def batch_objective(self, records, class_weights=None):
        import torch
        from .pretrained_contract import DECISIONS
        tokens = self.tokenize(records)
        # Do not pass labels through PEFT and then add the regularizer again.
        logits = self.model(**tokens).logits.float()
        labels = torch.tensor([DECISIONS.index(r['expected_decision']) for r in records], device=logits.device)
        ce = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
        reg = self.orthogonal_regularizer()
        return ce+reg, int(tokens['attention_mask'].sum()), {'cross_entropy': float(ce.detach()), 'orthogonal_penalty': float(reg.detach())}

    def checkpoint_tensors(self):
        return self.trainable | self.ranknum

    def export_checkpoint(self, path):
        from safetensors.torch import save_file
        save_file({n: p.detach().cpu().contiguous() for n, p in self.checkpoint_tensors().items()}, str(path))

    def restore_checkpoint(self, path, state):
        from .adaptive_artifacts_v4 import validate_checkpoint
        from .adaptive_rank_controller_v4 import apply_state
        import torch
        tensors = validate_checkpoint(path, self.adapter, state)
        # Validate complete inventory and masks before any live mutation.
        with torch.no_grad():
            for n, p in self.checkpoint_tensors().items():
                p.copy_(tensors[n].to(p.device))
        apply_state(self.model, self.adapter, state)


def build_adapter_config(adapter):
    """Build only the v4-admitted PEFT configuration for this adapter."""
    from .adaptive_contract_v4 import validate_adapter
    from peft import AdaLoraConfig, LoraConfig
    schedule=adapter.get('schedule')
    total_steps=schedule.get('total_step',1000) if isinstance(schedule,dict) else 1000
    validate_adapter(adapter,total_steps)
    args = dict(lora_alpha=adapter['lora_alpha'], lora_dropout=adapter['dropout'],
                target_modules=adapter['target_modules'], modules_to_save=['head', 'classifier'],
                bias='none')
    if adapter['method'] in ('adalora-12-to-8', 'adalora-static-r12'):
        return AdaLoraConfig(init_r=12, target_r=adapter['target_r'],
                             **adapter['schedule'], **args)
    return LoraConfig(r=adapter['init_r'], **args)
