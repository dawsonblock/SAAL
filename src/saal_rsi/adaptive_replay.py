"""Controller replay restores the admitted initial head before hashing it.

Seeded initialization is not a cross-platform serialization format. Native ARM
and x86 initialization can differ by one ULP. The archived initial head already
contains its exact admitted bytes; restoring them does not relax any hash or
prediction tolerance. This controller-only class does not change training jobs.
"""
from .adaptive_backend import AdaptiveModernBERTBackend
from .adaptive_artifacts import validate_checkpoint

HEAD_NAMES = ('head.dense.weight', 'head.norm.weight', 'classifier.weight', 'classifier.bias')


def restore_initial_head(model, tensors):
    import torch
    parameters = dict(model.named_parameters())
    selected = []
    for name in HEAD_NAMES:
        module, suffix = name.split('.', 1)
        key = 'base_model.model.'+module+'.modules_to_save.default.'+suffix
        if name not in parameters or key not in tensors:
            raise ValueError('complete admitted initial head required')
        p, value = parameters[name], tensors[key]
        if p.shape != value.shape or p.dtype != value.dtype or not torch.isfinite(value).all():
            raise ValueError('initial head shape / dtype / finite values')
        selected.append((p, value))
    # Validate the whole head before mutating any live parameter.
    with torch.no_grad():
        for p, value in selected:
            p.copy_(value.to(p.device))


class PortableReplayBackend(AdaptiveModernBERTBackend):
    def __init__(self, job, snapshot, initial_checkpoint, initial_state):
        super().__init__(job, snapshot)
        self.admitted_initial = validate_checkpoint(initial_checkpoint, self.adapter, initial_state)

    def initialize_experiment(self):
        restore_initial_head(self.model, self.admitted_initial)
        return super().initialize_experiment()
