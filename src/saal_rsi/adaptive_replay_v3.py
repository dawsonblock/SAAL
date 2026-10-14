"""Controller-only numerical-contract v3 replay; archived initial head is exact.

The numerical policy changes metric reconstruction, not weights, adapter
restoration or the registered probability tolerance. No old job is relabeled.
"""
from .adaptive_backend_v2 import CudaSafeAdaLoRABackend
from .adaptive_artifacts import validate_checkpoint
from .adaptive_replay import restore_initial_head


class PortableReplayBackendV3(CudaSafeAdaLoRABackend):
    def __init__(self, job, snapshot, initial_checkpoint, initial_state):
        super().__init__(job, snapshot)
        self.admitted_initial = validate_checkpoint(initial_checkpoint, self.adapter, initial_state)

    def initialize_experiment(self):
        restore_initial_head(self.model, self.admitted_initial)
        return super().initialize_experiment()
