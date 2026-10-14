"""Controller-only v2 replay, with the exact archived initial head."""
from .adaptive_backend_v2 import CudaSafeAdaLoRABackend
from .adaptive_artifacts import validate_checkpoint
from .adaptive_replay import restore_initial_head


class PortableReplayBackendV2(CudaSafeAdaLoRABackend):
    def __init__(self, job, snapshot, initial_checkpoint, initial_state):
        super().__init__(job, snapshot)
        self.admitted_initial = validate_checkpoint(initial_checkpoint, self.adapter, initial_state)

    def initialize_experiment(self):
        restore_initial_head(self.model, self.admitted_initial)
        return super().initialize_experiment()
