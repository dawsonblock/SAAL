"""Portable controller replay for v4 metric and adapter contracts."""
from .adaptive_backend_v2_v4 import CudaSafeAdaLoRABackend
from .adaptive_artifacts_v4 import (ROLES, file_hash, read_json,
                                   validate_bundle, validate_checkpoint)
from .adaptive_replay import restore_initial_head
from .spec import sha


class PortableReplayBackendV4(CudaSafeAdaLoRABackend):
    def __init__(self, job, snapshot, initial_checkpoint, initial_state):
        super().__init__(job, snapshot)
        self.admitted_initial = validate_checkpoint(initial_checkpoint, self.adapter, initial_state)

    def initialize_experiment(self):
        restore_initial_head(self.model, self.admitted_initial)
        return super().initialize_experiment()


def fresh_local_replay(bundle, job, pin, cache):
    """Reconstruct all v4 snapshots in a new local process after GPU export."""
    from pathlib import Path
    from .adaptive_backend import verify_peft_implementation
    from .epoch_metrics import prediction_equal
    from .learning_backend import verify_dependencies
    from .pretrained_registry import validate_snapshot
    import torch

    bundle = Path(bundle)
    result = validate_bundle(bundle, job, pin)
    versions = verify_dependencies()
    implementation = verify_peft_implementation()
    spec = job['model_spec']
    snapshot = Path(cache)/spec['model_repository'].split('/')[-1]/spec['model_revision']
    validate_snapshot(snapshot, spec['model_repository'], spec['model_revision'])
    torch.set_num_threads(2)
    backend = PortableReplayBackendV4(job, snapshot, bundle/'initial.safetensors',
                                      read_json(bundle/'initial_rank_pattern.json')).load_pretrained()
    if (backend.unadapted_model_hash != result['unadapted_model_hash']
            or backend.initial_frozen_hash != result['frozen_integrity']['initial']):
        raise ValueError('fresh v4 base / initial head / frozen backbone identity mismatch')
    rows = [row for row in job['dataset']['records'] if row['split'] == 'development']
    predictions = {}
    for role in ROLES:
        backend.restore_checkpoint(bundle/(role+'.safetensors'),
                                   read_json(bundle/(role+'_rank_pattern.json')))
        predictions[role] = backend.evaluate(rows)
    verified = all(prediction_equal(predictions[role], result['snapshots'][role]['predictions'],
                                    'modernbert-seqcls-v1', job['replay_tolerance'])
                   for role in ROLES)
    return {'schema': 'saal-adaptive-fresh-replay-v4', 'job_digest': sha(job),
            'inventory_sha256': file_hash(bundle/'artifact_inventory.json'),
            'numerical_metric_policy_digest': job['numerical_metric_policy_digest'],
            'worker_identity': job['worker_identity'], 'checkpoints': predictions,
            'verified': verified, 'tolerance': job['replay_tolerance'],
            'runtime': {'versions': versions, 'peft_implementation': implementation,
                        'device': str(next(backend.model.parameters()).device),
                        'independent_administration': False,
                        'qualification_status': 'NOT_EVALUATED'}}
