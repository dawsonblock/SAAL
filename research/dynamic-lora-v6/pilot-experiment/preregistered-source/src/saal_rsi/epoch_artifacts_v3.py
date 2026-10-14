"""Additional evidence checks; no reinterpretation of historical training jobs.

This validator adds protocol consistency checks to the v2 tensor/prediction
validator. It does not turn worker assertions into independent attestation.
"""
import math
from pathlib import Path
from .epoch_artifacts_v2 import FILES, validate_bundle as validate_previous
from .epoch_artifacts_v2 import metrics_equivalent, restoration_manifest, seal, tensor_inventory
from .checkpoint_selection import paired_family_gain
from .pretrained_registry import file_hash


def validate_evidence(result, job, checkpoint_hashes):
    steps = [0]+list(range(job['checkpoint_interval'], job['steps']+1, job['checkpoint_interval']))
    if steps[-1] != job['steps']:
        steps.append(job['steps'])
    history = result['selection_history']
    if any(type(row.get('step')) is not int for row in history) or [row['step'] for row in history] != steps:
        raise ValueError('registered checkpoint schedule missing or substituted')
    by_step = {row['step']: row['metrics'] for row in history}
    for name, snapshot in result['snapshots'].items():
        if type(snapshot['step']) is not int or snapshot['step'] not in by_step:
            raise ValueError('snapshot step not registered')
        if snapshot['metrics'] != by_step[snapshot['step']]:
            raise ValueError('snapshot/history metric substitution')
    selected = result['snapshots']['best']['step']
    if selected == 0 and checkpoint_hashes['best'] != checkpoint_hashes['initial']:
        raise ValueError('step-zero checkpoint substitution')
    if selected == job['steps'] and checkpoint_hashes['best'] != checkpoint_hashes['terminal']:
        raise ValueError('terminal selected checkpoint substitution')
    for alias in ('initial', 'terminal'):
        if selected == result['snapshots'][alias]['step'] and result['snapshots']['best']['predictions'] != result['snapshots'][alias]['predictions']:
            raise ValueError('same-step prediction substitution')
    runtime = result['runtime']
    spec = job['model_spec']
    if runtime.get('attention') != spec['attention_implementation']:
        raise ValueError('attention configuration substitution')
    if type(runtime.get('vision_used')) is not bool or runtime['vision_used']:
        raise ValueError('vision runtime not admitted')
    for field in ('free_gpu_bytes_preflight', 'total_gpu_bytes'):
        if type(runtime.get(field)) is not int or runtime[field] < 0:
            raise ValueError('GPU memory measurement type')
    if runtime['free_gpu_bytes_preflight'] > runtime['total_gpu_bytes']:
        raise ValueError('GPU memory measurement bounds')
    if spec['compute_backend'] == 'colab':
        if not runtime['python'].startswith('3.13.') or 'Linux' not in runtime['platform']:
            raise ValueError('hash-locked GPU platform mismatch')
    elif runtime.get('device') != 'cpu' or runtime.get('gpu') is not None:
        raise ValueError('local CPU configuration substitution')
    accounting = result['resource_accounting']
    for field in ('steps', 'microbatches', 'sample_exposures', 'unique_training_records', 'tokens',
                  'trainable_parameters', 'total_parameters', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'):
        if type(accounting.get(field)) is not int or accounting[field] < 0:
            raise ValueError('resource accounting type')
    if accounting['microbatches'] != job['steps']*job['accumulation']:
        raise ValueError('microbatch budget substitution')
    if accounting['trainable_parameters'] != sum(v['numel'] for v in result['parameter_inventory'].values()):
        raise ValueError('trainable parameter count substitution')
    if accounting['total_parameters'] < accounting['trainable_parameters']:
        raise ValueError('total parameter count substitution')
    for value in (accounting.get('elapsed_seconds'), result.get('total_worker_seconds')):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= job['max_seconds']:
            raise ValueError('nonfinite or exceeded worker time budget')
    if accounting['elapsed_seconds'] > result['total_worker_seconds']:
        raise ValueError('worker time accounting substitution')
    for name in ('best', 'terminal'):
        # The legacy algorithm iterated family insertion order. Canonical JSON
        # sorts object keys; reconstruct worker order from the signed record list
        # instead of falsely rejecting its original bootstrap diagnostics.
        families = list(dict.fromkeys(r['task_family_id'] for r in job['dataset']['records'] if r['split'] == 'development'))
        def ordered(metrics):
            return dict(metrics, family_accuracy={family: metrics['family_accuracy'][family] for family in families})
        expected = paired_family_gain(ordered(result['snapshots']['initial']['metrics']['development']),
                                      ordered(result['snapshots'][name]['metrics']['development']))
        if result.get('adaptation_gain_'+name) != expected:
            raise ValueError('adaptation gain substitution')
    return result


def validate_bundle(root, job, pin=None):
    result = validate_previous(root, job, pin)
    hashes = {name: file_hash(Path(root)/(name+'.safetensors')) for name in ('initial', 'best', 'terminal')}
    return validate_evidence(result, job, hashes)
