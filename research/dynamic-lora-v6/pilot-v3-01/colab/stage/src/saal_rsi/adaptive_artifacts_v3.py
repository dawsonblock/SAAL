"""V3 exact deterministic metrics; v1/v2 artifact semantics stay frozen."""
"""Exact versioned SVD/mask artifacts; integrity is distinct from qualification."""
import hashlib
import math
from pathlib import Path

from .adaptive_backend import tensor_inventory
from .adaptive_contract_v3 import validate_job, RANKNUM_POLICY
from .numerical_metrics_v1 import POLICY, POLICY_DIGEST, validate_policy
from .adaptive_rank_controller import budget_schedule, initial_state, validate_state
from .numerical_metrics_v1 import diagnostics
from .epoch_metrics import prediction_equal
from .epoch_sampler import sample_ids
from .model_contract import verify
from .spec import canonical, parse_unique_json, sha

ROLES = ('initial', 'best', 'terminal')
FILES = {'signed_training_job.json', 'model_spec.json', 'dataset_manifest.json',
         'snapshot_manifest.json', 'adapter_config.json', 'rank_schedule.json',
         'rank_history.jsonl', 'training_metrics.jsonl', 'resource_accounting.json',
         'sampler_manifest.json', 'checkpoint_selection.json', 'result.json'} | {
         r+'.safetensors' for r in ROLES} | {r+'_rank_pattern.json' for r in ROLES}


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 40_000_000:
        raise ValueError('bounded regular JSON artifact')
    return parse_unique_json(path.read_bytes())


def seal(root):
    root = Path(root)
    if {p.name for p in root.iterdir()} != FILES:
        raise ValueError('incomplete / extra artifact files before sealing')
    inventory = {'schema': 'saal-adaptive-artifact-inventory-v3',
                 'files': {n: {'size': (root/n).stat().st_size, 'sha256': file_hash(root/n)} for n in sorted(FILES)}}
    (root/'artifact_inventory.json').write_bytes(canonical(inventory))
    return inventory


def validate_checkpoint(path, adapter, state):
    import torch
    from safetensors.torch import load_file
    validate_state(state, adapter)
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 64_000_000:
        raise ValueError('checkpoint bounds')
    values = load_file(str(path), device='cpu')
    expected = tensor_inventory(adapter)
    if set(values) != set(expected):
        raise ValueError('exact adapter / head / singular tensor inventory')
    for n, t in values.items():
        if list(t.shape) != expected[n] or t.dtype != torch.float32 or not torch.isfinite(t).all():
            raise ValueError('checkpoint shape / dtype / finite values')
    if adapter['method'] == 'adalora-12-to-8':
        for target, mask in state['masks'].items():
            prefix = 'base_model.model.'+target
            if values[prefix+'.ranknum.default'].item() != float(adapter['init_r']):
                raise ValueError('ranknum scaling changed')
            e = values[prefix+'.lora_E.default']
            if any(e[j].item() != 0 for j, enabled in enumerate(mask) if not enabled):
                raise ValueError('masked singular value is nonzero')
    return values


def rank_history(root, job):
    path = Path(root)/'rank_history.jsonl'
    if path.stat().st_size > 64_000_000:
        raise ValueError('rank history bound')
    rows = [parse_unique_json(line) for line in path.read_bytes().splitlines()]
    if len(rows) != job['steps']:
        raise ValueError('one rank event per completed update required')
    adapter = job['adapter']
    previous = 'sha256:'+'0'*64
    state = initial_state(adapter)
    applied = adapter['init_r']*len(adapter['target_modules'])
    dynamic = adapter['method'] == 'adalora-12-to-8'
    fields = {'schema', 'completed_optimizer_step', 'previous_hash', 'allocated_this_update',
              'scheduled_budget', 'applied_budget', 'state', 'active_rank_by_layer',
              'aggregate_active_rank', 'nonzero_singular_by_layer', 'importance_scores',
              'gradients', 'elapsed_seconds', 'gpu_memory', 'event_hash'}
    fields.add('adapter_gradient_norms')
    allocations = 0
    previous_elapsed = -1
    for step, row in enumerate(rows, 1):
        if type(row) is not dict or set(row) != fields or row['schema'] != 'saal-adaptive-rank-event-v1':
            raise ValueError('rank event schema')
        digest = sha({k: v for k, v in row.items() if k != 'event_hash'})
        if type(row['completed_optimizer_step']) is not int or row['completed_optimizer_step'] != step or row['previous_hash'] != previous or row['event_hash'] != digest:
            raise ValueError('manipulated / replayed / reordered rank history')
        previous = digest
        scheduled, due = budget_schedule(adapter, step)
        final_start = adapter['schedule']['total_step']-adapter['schedule']['tfinal'] if dynamic else -1
        allocated = dynamic and (step == final_start or adapter['schedule']['tinit'] < step < final_start and due)
        if row['allocated_this_update'] is not allocated or type(row['scheduled_budget']) is not int or row['scheduled_budget'] != scheduled:
            raise ValueError('allocation schedule omission / deviation')
        current = validate_state(row['state'], adapter)
        if current['step'] != step:
            raise ValueError('rank state update identity')
        if not allocated and current['masks'] != state['masks']:
            raise ValueError('mask changed outside allocation event')
        if allocated:
            applied = scheduled
            allocations += 1
        active = {t: sum(m) for t, m in current['masks'].items()}
        if (canonical(row['active_rank_by_layer']) != canonical(active) or type(row['aggregate_active_rank']) is not int
                or row['aggregate_active_rank'] != sum(active.values()) or type(row['applied_budget']) is not int or row['applied_budget'] != applied
                or sum(active.values()) > applied):
            raise ValueError('rank budget / allocation evidence mismatch')
        scores = row['importance_scores']
        if type(scores) is not dict or dynamic and set(scores) != set(adapter['target_modules']) or not dynamic and scores:
            raise ValueError('importance target inventory')
        for values in scores.values():
            if type(values) is not list or len(values) != adapter['init_r'] or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
                raise ValueError('invalid triplet importance')
        nonzero = row['nonzero_singular_by_layer']
        if dynamic:
            if type(nonzero) is not dict or set(nonzero) != set(active) or any(type(nonzero[t]) is not int or not 0 <= nonzero[t] <= active[t] for t in active):
                raise ValueError('nonzero singular / active rank mismatch')
        elif nonzero is not None:
            raise ValueError('fixed LoRA has no SVD singular counts')
        gradients = row['gradients']
        layer_norms = row['adapter_gradient_norms']
        if (type(layer_norms) is not dict or set(layer_norms) != set(adapter['target_modules'])
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in layer_norms.values())):
            raise ValueError('finite per-layer adapter gradient evidence required')
        expected_gradients = len(tensor_inventory(adapter))-(len(active) if dynamic else 0)
        if (type(gradients) is not dict or set(gradients) != {'finite', 'norm', 'gradient_tensors'}
                or gradients['finite'] is not True or type(gradients['norm']) not in (int, float)
                or not math.isfinite(gradients['norm']) or gradients['norm'] < 0
                or type(gradients['gradient_tensors']) is not int or gradients['gradient_tensors'] != expected_gradients):
            raise ValueError('incomplete gradient evidence')
        elapsed = row['elapsed_seconds']
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not previous_elapsed <= elapsed <= job['max_seconds']:
            raise ValueError('rank event time budget')
        previous_elapsed = elapsed
        memory = row['gpu_memory']
        if type(memory) is not dict or set(memory) != {'allocated_bytes', 'reserved_bytes'} or any(type(v) is not int or v < 0 for v in memory.values()) or memory['allocated_bytes'] > memory['reserved_bytes']:
            raise ValueError('GPU memory evidence')
        state = current
    if dynamic and (allocations == 0 or state['masks'] == initial_state(adapter)['masks']
                    or not 0 < sum(sum(m) for m in state['masks'].values()) <= adapter['target_r']*len(state['masks'])
                    or sum(rows[-1]['nonzero_singular_by_layer'].values()) == 0):
        raise ValueError('adaptive mode never performed a valid rank reduction')
    return rows, {'completed_optimizer_updates': job['steps'], 'allocation_events': allocations,
                  'rank_history_digest': previous, 'stored_rank_per_layer': adapter['init_r'],
                  'active_rank_total': sum(sum(m) for m in state['masks'].values())}


def validate_bundle(root, job, pin, *, require_replay=False):
    root = Path(root)
    validate_job(job)
    actual = {p.name for p in root.iterdir()}
    allowed = FILES | {'artifact_inventory.json'}
    if (actual != allowed and actual != allowed | {'fresh_process_replay.json'}) or root.is_symlink():
        raise ValueError('exact adaptive artifact inventory')
    inventory = read_json(root/'artifact_inventory.json')
    if set(inventory) != {'schema', 'files'} or inventory['schema'] != 'saal-adaptive-artifact-inventory-v3' or set(inventory['files']) != FILES:
        raise ValueError('artifact inventory schema')
    total = 0
    for n, record in inventory['files'].items():
        p = root/n
        if (p.is_symlink() or not p.is_file() or type(record) is not dict or set(record) != {'size', 'sha256'}
                or type(record['size']) is not int or not 1 <= record['size'] <= 96_000_000
                or p.stat().st_size != record['size'] or file_hash(p) != record['sha256']):
            raise ValueError('artifact bytes / bounds: '+n)
        total += record['size']
    if total > 250_000_000:
        raise ValueError('aggregate artifact bound')
    if verify(read_json(root/'signed_training_job.json'), pin) != job:
        raise ValueError('signed job substitution')
    for name, value in [('model_spec.json', job['model_spec']), ('dataset_manifest.json', job['dataset']),
                        ('adapter_config.json', job['adapter']),
                        ('rank_schedule.json', {'schema': 'saal-adaptive-rank-schedule-v1', 'schedule': job['adapter']['schedule']})]:
        if canonical(read_json(root/name)) != canonical(value):
            raise ValueError('admitted identity mismatch: '+name)
    result = read_json(root/'result.json')
    validate_policy(result.get('numerical_metric_policy'))
    if result.get('numerical_metric_policy_digest') != POLICY_DIGEST:
        raise ValueError('result numerical metric policy substitution')
    if type(result.get('ranknum_normalized')) is not int or not 0 <= result['ranknum_normalized'] <= 22:
        raise ValueError('ranknum initialization normalization evidence')
    if job['adapter']['method'] != 'adalora-12-to-8' and result['ranknum_normalized'] != 0:
        raise ValueError('fixed adapter cannot normalize ranknum')
    if (result.get('schema') != 'saal-adaptive-result-v3' or result.get('ranknum_policy') != RANKNUM_POLICY or result.get('job_digest') != sha(job)
            or result.get('worker_identity') != job['worker_identity'] or result.get('adapter_method') != job['adapter']['method']
            or result.get('qualification_status') != 'NOT_EVALUATED' or result.get('external_deployment_allowed') is not False
            or result.get('real_parameter_updates') is not True):
        raise ValueError('result identity / authority / update evidence')
    from .pretrained_contract import registry_path, entry, VERSIONS
    from .adaptive_contract import PEFT_IMPLEMENTATION
    spec = job['model_spec']
    upstream = entry(spec['model_repository'], spec['model_revision'])
    expected_snapshot = {'repository': spec['model_repository'], 'revision': spec['model_revision'],
                         'files_digest': upstream['model_files_digest'],
                         'tensor_inventory': parse_unique_json(registry_path().with_name('tensor_inventories.json').read_bytes())[spec['model_repository']]}
    if canonical(read_json(root/'snapshot_manifest.json')) != canonical(expected_snapshot):
        raise ValueError('upstream snapshot / tensor inventory substitution')
    import re
    if type(result.get('unadapted_model_hash')) is not str or not re.fullmatch('[0-9a-f]{64}', result['unadapted_model_hash']):
        raise ValueError('unadapted model / head identity')
    runtime = result.get('runtime', {})
    if (runtime.get('versions') != VERSIONS or runtime.get('peft_implementation') != PEFT_IMPLEMENTATION
            or runtime.get('attention') != spec['attention_implementation']
            or runtime.get('deterministic_algorithms') is not True or runtime.get('matmul_tf32') is not False):
        raise ValueError('runtime dependency / attention identity')
    if spec['compute_backend'] == 'colab':
        if runtime.get('device') != 'cuda:0' or type(runtime.get('gpu')) is not str or 'T4' not in runtime['gpu'] or runtime.get('dependency_hash_lock_enforced') is not True:
            raise ValueError('admitted T4 / hash lock not recorded')
    if type(result.get('total_worker_seconds')) not in (int, float) or not math.isfinite(result['total_worker_seconds']) or not 0 < result['total_worker_seconds'] <= job['max_seconds']:
        raise ValueError('end-to-end worker deadline')
    preflight = result.get('preflight', {})
    if (preflight.get('forward_backward_completed') is not True or preflight.get('weights_unchanged') is not True
            or type(preflight.get('optimizer_updates')) is not int or preflight['optimizer_updates'] != 0
            or preflight.get('sample_exposures_without_updates') != job['batch_size']
            or type(preflight.get('tokens')) is not int or preflight['tokens'] < job['batch_size']):
        raise ValueError('forward/backward admission probe evidence')
    rows, summary = rank_history(root, job)
    if canonical(result.get('allocation')) != canonical(summary):
        raise ValueError('rank completion summary')
    training = [r for r in job['dataset']['records'] if r['split'] == 'train']
    expected_stream = sample_ids(training, 'complete-balanced-epochs-v1', job['sampler_seed'], job['steps']*job['batch_size']*job['accumulation'])
    if read_json(root/'sampler_manifest.json') != {'schema': 'saal-adaptive-sample-stream-v1', 'record_ids': expected_stream,
                                                  'seed': job['sampler_seed'], 'sample_exposures': len(expected_stream), 'job_digest': sha(job)}:
        raise ValueError('sample stream / exposure mismatch')
    resources = result.get('resource_accounting')
    if read_json(root/'resource_accounting.json') != resources:
        raise ValueError('resource accounting substitution')
    for name, value in [('completed_optimizer_updates', job['steps']), ('microbatches', job['steps']*job['accumulation']), ('sample_exposures', len(expected_stream))]:
        if type(resources.get(name)) is not int or resources[name] != value:
            raise ValueError('completed update / microbatch / sample budget')
    elapsed = resources.get('elapsed_seconds')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not rows[-1]['elapsed_seconds'] <= elapsed <= job['max_seconds']:
        raise ValueError('cumulative training time accounting')
    parameter_count = sum(math.prod(shape) for name, shape in tensor_inventory(job['adapter']).items() if '.ranknum.' not in name)
    if type(resources.get('trainable_parameters')) is not int or resources['trainable_parameters'] != parameter_count:
        raise ValueError('trainable / stored parameter accounting')
    for field in ('tokens', 'total_parameters', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'):
        if type(resources.get(field)) is not int or resources[field] < 0:
            raise ValueError('bounded resource measurements')
    if resources['tokens'] < resources['sample_exposures'] or resources['total_parameters'] < parameter_count or resources['peak_gpu_allocated_bytes'] > resources['peak_gpu_reserved_bytes']:
        raise ValueError('resource measurements inconsistent')
    frozen = result.get('frozen_integrity', {})
    if (frozen.get('unchanged') is not True or frozen.get('initial') != frozen.get('final')
            or type(frozen.get('initial')) is not str or not re.fullmatch('[0-9a-f]{64}', frozen['initial'])):
        raise ValueError('frozen backbone modification')
    if result.get('same_runtime_replay') != {r: True for r in ROLES}:
        raise ValueError('all three snapshots must restore')
    development = [r for r in job['dataset']['records'] if r['split'] == 'development']
    selection = read_json(root/'checkpoint_selection.json')
    expected_steps = [0]+[s for s in range(1, job['steps']+1) if s % job['checkpoint_interval'] == 0 or s == job['steps']]
    history = selection.get('history', [])
    if (selection.get('policy') != job['selection_policy'] or selection.get('data_scope') != 'PUBLIC_DEVELOPMENT_ONLY'
            or any(type(m.get('step')) is not int for m in history) or [m['step'] for m in history] != expected_steps):
        raise ValueError('development selection schedule')
    from .adaptive_training_v3 import selection_key
    best = history[0]
    for m in history[1:]:
        if selection_key(m['metrics']) > selection_key(best['metrics']):
            best = m
    snapshots = result.get('snapshots')
    if type(snapshots) is not dict or set(snapshots) != set(ROLES):
        raise ValueError('initial/best/terminal snapshots required')
    if snapshots['initial']['step'] != 0 or snapshots['terminal']['step'] != job['steps'] or snapshots['best']['step'] != best['step'] or selection.get('best_step') != best['step']:
        raise ValueError('checkpoint selection / terminal update identity')
    for role in ROLES:
        evidence = snapshots[role]
        if (set(evidence) != {'step', 'predictions', 'metrics'} or type(evidence['step']) is not int
                or canonical(evidence['metrics']) != canonical(diagnostics(development, evidence['predictions']))):
            raise ValueError('snapshot predictions / metrics consistency')
        if evidence['metrics'] != next(m['metrics'] for m in history if m['step'] == evidence['step']):
            raise ValueError('snapshot / selection-history substitution')
        state = read_json(root/(role+'_rank_pattern.json'))
        expected_state = initial_state(job['adapter']) if evidence['step'] == 0 else rows[evidence['step']-1]['state']
        if state != expected_state:
            raise ValueError('checkpoint rank pattern / history substitution')
        checkpoint = validate_checkpoint(root/(role+'.safetensors'), job['adapter'], state)
        if job['adapter']['method'] == 'adalora-12-to-8' and evidence['step']:
            import torch
            actual_nonzero = {t: int(torch.count_nonzero(checkpoint['base_model.model.'+t+'.lora_E.default'])) for t in job['adapter']['target_modules']}
            if actual_nonzero != rows[evidence['step']-1]['nonzero_singular_by_layer']:
                raise ValueError('checkpoint / rank history singular count substitution')
    if inventory['files']['initial.safetensors']['sha256'] == inventory['files']['terminal.safetensors']['sha256']:
        raise ValueError('claimed updates but terminal tensors equal initial tensors')
    for a in ROLES:
        for b in ROLES:
            if snapshots[a]['step'] == snapshots[b]['step'] and (snapshots[a] != snapshots[b] or inventory['files'][a+'.safetensors']['sha256'] != inventory['files'][b+'.safetensors']['sha256']):
                raise ValueError('same-step checkpoint substitution')
    if require_replay:
        replay = read_json(root/'fresh_process_replay.json')
        if (replay.get('schema') != 'saal-adaptive-fresh-replay-v3' or replay.get('job_digest') != sha(job)
                or replay.get('numerical_metric_policy_digest') != POLICY_DIGEST
                or replay.get('inventory_sha256') != file_hash(root/'artifact_inventory.json')
                or replay.get('tolerance') != job['replay_tolerance'] or replay.get('verified') is not True
                or set(replay.get('checkpoints', {})) != set(ROLES)):
            raise ValueError('fresh replay identity / completeness')
        for role in ROLES:
            if not prediction_equal(replay['checkpoints'][role], snapshots[role]['predictions'], 'modernbert-seqcls-v1', job['replay_tolerance']):
                raise ValueError('independent restoration prediction mismatch: '+role)
    return result

