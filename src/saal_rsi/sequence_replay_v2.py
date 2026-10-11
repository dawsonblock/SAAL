"""Full per-record restoration of sequence-v2 bundles in a fresh process."""
import hashlib
import math
from pathlib import Path
from .model_contract import verify
from .sequence_v2 import validate_protocol
from .pretrained_registry import validate_snapshot
from .learning_backend import create_backend, verify_dependencies
from .replay_sampling import batch_plan
from .checkpoint_selection import metrics
from .spec import parse_unique_json, sha


def validate_bundle(root, pin):
    root = Path(root)
    job_envelope = parse_unique_json((root / 'signed_job.json').read_bytes())
    job = verify(job_envelope, pin)
    protocol = verify(parse_unique_json((root / 'signed_protocol.json').read_bytes()), pin)
    validate_protocol(protocol, job)
    expected = {'result.json', 'signed_job.json', 'signed_protocol.json'} | {
        f'{stage}-{branch}.safetensors' for stage in range(len(protocol['groups'])) for branch in ('experimental', 'fresh_control')} | {
        f'{stage}-replay-plan.json' for stage in range(len(protocol['groups']))}
    inventory = parse_unique_json((root / 'inventory.json').read_bytes())
    if set(inventory) != {'schema', 'files'} or inventory['schema'] != 'saal-sequence-inventory-v2' or set(inventory['files']) != expected or {p.name for p in root.iterdir()} != expected | {'inventory.json'}:
        raise ValueError('sequence-v2 exact artifact inventory')
    for name, entry in inventory['files'].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or set(entry) != {'size', 'sha256'} or type(entry['size']) is not int or not 1 <= entry['size'] <= 64_000_000 or path.stat().st_size != entry['size'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('sequence-v2 artifact substitution or bounds')
    records = job['dataset']['records']; seen = []
    for stage, categories in enumerate(protocol['groups']):
        current = [r for r in records if r['split'] == 'train' and r['category'] in categories]
        plan = batch_plan(current, seen, seed=job['seed'], replay_seed=protocol['replay_seed'], stage=stage,
                          steps=protocol['steps_per_stage'], batch_size=job['batch_size'], accumulation=job['accumulation'],
                          replay_ratio=protocol['replay_ratio'], policy=protocol['replay_policy'], capacity=protocol['reservoir_capacity'])
        if parse_unique_json((root / f'{stage}-replay-plan.json').read_bytes()) != plan:
            raise ValueError('replay policy, RNG or example identity substitution')
        seen.extend(current)
    result = parse_unique_json((root / 'result.json').read_bytes())
    if result.get('schema') != 'saal-sequence-result-v2' or result.get('protocol_digest') != sha(protocol) or result.get('worker_identity') != protocol['worker_identity'] or result.get('completed_updates') != protocol['steps_per_stage'] * len(protocol['groups']) * 2 or result.get('qualification_status') != 'NOT_EVALUATED' or result.get('external_deployment_allowed') is not False:
        raise ValueError('sequence-v2 result identity or authority')
    return job, protocol, result


def prediction_equal(actual, expected, kind, tolerance):
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1e-4:
        return False
    if type(actual) is not list or type(expected) is not list or len(actual) != len(expected):
        return False
    if kind != 'modernbert-seqcls-v1':
        return actual == expected
    from .pretrained_contract import DECISIONS
    def valid(output):
        if type(output) is not dict or set(output) != {'decision', 'probabilities'}:
            return False
        values = output['probabilities']
        if type(values) is not list or len(values) != len(DECISIONS):
            return False
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
            return False
        return (abs(sum(values) - 1) < 1e-4 and
                output['decision'] == DECISIONS[max(range(len(values)), key=lambda i: values[i])])
    for a, b in zip(actual, expected):
        if not valid(a) or not valid(b) or a['decision'] != b['decision']:
            return False
        if any(abs(x - y) > tolerance for x, y in zip(a['probabilities'], b['probabilities'])):
            return False
    return True


def replay(root, pin, cache, *, tolerance=1e-4):
    if type(tolerance) is not float or not 0 <= tolerance <= 1e-4:
        raise ValueError('replay numerical tolerance')
    job, protocol, result = validate_bundle(root, pin)
    verify_dependencies()
    spec = job['model_spec']; root = Path(root)
    snapshot = Path(cache) / spec['model_repository'].split('/')[-1] / spec['model_revision']
    validate_snapshot(snapshot, spec['model_repository'], spec['model_revision'])
    backend = create_backend(spec, snapshot, job['seed']).load_pretrained()
    groups = [[r for r in job['dataset']['records'] if r['split'] == 'development' and r['category'] in g] for g in protocol['groups']]
    if len(result['initial_predictions']) != len(groups) or len(result['stage_predictions']) != len(groups): raise ValueError('complete per-record stage evidence required')
    agreement = True; restored = 0; measurements = []
    for group, expected in zip(groups,result['initial_predictions']):
        actual = backend.evaluate(group,job['max_new_tokens'])
        agreement &= prediction_equal(actual,expected,spec['model_kind'],tolerance)
    for stage, evidence in enumerate(result['stage_predictions']):
        if evidence['stage'] != stage or evidence['task_record_ids'] != [[r['record_id'] for r in group] for group in groups] or len(evidence['experimental_predictions']) != len(groups):
            raise ValueError('prediction task/order substitution')
        row=[]
        for k,group in enumerate(groups):
            checkpoint_stage = k if protocol['method'] == 'isolated' and k <= stage else stage
            backend.restore_checkpoint(root/f'{checkpoint_stage}-experimental.safetensors');restored+=1
            actual=backend.evaluate(group,job['max_new_tokens']);agreement &= prediction_equal(actual,evidence['experimental_predictions'][k],spec['model_kind'],tolerance)
            row.append(metrics(group,actual,spec['model_kind'])['accuracy'])
        backend.restore_checkpoint(root/f'{stage}-fresh_control.safetensors');restored+=1
        actual=backend.evaluate(groups[stage],job['max_new_tokens'])
        if evidence['fresh_control_record_ids'] != [r['record_id'] for r in groups[stage]]:raise ValueError('fresh control record substitution')
        agreement &= prediction_equal(actual,evidence['fresh_control_predictions'],spec['model_kind'],tolerance)
        agreement &= row==result['accuracy_matrix'][stage] and metrics(groups[stage],actual,spec['model_kind'])['accuracy']==result['fresh_control_accuracy'][stage]
        measurements.append({'stage':stage,'accuracy_row':row,'fresh_control_predictions':actual})
    return {'schema':'saal-sequence-controller-replay-v2','protocol_digest':sha(protocol),'per_record_agreement':bool(agreement),
            'restoration_operations':restored,'probability_tolerance':tolerance,'measurements':measurements,
            'independent_evaluator':False,'qualification_status':'NOT_EVALUATED','external_deployment_allowed':False}
