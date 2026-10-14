"""Versioned v6 qualification integration with witness-first attempt consumption.

Run on the evaluator/custodian hosts, never the training worker. Independent
administration and truthful sampling remain operational requirements. No API
here grants promotion, runtime capability or external deployment permission.
"""
from pathlib import Path
import hashlib
import math
import re
import platform
from collections import defaultdict
from .spec import sha, canonical, parse_unique_json
from .model_contract import verify, sign
from .validation import key_bytes
from .attempt_authority import digest, validate_bindings, evaluate_once, checked_head, validate_enrollment
from .behavior_screen import screen
from .checkpoint_selection import metrics
from .dataset_registry import validate_manifest, normalize, FIELDS as DATASET_FIELDS
from .family_statistics import compare_families, V2_RULES

RULES = dict(V2_RULES, minimum_complete_pair_accuracy=.9,
             public_screen_required=True, prior_family_loss_allowed=0.)
PINS = {'controller', 'custodian', 'evaluator', 'witness'}


def minimum_zero_error_families(alpha):
    if type(alpha) is not float or not math.isfinite(alpha) or not 0 < alpha <= .025:
        raise ValueError('preregistered witness alpha required')
    return max(RULES['minimum_families'], math.ceil(math.log(alpha)/math.log1p(-RULES['maximum_critical_family_error_upper'])))


def reserve(protocol, mirror, witness_client, enrollment):
    """Do not spend an attempt on a population too small to meet its bound."""
    validate_protocol(protocol)
    enrollment = validate_enrollment(enrollment)
    head = checked_head(witness_client('head', {}), protocol['pins']['witness'], enrollment)
    alpha = enrollment['total_alpha']/2**(head['attempt_count']+1)
    count = len({r['task_family_id'] for r in protocol['holdout_dataset']['records']})
    if count < minimum_zero_error_families(alpha):
        raise PermissionError('insufficient families for the allocated critical error bound, even with zero errors')
    return mirror.reserve(bindings(protocol))


def qualification_identity():
    from .learning_backend import verify_dependencies
    return {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'inference_source_sha256': hashlib.sha256(Path(__file__).with_name('pretrained_inference.py').read_bytes()).hexdigest(),
            'model_dependencies': verify_dependencies(),
            'python': platform.python_version(), 'system': platform.system(),
            'machine': platform.machine(), 'inference_device': 'cpu',
            'protocol': 'saal-pretrained-family-protocol-v1'}


def freeze(bundle, controller_pin, snapshot='terminal'):
    """Authenticate real tensors and recompute screens; no worker PASS is trusted."""
    from .epoch_artifacts import validate_bundle
    from .epoch_contract import validate_job
    from .pretrained_registry import file_hash
    root = Path(bundle)
    if snapshot not in ('initial', 'best', 'terminal'):
        raise ValueError('registered physical snapshot required')
    job = verify(parse_unique_json((root/'signed_training_job.json').read_bytes()), controller_pin)
    validate_job(job, execution=False)
    result = validate_bundle(root, job, controller_pin)
    measures = {}
    for split in ('development', 'regression'):
        records = [r for r in job['dataset']['records'] if r['split'] == split]
        measured = metrics(records, result['snapshots'][snapshot]['predictions'][split], 'modernbert-seqcls-v1')
        measures[split] = {'metrics': measured, 'screen': screen(measured)}
    return {'schema': 'saal-pretrained-frozen-candidate-v1', 'snapshot': snapshot,
            'job_digest': sha(job), 'dataset_digest': sha(job['dataset']),
            'model_spec_digest': sha(job['model_spec']),
            'checkpoint_digest': 'sha256:'+file_hash(root/(snapshot+'.safetensors')),
            'inventory_digest': sha(parse_unique_json((root/'artifact_inventory.json').read_bytes())),
            'worker_identity': job['worker_identity'], 'public_measurements': measures,
            'public_admission_passed': all(v['screen']['passes_public_screen'] for v in measures.values()),
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}


def cross_population_audit(public, holdout):
    """Custodian-side bounded screening; retain unresolved similarity flags."""
    from .learning_dataset import validate_dataset
    validate_dataset(public, 'modernbert-seqcls-v1'); validate_manifest(holdout)
    if {r['split'] for r in holdout['records']} != {'holdout'}:
        raise ValueError('final tasks must be holdout only')
    indexed = defaultdict(set); spans = defaultdict(set)
    for r in public['records']:
        for field in ('record_id', 'task_family_id', 'cluster_id', 'template_id', 'source_digest'):
            indexed[(field, r[field])].add(r['record_id'])
        indexed[('normalized', normalize(r['prompt']))].add(r['record_id'])
        words = re.findall(r'\w+', r['prompt'].casefold())
        for i in range(len(words)-4):
            spans[tuple(words[i:i+5])].add(r['task_family_id'])
    denied = []; flags = []
    for r in holdout['records']:
        for field in ('record_id', 'task_family_id', 'cluster_id', 'template_id', 'source_digest'):
            if (field, r[field]) in indexed:
                denied.append({'record_id': r['record_id'], 'reason': field+'_OVERLAP'})
        if ('normalized', normalize(r['prompt'])) in indexed or ('record_id', r['parent_record_id']) in indexed:
            denied.append({'record_id': r['record_id'], 'reason': 'PUBLIC_PROMPT_OR_PARENT'})
        matches = set(); words = re.findall(r'\w+', r['prompt'].casefold())
        for i in range(len(words)-4):
            matches.update(spans.get(tuple(words[i:i+5]), ()))
        for family in sorted(matches):
            flags.append({'record_id': r['record_id'], 'public_family': family,
                          'reason': 'LEXICAL_SIMILARITY_REQUIRES_SEMANTIC_REVIEW'})
            if len(flags) > 2000:
                raise ValueError('screen exceeds review budget; no truncated approval')
    return {'schema': 'saal-pretrained-cross-population-audit-v1',
            'public_digest': sha(public), 'holdout_digest': sha(holdout),
            'denied': denied, 'review_flags': flags,
            'semantic_independence_established': False,
            'limits': 'Lexical screening cannot prove semantic independence or absence of pretraining exposure. Signed source and sampling review are required.'}


def verify_custody(review, pin, audit):
    body = verify(review, pin)
    fields = {'schema', 'audit_digest', 'decision', 'operational_separation',
              'sampling_justification', 'sign_null_justification', 'pretraining_exposure_review',
              'deployment_review_digest', 'resolutions'}
    if type(body) is not dict or set(body) != fields or body['schema'] != 'saal-pretrained-custody-review-v1':
        raise ValueError('versioned custody review required')
    if body['audit_digest'] != sha(audit) or audit['denied']:
        raise PermissionError('public derivation or stale custody review')
    if body['decision'] != 'approved-independent' or body['operational_separation'] != 'independently-managed':
        raise PermissionError('operationally independent custody required')
    digest(body['deployment_review_digest'])
    for field in ('sampling_justification', 'sign_null_justification', 'pretraining_exposure_review'):
        if type(body[field]) is not str or not 40 <= len(body[field].strip()) <= 4000:
            raise ValueError('sampling, test assumptions and pretraining review required')
    resolutions = body['resolutions']
    if type(resolutions) is not dict or set(resolutions) != {sha(f) for f in audit['review_flags']}:
        raise PermissionError('complete similarity review required')
    if any(type(v) is not str or not 40 <= len(v.strip()) <= 2000 for v in resolutions.values()):
        raise ValueError('reasoned semantic resolutions required')
    return body


def make_protocol(candidate, baseline, public, holdout, custody_review, pins, *, question_digest, selection_digest, replay_review, evaluator_identity=None):
    """Freeze one selected candidate; the custodian retains this private protocol."""
    if type(pins) is not dict or set(pins) != PINS:
        raise ValueError('controller, custodian, evaluator and witness pins required')
    pins = {k: key_bytes(v).hex() for k, v in pins.items()}
    if len(set(pins.values())) != len(PINS):
        raise PermissionError('training cannot impersonate evaluation/custody/witness')
    digest(question_digest); digest(selection_digest)
    for frozen in (candidate, baseline):
        if frozen.get('schema') != 'saal-pretrained-frozen-candidate-v1' or frozen.get('qualification_status') != 'NOT_EVALUATED' or frozen.get('external_deployment_allowed') is not False:
            raise ValueError('frozen unqualified model identity required')
        for field in ('job_digest', 'dataset_digest', 'model_spec_digest', 'checkpoint_digest', 'inventory_digest', 'worker_identity'):
            digest(frozen[field])
    if candidate['dataset_digest'] != sha(public) or baseline['dataset_digest'] != sha(public):
        raise PermissionError('matched public population required')
    if candidate['checkpoint_digest'] == baseline['checkpoint_digest']:
        raise ValueError('baseline and candidate weights must differ')
    reproduction = verify(replay_review, pins['evaluator'])
    replay_fields = {'schema','candidate_digest','baseline_digest','tolerance','probabilities_match','comparisons','scope','external_deployment_allowed'}
    if set(reproduction) != replay_fields or reproduction.get('schema') != 'saal-pretrained-public-replay-v1' or reproduction.get('candidate_digest') != sha(candidate) or reproduction.get('baseline_digest') != sha(baseline) or reproduction.get('tolerance') != 1e-4 or reproduction.get('probabilities_match') is not True or reproduction.get('scope') != 'EVALUATOR_REPRODUCED_PUBLIC_CHECKPOINTS' or reproduction.get('external_deployment_allowed') is not False:
        raise PermissionError('independent checkpoint replay required')
    comparisons = reproduction['comparisons']
    if type(comparisons) is not dict or set(comparisons) != {'candidate','baseline'}:
        raise ValueError('complete independent replay inventory required')
    for by_split in comparisons.values():
        if type(by_split) is not dict or set(by_split) != {'development','regression'}:
            raise ValueError('replay must cover both public populations')
        for split, comparison in by_split.items():
            if type(comparison) is not dict or set(comparison) != {'probabilities_match','decision_labels_match','prediction_digest','records'} or comparison['probabilities_match'] is not True or comparison['decision_labels_match'] is not True or type(comparison['records']) is not int or comparison['records'] != sum(r['split']==split for r in public['records']):
                raise PermissionError('incomplete or failed checkpoint reproduction')
            digest(comparison['prediction_digest'])
    # Recompute the boolean instead of accepting a substituted admission flag.
    if set(candidate['public_measurements']) != {'development', 'regression'} or not all(screen(v['metrics'])['passes_public_screen'] for v in candidate['public_measurements'].values()):
        raise PermissionError('failed public screen cannot spend a protected attempt')
    audit = cross_population_audit(public, holdout)
    verify_custody(custody_review, pins['custodian'], audit)
    if holdout['origin'] != 'custodian-sequestered':
        raise PermissionError('public tasks cannot be final holdout')
    target = qualification_identity() if evaluator_identity is None else evaluator_identity
    reference = qualification_identity()
    if type(target) is not dict or set(target) != set(reference) or any(target[k] != reference[k] for k in ('source_sha256','inference_source_sha256','model_dependencies','inference_device','protocol')):
        raise PermissionError('registered evaluator source/dependency configuration required')
    if type(target['python']) is not str or not re.fullmatch(r'3\.[0-9]{1,2}\.[0-9]{1,3}',target['python']) or target['system'] not in ('Linux','Darwin') or target['machine'] not in ('arm64','aarch64','x86_64'):
        raise ValueError('bounded preregistered evaluator platform identity')
    return {'schema': 'saal-pretrained-family-protocol-v1', 'candidate': candidate,
            'baseline': baseline, 'public_dataset': public, 'holdout_dataset': holdout,
            'custody_review': custody_review, 'audit': audit, 'pins': pins,
            'question_digest': question_digest, 'selection_digest': selection_digest,
            'rules': RULES.copy(), 'evaluator_identity': target, 'replay_review': replay_review,
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}


def validate_protocol(protocol):
    if type(protocol) is not dict or protocol != make_protocol(
            protocol['candidate'], protocol['baseline'], protocol['public_dataset'],
            protocol['holdout_dataset'], protocol['custody_review'], protocol['pins'],
            question_digest=protocol['question_digest'], selection_digest=protocol['selection_digest'], replay_review=protocol['replay_review'], evaluator_identity=protocol['evaluator_identity']):
        raise PermissionError('frozen protocol/runtime/rules substitution')
    return protocol


def bindings(protocol):
    return validate_bindings({'candidate_digest': sha(protocol['candidate']),
        'baseline_digest': sha(protocol['baseline']), 'protocol_digest': sha(protocol),
        'holdout_digest': sha(protocol['holdout_dataset']),
        'sampling_review_digest': sha(protocol['custody_review']),
        'question_digest': protocol['question_digest']})


def measures(protocol, predictions, alpha):
    if type(alpha) is not float or not math.isfinite(alpha) or not 0 < alpha <= .025:
        raise ValueError('witness alpha allocation required')
    if type(predictions) is not dict or set(predictions) != {'holdout', 'regression'}:
        raise ValueError('both protected and retention populations required')
    populations = {'holdout': protocol['holdout_dataset']['records'],
                   'regression': [r for r in protocol['public_dataset']['records'] if r['split'] == 'regression']}
    measured = {}; diagnostics = {}
    for split, rows in populations.items():
        pair = predictions[split]
        if type(pair) is not dict or set(pair) != {'baseline', 'candidate'}:
            raise ValueError('matched baseline/candidate predictions required')
        for outputs in pair.values():
            current = metrics(rows, outputs, 'modernbert-seqcls-v1')
            if current['invalid_outputs']:
                raise ValueError('malformed predictions are failures, not repaired labels')
        diagnostics[split] = metrics(rows, pair['candidate'], 'modernbert-seqcls-v1')
        if split == 'holdout':
            measured[split] = compare_families(rows, [v['decision'] for v in pair['baseline']],
                [v['decision'] for v in pair['candidate']], alpha,
                independence_verified=True, leakage_review_complete=True,
                origin=protocol['holdout_dataset']['origin'])
        else:
            before = metrics(rows, pair['baseline'], 'modernbert-seqcls-v1')
            measured[split] = {'per_family': {family: {'baseline_accuracy': value,
                'candidate_accuracy': diagnostics[split]['family_accuracy'][family]}
                for family, value in before['family_accuracy'].items()},
                'baseline_metrics': before, 'candidate_metrics': diagnostics[split],
                'scope': 'DESCRIPTIVE_PUBLIC_RETENTION_NO_POPULATION_INFERENCE'}
    final = measured['holdout']; retention = measured['regression']
    gates = final['gates'].copy()
    gates.update(complete_counterfactual_pairs=diagnostics['holdout']['complete_pair_accuracy'] is not None and diagnostics['holdout']['complete_pair_accuracy'] >= .9,
                 correct_four_class_coverage=all(v > 0 for v in diagnostics['holdout']['class_accuracy'].values()) and len(diagnostics['holdout']['class_accuracy']) == 4,
                 regression_public_screen=screen(diagnostics['regression'])['passes_public_screen'],
                 prior_capability_retention=all(v['candidate_accuracy'] >= v['baseline_accuracy'] for v in retention['per_family'].values()))
    return {'schema': 'saal-pretrained-family-measures-v1', 'gates': gates,
            'research_qualified': all(gates.values()), 'holdout': final, 'retention': retention,
            'probability_diagnostics': diagnostics, 'external_deployment_allowed': False}


def evaluate(protocol, candidate_bundle, baseline_bundle, reservation, witness_client, enrollment, cache, inference=None):
    """Spent alpha is never refunded, including inference/replay failures."""
    validate_protocol(protocol)
    if protocol['evaluator_identity'] != qualification_identity():
        raise PermissionError('actual evaluator platform differs from preregistration')
    pin = protocol['pins']['controller']
    if freeze(candidate_bundle, pin, protocol['candidate']['snapshot']) != protocol['candidate'] or freeze(baseline_bundle, pin, protocol['baseline']['snapshot']) != protocol['baseline']:
        raise PermissionError('candidate/baseline artifact replacement')
    if inference is None:
        from .pretrained_inference import infer_pair
        inference = lambda rows: infer_pair(candidate_bundle, baseline_bundle, protocol, rows, cache)
    rows = {'holdout': protocol['holdout_dataset']['records'],
            'regression': [r for r in protocol['public_dataset']['records'] if r['split'] == 'regression']}
    def run(alpha):
        outputs = {split: inference(records) for split, records in rows.items()}
        result = measures(protocol, outputs, alpha)
        # Re-check actual bytes after the independent child exits.
        if freeze(candidate_bundle, pin, protocol['candidate']['snapshot']) != protocol['candidate'] or freeze(baseline_bundle, pin, protocol['baseline']['snapshot']) != protocol['baseline']:
            raise PermissionError('artifact changed during inference')
        return {'predictions': outputs, 'metrics': result}
    witnessed = evaluate_once(reservation, bindings(protocol), witness_client,
                             protocol['pins']['witness'], enrollment, run)
    return {'schema': 'saal-pretrained-qualification-report-v1', 'protocol_digest': sha(protocol),
            'consumption': witnessed['consumption'], 'measurement': witnessed['measurement'],
            'evaluator_identity': qualification_identity(),
            'qualification_status': 'RESEARCH_QUALIFIED' if witnessed['measurement']['metrics']['research_qualified'] else 'NOT_QUALIFIED',
            'external_deployment_allowed': False, 'model_activation_allowed': False,
            'limits': 'Pinned roles attribute assertions; operational separation, independently authored sampling and truthful execution need external assurance. No alignment, self-improvement or production authorization claim.'}


def verify_report(envelope, protocol):
    validate_protocol(protocol)
    report = verify(envelope, protocol['pins']['evaluator'])
    fields = {'schema', 'protocol_digest', 'consumption', 'measurement', 'evaluator_identity', 'qualification_status', 'external_deployment_allowed', 'model_activation_allowed', 'limits'}
    if set(report) != fields or report.get('schema') != 'saal-pretrained-qualification-report-v1' or report.get('protocol_digest') != sha(protocol) or report.get('evaluator_identity') != protocol['evaluator_identity'] or report.get('external_deployment_allowed') is not False or report.get('model_activation_allowed') is not False:
        raise PermissionError('report scope/identity substitution')
    consumed = report['consumption']
    if type(consumed) is not dict or set(consumed) != {'event', 'head'}:
        raise ValueError('witness consumption/head pair required')
    head_payload = verify(consumed['head'], protocol['pins']['witness'])
    enrollment = validate_enrollment(head_payload['enrollment'])
    head = checked_head(consumed['head'], protocol['pins']['witness'], enrollment)
    event = verify(report['consumption']['event'], protocol['pins']['witness'])
    if set(event) != {'schema','enrollment_digest','seq','previous','kind','attempt','bindings','alpha','time'} or event.get('schema') != 'saal-attempt-event-v1' or event.get('kind') != 'CONSUMED' or event.get('bindings') != bindings(protocol) or event['enrollment_digest'] != sha(enrollment) or head['head'] != sha(consumed['event']) or head['seq'] != event['seq'] or enrollment['question_digest'] != protocol['question_digest']:
        raise PermissionError('unconsumed or substituted attempt')
    digest(event['previous'])
    if type(event.get('attempt')) is not int or not 1 <= event['attempt'] <= enrollment['max_attempts'] or type(event.get('alpha')) is not float or event['alpha'] != enrollment['total_alpha'] / 2**event['attempt'] or head['attempt_count'] < event['attempt']:
        raise PermissionError('invalid attempt/alpha')
    rebuilt = measures(protocol, report['measurement']['predictions'], event['alpha'])
    if rebuilt != report['measurement']['metrics'] or report['qualification_status'] != ('RESEARCH_QUALIFIED' if rebuilt['research_qualified'] else 'NOT_QUALIFIED'):
        raise PermissionError('forged qualification statistics/status')
    return report
