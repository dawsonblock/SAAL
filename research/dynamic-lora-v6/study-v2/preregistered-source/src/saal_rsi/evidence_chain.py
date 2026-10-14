"""Typed predecessor verification. A valid chain is evidence, never a capability.

Pretrained activation still has no approved path. Existing promotion.py is the
only activation authority and rejects these records. Passing this verifier must
not be wired directly to a tool executor.
"""
from .attempt_authority import BINDINGS, digest, validate_bindings
from .model_contract import verify
from .spec import sha

STAGES = ('CandidateManifest', 'BuildManifest', 'EvaluationBundle',
          'QualificationRecord', 'PromotionDecision', 'RuntimeManifest')
ROLES = ('controller', 'controller', 'evaluator', 'evaluator', 'promotion', 'executor')
IDENTITY = BINDINGS | {'model_spec_digest', 'source_digest', 'dependency_digest',
                      'dataset_digest', 'adapter_inventory_digest'}
GATES = {'independent_custody', 'sampling_review', 'statistical_improvement',
         'utility', 'critical_error_bound', 'retention', 'artifact_replay'}


def verify_chain(chain, pins, expected_identity, *, now, consumption=None):
    if type(now) is not int or now < 0:
        raise ValueError('trusted time required')
    if type(chain) is not list or len(chain) != 6 or set(pins) != {'controller', 'custodian', 'evaluator', 'promotion', 'executor'}:
        raise ValueError('complete authority chain and distinct role pins required')
    if len(set(pins.values())) != 5:
        raise PermissionError('training, custody, evaluation and activation keys must differ')
    if type(expected_identity) is not dict or set(expected_identity) != IDENTITY:
        raise ValueError('independently verified artifact/runtime identity required')
    for value in expected_identity.values():
        digest(value)
    previous = None
    bodies = []
    for stage, role, envelope in zip(STAGES, ROLES, chain):
        body = verify(envelope, pins[role])
        if type(body) is not dict or set(body) != {'schema', 'stage', 'predecessor', 'identity', 'details'} or body['schema'] != 'saal-artifact-authority-chain-v1' or body['stage'] != stage or body['predecessor'] != previous or body['identity'] != expected_identity:
            raise PermissionError('missing, substituted, reordered or incompatible predecessor')
        if type(body['details']) is not dict:
            raise ValueError('stage details shape')
        previous = sha(envelope)
        bodies.append(body)
    candidate, build, evaluation, qualification, promotion, runtime = [b['details'] for b in bodies]
    if set(candidate) != {'training_job_digest', 'parent_checkpoint_digest', 'status'} or candidate['status'] != 'NOT_EVALUATED':
        raise ValueError('candidate must begin unqualified')
    digest(candidate['training_job_digest']); digest(candidate['parent_checkpoint_digest'])
    if set(build) != {'artifact_inventory_digest', 'runtime_identity_digest', 'replay_digest', 'attestation_scope'} or build['attestation_scope'] != 'CONTROLLER_VERIFIED_ARTIFACTS_NOT_HOST_ATTESTATION':
        raise ValueError('build evidence scope')
    for field in ('artifact_inventory_digest', 'runtime_identity_digest', 'replay_digest'):
        digest(build[field])
    if set(evaluation) != {'consumption_digest', 'metrics_digest', 'protocol_version'} or evaluation['protocol_version'] != 'independent-family-v1':
        raise ValueError('frozen evaluation protocol')
    digest(evaluation['metrics_digest'])
    if consumption is None or evaluation['consumption_digest'] != sha(consumption):
        raise PermissionError('one-use witnessed consumption required')
    event = verify(consumption, pins['custodian'])
    bindings = {k: expected_identity[k] for k in BINDINGS}
    validate_bindings(bindings)
    if set(event) != {'schema','kind','bindings','seq','attempt','previous','enrollment_digest','alpha','time'} or event.get('schema') != 'saal-attempt-event-v1' or event.get('kind') != 'CONSUMED' or event.get('bindings') != bindings:
        raise PermissionError('consumption candidate/dataset/protocol substitution')
    digest(event['previous']); digest(event['enrollment_digest'])
    if type(event['attempt']) is not int or not 1 <= event['attempt'] <= 30 or type(event['seq']) is not int or event['seq'] < event['attempt'] + 1 or type(event['alpha']) is not float or not 0 < event['alpha'] <= .05 / 2 ** event['attempt'] or type(event['time']) is not int or not now - 7 * 86400 <= event['time'] <= now:
        raise PermissionError('stale or malformed evaluation consumption')
    if set(qualification) != {'evaluation_digest', 'gates', 'research_qualified', 'scope'} or qualification['evaluation_digest'] != sha(chain[2]) or qualification['scope'] != 'BOUNDED_RESEARCH_ONLY':
        raise ValueError('qualification predecessor and scope')
    gates = qualification['gates']
    if type(gates) is not dict or set(gates) != GATES or any(type(v) is not bool for v in gates.values()) or type(qualification['research_qualified']) is not bool or qualification['research_qualified'] != all(gates.values()):
        raise PermissionError('failed or missing gate cannot become qualification')
    if set(promotion) != {'qualification_digest', 'authorized', 'scope', 'expires_at'} or promotion['qualification_digest'] != sha(chain[3]) or promotion['scope'] != 'SIMULATION_ONLY' or type(promotion['authorized']) is not bool or type(promotion['expires_at']) is not int or promotion['expires_at'] < 0:
        raise ValueError('promotion binding and simulation scope')
    if promotion['authorized'] and (not qualification['research_qualified'] or not now < promotion['expires_at'] <= now + 86400):
        raise PermissionError('failed qualification, stale or excessive approval lifetime')
    if set(runtime) != {'promotion_digest', 'enabled', 'scope', 'runtime_config_digest'} or runtime['promotion_digest'] != sha(chain[4]) or runtime['scope'] != 'SIMULATION_ONLY' or type(runtime['enabled']) is not bool:
        raise ValueError('runtime binding')
    digest(runtime['runtime_config_digest'])
    if runtime['enabled'] and not promotion['authorized']:
        raise PermissionError('unauthorized runtime')
    return {'chain_digest': sha(chain), 'research_qualified': qualification['research_qualified'],
            'status': 'VERIFIED_EVIDENCE_ONLY', 'runtime_capability_issued': False,
            'external_deployment_allowed': False,
            'limits': 'Signatures attribute assertions to pinned roles. Truthful inference, independently managed witness/custody, sampling review and actual artifact reproduction remain operational requirements.'}
