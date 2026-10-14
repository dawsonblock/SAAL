"""Constraint preflight plus the existing capability authority, simulation only.

Context and clock are supplied by the trusted simulation host, never extracted
from model text. Local Python callers are a trust assumption, not OS isolation.
No keys, approvals, model activation or external tool adapters live here.
"""
import copy
import math
from .choice_policy_v1 import validate_state, decide
from .spec import sha
from .task_oracles import parse_output
from .sequence_replay_v2 import prediction_equal


def proposal_decision(proposal):
    if type(proposal) is dict:
        if not prediction_equal([proposal], [proposal], 'modernbert-seqcls-v1', 1e-4):
            raise ValueError('malformed classifier proposal')
        return proposal['decision'], max(proposal['probabilities'])
    value = parse_output(proposal, 'decision')
    decision = value['decision']
    if value['requires_clarification'] != (decision == 'ASK') or value['action_authorized'] != (decision in ('CHEAP', 'FAST')):
        raise ValueError('contradictory structured proposal')
    # A generative model's assertion is never permission or calibrated confidence.
    return decision, None


def preflight(proposal, trusted_context, *, now, confidence_threshold=0.0):
    if type(now) is not int or not 0 <= now <= 10000:
        raise ValueError('trusted synthetic clock required')
    if type(confidence_threshold) not in (int, float) or not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError('bounded confidence threshold')
    state = copy.deepcopy(validate_state(trusted_context))
    state['authorization']['now'] = now
    decision, confidence = proposal_decision(proposal)
    reasons = []
    a = state['authorization']
    if not a['granted']: reasons.append('PERMISSION_REVOKED_OR_ABSENT')
    if a['source'] != 'trusted': reasons.append('UNTRUSTED_PERMISSION_SOURCE')
    if a['allowed_scope'] != a['requested_scope']: reasons.append('SCOPE_MISMATCH')
    if now >= a['expires']: reasons.append('PERMISSION_EXPIRED')
    if decision not in ('CHEAP', 'FAST'): reasons.append('NON_ACTION_PROPOSAL')
    if state['priority'] is None: reasons.append('MISSING_CURRENT_OBJECTIVE')
    for name, field in [('budget', 'cost'), ('deadline', 'duration')]:
        constraint = state[name]
        if constraint['required'] and constraint['limit'] is None:
            reasons.append('MISSING_REQUIRED_' + name.upper())
        elif decision in ('CHEAP', 'FAST') and constraint['required'] and state['options'][decision][field] > constraint['limit']:
            reasons.append('VIOLATES_' + name.upper())
    if confidence_threshold and (confidence is None or confidence < confidence_threshold):
        reasons.append('DEFER_TO_REVIEW')
    if not reasons and decide(state)[0] != decision:
        reasons.append('VIOLATES_TRUSTED_OBJECTIVE')
    # Clock progress does not substitute a task; expiry is checked again now.
    bound = copy.deepcopy(state)
    bound['authorization']['now'] = 0
    return {'schema': 'saal-decision-preflight-v1', 'decision': decision,
            'constraints_satisfied': not reasons, 'reasons': reasons,
            'context_digest': sha(bound), 'capability_still_required': True,
            'external_deployment_allowed': False}


def simulated_action(proposal, trusted_context, *, principal, task_id, now, confidence_threshold=0.0):
    from saal.types import Action
    state = copy.deepcopy(validate_state(trusted_context))
    checked = preflight(proposal, state, now=now, confidence_threshold=confidence_threshold)
    if not checked['constraints_satisfied']:
        raise PermissionError(','.join(checked['reasons']))
    if type(principal) is not str or not principal or type(task_id) is not str or not task_id:
        raise ValueError('trusted task identity')
    decision = checked['decision']
    return Action('sim.purchase', {'item_id': decision + ':' + checked['context_digest'][7:],
                                  'cost_cents': state['options'][decision]['cost']}, principal, task_id)


class GuardedSimulatedExecutor:
    """Requires host context AND a one-use, human-approved Authority capability."""
    def __init__(self, authority):
        from saal.authority import SimulatedExecutor
        self.executor = SimulatedExecutor(authority)

    @property
    def events(self):
        return self.executor.events

    def execute(self, proposal, trusted_context, token, *, principal, task_id, now, confidence_threshold=0.0):
        action = simulated_action(proposal, trusted_context, principal=principal, task_id=task_id,
                                  now=now, confidence_threshold=confidence_threshold)
        return self.executor.execute(action, token, now=now)
