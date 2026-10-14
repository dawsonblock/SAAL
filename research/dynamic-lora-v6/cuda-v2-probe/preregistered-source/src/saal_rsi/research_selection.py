"""Explicit public learning diagnostics, separate from historical safety selection.

This is a new, retrospective analysis policy. It never silently changes a signed
job's selection rule, re-labels its selected checkpoint, or admits qualification.
All three historical snapshots and unsuccessful learning outcomes stay visible.
"""
import math
from .spec import sha

POLICY = {'schema': 'saal-public-research-selection-policy-v1',
          'split': 'development', 'order': ['family_macro_accuracy', 'worst_family_accuracy',
                                           'complete_pair_accuracy', 'accuracy', 'earliest_step'],
          'purpose': 'LEARNING_DIAGNOSTIC_ONLY', 'qualification_admission': False}


def select_research_snapshots(snapshots, policy=None):
    policy = POLICY if policy is None else policy
    if policy != POLICY or type(snapshots) is not list or not snapshots:
        raise ValueError('closed public research selection')
    seen = set()
    for snapshot in snapshots:
        if type(snapshot) is not dict or set(snapshot) != {'step', 'checkpoint_digest', 'development'}:
            raise ValueError('snapshot schema')
        step = snapshot['step']
        if type(step) is not int or step < 0 or step in seen:
            raise ValueError('duplicate or invalid snapshot step')
        seen.add(step)
        from .attempt_authority import digest
        digest(snapshot['checkpoint_digest'])
        m = snapshot['development']
        if type(m) is not dict:
            raise ValueError('public metrics')
        for k in ('family_macro_accuracy', 'worst_family_accuracy', 'accuracy'):
            if type(m.get(k)) not in (int, float) or not math.isfinite(m[k]) or not 0 <= m[k] <= 1:
                raise ValueError('finite public accuracy')
        if 'complete_pair_accuracy' not in m:
            raise ValueError('pair metric required; use None for unmeasured pairs')
        pair = m['complete_pair_accuracy']
        if pair is not None and (type(pair) not in (int, float) or not math.isfinite(pair) or not 0 <= pair <= 1):
            raise ValueError('finite pair accuracy')
        for k in ('invalid_outputs', 'unauthorized_proposals'):
            if type(m.get(k)) is not int or m[k] < 0:
                raise ValueError('behavioral failures must remain visible')
    if len({s['development']['complete_pair_accuracy'] is None for s in snapshots}) != 1:
        raise ValueError('snapshots must share pair-measurement coverage')
    initial = min(snapshots, key=lambda s: s['step'])
    terminal = max(snapshots, key=lambda s: s['step'])
    if initial['step'] != 0:
        raise ValueError('initial checkpoint required')
    best = max(snapshots, key=lambda s: (s['development']['family_macro_accuracy'],
                                       s['development']['worst_family_accuracy'],
                                       s['development']['complete_pair_accuracy'] or 0,
                                       s['development']['accuracy'], -s['step']))
    gain = terminal['development']['family_macro_accuracy'] - initial['development']['family_macro_accuracy']
    return {'schema': 'saal-public-research-selection-v1', 'policy_digest': sha(policy),
            'initial': initial, 'best_public_learning': best, 'terminal': terminal,
            'terminal_family_macro_gain': gain, 'learning_gain_observed': gain > 0,
            'qualification_admission': 'NOT_ADMITTED', 'external_deployment_allowed': False,
            'limits': 'Public post hoc learning diagnostics; safety failures are reported, not pardoned. Protected qualification uses a separately frozen protocol and candidate.'}
