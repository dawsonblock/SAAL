import copy
import json

import pytest

from saal_rsi.adaptive_artifacts import rank_history
from saal_rsi.adaptive_backend import tensor_inventory
from saal_rsi.adaptive_rank_controller import budget_schedule, initial_state
from saal_rsi.spec import sha


def synthetic_history(job):
    """Validator wire-format fixture; no model training claim."""
    adapter = job['adapter']; state = initial_state(adapter); previous = 'sha256:'+'0'*64; rows = []
    for step in range(1, 5):
        scheduled, due = budget_schedule(adapter, step)
        allocated = step in (2, 3)
        if allocated: state['masks'] = {t: [True]*8+[False]*4 for t in adapter['target_modules']}
        state['step'] = step
        active = {t: sum(v) for t, v in state['masks'].items()}
        row = {'schema': 'saal-adaptive-rank-event-v1', 'completed_optimizer_step': step,
               'previous_hash': previous, 'allocated_this_update': allocated,
               'scheduled_budget': scheduled, 'applied_budget': scheduled if allocated else (264 if step == 1 else 176),
               'state': copy.deepcopy(state), 'active_rank_by_layer': active, 'aggregate_active_rank': sum(active.values()),
               'nonzero_singular_by_layer': {t: 1 for t in active},
               'importance_scores': {t: [0.]*12 for t in active},
               'gradients': {'finite': True, 'norm': .1, 'gradient_tensors': len(tensor_inventory(adapter))-22},
               'adapter_gradient_norms': {t: .01 for t in active},
               'elapsed_seconds': float(step), 'gpu_memory': {'allocated_bytes': 0, 'reserved_bytes': 0}}
        row['event_hash'] = sha(row); previous = row['event_hash']; rows.append(row)
    return rows


def test_complete_history_and_final_budget(tmp_path, admitted_job):
    rows = synthetic_history(admitted_job)
    (tmp_path/'rank_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    validated, summary = rank_history(tmp_path, admitted_job)
    assert summary['active_rank_total'] == 176 and summary['allocation_events'] == 2


@pytest.mark.parametrize('fault', ['edit_event', 'omit_step', 'repeat_step', 'no_change', 'budget',
                                  'missing_scores', 'bad_gradients', 'over_time', 'mask_change', 'skip_allocation', 'no_singular_updates'])
def test_manipulated_rank_history_is_rejected(tmp_path, admitted_job, fault):
    rows = synthetic_history(admitted_job)
    if fault == 'edit_event': rows[1]['importance_scores'][next(iter(rows[1]['importance_scores']))][0] = 1
    elif fault == 'omit_step': rows.pop(1)
    elif fault == 'repeat_step': rows[1]['completed_optimizer_step'] = 1
    elif fault == 'no_change': rows[-1]['state']['masks'] = initial_state(admitted_job['adapter'])['masks']
    elif fault == 'budget': rows[2]['aggregate_active_rank'] = 265
    elif fault == 'missing_scores': rows[1]['importance_scores'] = {}
    elif fault == 'bad_gradients': rows[1]['gradients']['finite'] = False
    elif fault == 'over_time': rows[1]['elapsed_seconds'] = 100000
    elif fault == 'mask_change': rows[3]['state']['masks'][next(iter(rows[3]['state']['masks']))][0] = False
    elif fault == 'skip_allocation': rows[1]['allocated_this_update'] = False
    else: rows[-1]['nonzero_singular_by_layer'] = {t: 0 for t in rows[-1]['state']['masks']}
    # Rehash to exercise semantic checks as well as cryptographic byte changes.
    if fault != 'edit_event':
        previous = 'sha256:'+'0'*64
        for row in rows:
            row['previous_hash'] = previous; row['event_hash'] = sha({k: v for k, v in row.items() if k != 'event_hash'}); previous = row['event_hash']
    (tmp_path/'rank_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): rank_history(tmp_path, admitted_job)
