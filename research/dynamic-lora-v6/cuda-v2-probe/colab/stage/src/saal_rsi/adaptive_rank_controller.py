"""Completed-update rank allocation, including PEFT's delayed masking cadence."""
import copy
import math

from .spec import sha


def budget_schedule(adapter, step):
    count = len(adapter['target_modules'])
    initial, target = adapter['init_r']*count, adapter['target_r']*count
    if adapter['method'] != 'adalora-12-to-8':
        return initial, False
    s = adapter['schedule']
    if step <= s['tinit']:
        return initial, False
    if step > s['total_step']-s['tfinal']:
        return target, True
    fraction = 1-(step-s['tinit'])/(s['total_step']-s['tfinal']-s['tinit'])
    return int((initial-target)*fraction**3+target), step % s['deltaT'] == 0


def initial_state(adapter):
    return {'schema': 'saal-adaptive-rank-state-v1', 'step': 0,
            'method': adapter['method'],
            'masks': {t: [True]*adapter['init_r'] for t in adapter['target_modules']},
            'ranknum': {t: float(adapter['init_r']) for t in adapter['target_modules']}
                       if adapter['method'] == 'adalora-12-to-8' else {}}


def validate_state(state, adapter):
    if type(state) is not dict or set(state) != {'schema', 'step', 'method', 'masks', 'ranknum'}:
        raise ValueError('rank state schema')
    if state['schema'] != 'saal-adaptive-rank-state-v1' or state['method'] != adapter['method'] or type(state['step']) is not int or state['step'] < 0:
        raise ValueError('rank state identity')
    if type(state['masks']) is not dict or set(state['masks']) != set(adapter['target_modules']):
        raise ValueError('rank mask target inventory')
    for mask in state['masks'].values():
        if type(mask) is not list or len(mask) != adapter['init_r'] or any(type(v) is not bool for v in mask):
            raise ValueError('rank mask shape / Boolean values')
    if state['ranknum'] != initial_state(adapter)['ranknum']:
        raise ValueError('rank scaling substitution')
    if adapter['method'] != 'adalora-12-to-8' and state['masks'] != initial_state(adapter)['masks']:
        raise ValueError('fixed adapter cannot be dynamically masked')
    return state


def apply_state(model, adapter, state):
    """Restore mask metadata without compacting tensors or changing scaling."""
    validate_state(state, adapter)
    if adapter['method'] == 'adalora-12-to-8':
        pattern = {t+'.lora_E.default': mask for t, mask in state['masks'].items()}
        model.base_model.peft_config['default'].rank_pattern = pattern
        model.base_model.rankallocator.mask_using_rank_pattern(model.base_model.model, pattern)


def importance_scores(allocator, targets):
    """Triplet scores exactly as combined by pinned PEFT 0.21.2."""
    result = {}
    for target in targets:
        keys = [target+'.'+k+'.default' for k in ('lora_A', 'lora_B', 'lora_E')]
        if any(k not in allocator.exp_avg_ipt for k in keys):
            return {}
        a, b, e = [allocator._element_score(k) for k in keys]
        values = e.view(-1)+a.mean(dim=1)+b.mean(dim=0)
        result[target] = values.detach().cpu().tolist()
    return result


class RankController:
    def __init__(self, model, adapter):
        self.model, self.adapter = model, adapter
        self.state = initial_state(adapter)
        self.applied_budget = adapter['init_r']*len(adapter['target_modules'])
        self.previous_hash = 'sha256:'+'0'*64
        self.allocations = 0
        self.history = []
        self.last_importance = {}

    def complete_update(self, step, *, optimizer_updated, gradient_evidence, elapsed_seconds, gpu_memory):
        # Failed/skipped optimizers must never advance the allocation counter.
        if optimizer_updated is not True:
            return None
        if type(step) is not int or step != self.state['step']+1:
            raise ValueError('allocation must follow exactly one completed optimizer update')
        if (type(gradient_evidence) is not dict or gradient_evidence.get('finite') is not True
                or type(gradient_evidence.get('norm')) not in (float, int)
                or not math.isfinite(gradient_evidence['norm']) or gradient_evidence['norm'] < 0):
            raise ValueError('finite optimizer gradient evidence required')
        scheduled_budget, due = budget_schedule(self.adapter, step)
        scores = {}
        allocated = False
        nonzero = None
        dynamic = self.adapter['method'] == 'adalora-12-to-8'
        import torch
        named = dict(self.model.named_parameters())
        adapter_norms = {}
        for target in self.adapter['target_modules']:
            suffixes = ('lora_A.default', 'lora_B.default', 'lora_E.default') if dynamic else ('lora_A.default.weight', 'lora_B.default.weight')
            parameters = [named['base_model.model.'+target+'.'+suffix] for suffix in suffixes]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError('missing / nonfinite adapter gradients')
            adapter_norms[target] = float(torch.sqrt(sum(p.grad.detach().float().square().sum() for p in parameters)))
        if dynamic:
            s = self.adapter['schedule']
            if step > s['total_step']:
                raise ValueError('rank schedule budget exhausted')
            allocator = self.model.base_model.rankallocator
            # At the final allocation PEFT clears importance buffers. Preserve
            # the scores used for that allocation before the reset.
            old_scores = importance_scores(allocator, self.adapter['target_modules'])
            params = dict(self.model.base_model.model.named_parameters())
            import torch
            for target in self.adapter['target_modules']:
                for name in ('lora_A', 'lora_B', 'lora_E'):
                    p = params[target+'.'+name+'.default']
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError('missing / nonfinite AdaLoRA allocation gradients')
            self.model.base_model.update_and_allocate(step)
            scores = importance_scores(allocator, self.adapter['target_modules']) or old_scores or self.last_importance
            self.last_importance = copy.deepcopy(scores)
            pattern = self.model.base_model.peft_config['default'].rank_pattern
            allocated = step == s['total_step']-s['tfinal'] or (s['tinit'] < step < s['total_step']-s['tfinal'] and due)
            if pattern is not None:
                expected = {t+'.lora_E.default' for t in self.adapter['target_modules']}
                if set(pattern) != expected:
                    raise ValueError('unexpected PEFT rank target names')
                self.state['masks'] = {t: list(pattern[t+'.lora_E.default']) for t in self.adapter['target_modules']}
                # PEFT 0.21.2 only reapplies masks periodically during budget
                # reduction. Its zeroed E values can regrow after an optimizer
                # step. Our separately admitted persistent-mask policy prevents
                # that temporary capacity expansion. Importance was updated
                # above, so a future allocation can still reactivate a triplet.
                allocator.mask_using_rank_pattern(self.model.base_model.model, pattern)
            if allocated:
                self.applied_budget = scheduled_budget
                self.allocations += 1
            nonzero = {}
            for target, mask in self.state['masks'].items():
                p = params[target+'.lora_E.default'].detach()
                if not torch.isfinite(p).all() or any(float(p[j]) != 0 for j, enabled in enumerate(mask) if not enabled):
                    raise ValueError('singular values do not obey rank masks')
                nonzero[target] = int(torch.count_nonzero(p))
        self.state['step'] = step
        validate_state(self.state, self.adapter)
        active = {t: sum(mask) for t, mask in self.state['masks'].items()}
        # A budget changes only at actual mask events. Between events PEFT's
        # scheduled target is smaller than the last applied mask budget.
        if sum(active.values()) > self.applied_budget:
            raise ValueError('aggregate active rank exceeds applied budget')
        if any(not math.isfinite(v) or v < 0 for values in scores.values() for v in values):
            raise ValueError('invalid importance estimates')
        row = {'schema': 'saal-adaptive-rank-event-v1', 'completed_optimizer_step': step,
               'previous_hash': self.previous_hash, 'allocated_this_update': allocated,
               'scheduled_budget': scheduled_budget, 'applied_budget': self.applied_budget,
               'state': copy.deepcopy(self.state), 'active_rank_by_layer': active,
               'aggregate_active_rank': sum(active.values()), 'nonzero_singular_by_layer': nonzero,
               'adapter_gradient_norms': adapter_norms,
               'importance_scores': scores, 'gradients': gradient_evidence,
               'elapsed_seconds': elapsed_seconds, 'gpu_memory': gpu_memory}
        row['event_hash'] = sha(row)
        self.previous_hash = row['event_hash']
        self.history.append(row)
        return row

    def require_completion(self, steps):
        if self.state['step'] != steps or len(self.history) != steps:
            raise ValueError('incomplete optimizer / allocation history')
        if self.adapter['method'] == 'adalora-12-to-8':
            final = sum(sum(v) for v in self.state['masks'].values())
            if (self.allocations == 0 or not 0 < final <= self.adapter['target_r']*len(self.adapter['target_modules'])
                    or self.state['masks'] == initial_state(self.adapter)['masks']
                    or sum(self.history[-1]['nonzero_singular_by_layer'].values()) == 0):
                raise ValueError('adaptive completion requires actual rank reduction')
        return {'completed_optimizer_updates': steps, 'allocation_events': self.allocations,
                'rank_history_digest': self.previous_hash,
                'stored_rank_per_layer': self.adapter['init_r'],
                'active_rank_total': sum(sum(v) for v in self.state['masks'].values())}

