"""Versioned replay plans. Current-example and replay RNG streams are separate."""
import random
from collections import defaultdict
from .spec import sha


def reservoir(records, capacity, seed):
    if type(capacity) is not int or not 1 <= capacity <= 1024 or type(seed) is not int or seed < 0:
        raise ValueError('bounded replay reservoir')
    rng = random.Random(seed); pool = []
    for i, record in enumerate(records):
        if i < capacity: pool.append(record)
        else:
            index = rng.randrange(i + 1)
            if index < capacity: pool[index] = record
    return pool


def batch_plan(current, previous, *, seed, replay_seed, stage, steps, batch_size,
               accumulation, replay_ratio, policy, capacity=512):
    if not current or any(r['split'] != 'train' for r in current + previous):
        raise ValueError('replay accepts approved public training examples only')
    ids = [r['record_id'] for r in current + previous]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate or current/previous overlap')
    if policy not in ('none', 'reservoir-v1', 'stratified-family-v1') or type(replay_ratio) is not float or replay_ratio not in (0., .25, .5) or (policy == 'none') != (replay_ratio == 0.):
        raise ValueError('declared replay policy and ratio')
    if any(type(x) is not int or x < 0 for x in (seed, replay_seed, stage)) or any(type(x) is not int or not 1 <= x <= bound for x, bound in ((steps, 100), (batch_size, 32), (accumulation, 16))):
        raise ValueError('replay work budget')
    if replay_ratio and batch_size < 4:
        raise ValueError('replay requires at least four admitted batch slots')
    pool = reservoir(previous, capacity, replay_seed + stage * 1013)
    families = defaultdict(list)
    for r in pool: families[r['task_family_id']].append(r)
    if policy == 'stratified-family-v1' and len(families) != len({r['task_family_id'] for r in previous}):
        raise ValueError('reservoir omitted a prior family; increase admitted capacity')
    current_rng = random.Random(seed + stage * 1009)
    replay_rng = random.Random(replay_seed + stage * 1019)
    replay_count = round(batch_size * replay_ratio) if pool else 0
    current_count = batch_size - replay_count
    plans = []
    family_names = sorted(families)
    for _ in range(steps * accumulation):
        current_ids = [current_rng.choice(current)['record_id'] for _ in range(current_count)]
        replay_ids = []
        for _ in range(replay_count):
            if policy == 'stratified-family-v1':
                record = replay_rng.choice(families[replay_rng.choice(family_names)])
            else: record = replay_rng.choice(pool)
            replay_ids.append(record['record_id'])
        plans.append({'current_ids': current_ids, 'replay_ids': replay_ids})
    return {'schema': 'saal-replay-batch-plan-v1', 'policy': policy,
            'current_seed': seed + stage * 1009, 'replay_seed': replay_seed + stage * 1019,
            'pool_record_ids': [r['record_id'] for r in pool], 'batches': plans,
            'current_records_digest': sha(current), 'previous_records_digest': sha(previous),
            'current_order_matched_with_fresh_control': True,
            'limits': 'Fresh control receives identical current examples. Experimental replay adds prior-example exposure and measured work; equal steps do not imply equal tokens or compute.'}
