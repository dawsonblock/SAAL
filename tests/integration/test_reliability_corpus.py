import copy
import collections
import json
from pathlib import Path
import pytest
from saal_rsi.reliability_corpus import build, validate, compact_population, render
from saal_rsi.choice_policy_v1 import decide, differences


@pytest.fixture(scope='module')
def corpus():
    root = Path(__file__).resolve().parents[2] / 'research/diverse-learning-v1/experiment'
    diverse = json.loads((root / 'diverse_dataset.json').read_text()); old = json.loads((root / 'old_control_dataset.json').read_text())
    data, audit = build(diverse, old)
    return data, audit, diverse, old


def test_replay_balance_and_evaluation_preservation(corpus):
    data, audit, diverse, old = corpus
    summary = validate(data, audit, diverse, old)
    assert summary['records'] == 5464
    assert summary['split_class_counts']['train'] == {d: 960 for d in ('CHEAP', 'FAST', 'ASK', 'REFUSE')}
    assert not summary['independence_established'] and summary['human_reviews'] == 0
    prior_eval = {r['record_id'] for r in diverse['records'] if r['split'] != 'train'}
    current_train = {r['record_id'] for r in data['records'] if r['split'] == 'train'}
    assert not prior_eval & current_train
    assert collections.Counter(audit['record_sources'][r['record_id']] for r in data['records'] if r['split'] == 'train') == {'compact': 1536, 'prior_diverse': 1536, 'historical_v3': 768}


def test_compact_states_reconstruct_labels_and_single_condition_pairs():
    data, states = compact_population(); by_id = {r['record_id']: r for r in data['records']}; pairs = collections.defaultdict(list)
    for a in states:
        r = by_id[a['record_id']]
        assert decide(a['state'])[0] == r['expected_decision']
        assert render(a['state'], a['recipe'], a['number'], a['quote_cost'], a['quote_granted']) == r['prompt']
        pairs[r['cluster_id']].append(a)
    assert len(pairs) == 1152
    for a, b in pairs.values(): assert differences(a['state'], b['state']) == [a['changed_field']]


@pytest.mark.parametrize('mutation', ['training', 'evaluation', 'state', 'source', 'review'])
def test_replay_or_admission_tampering_rejected(corpus, mutation):
    data, audit, diverse, old = copy.deepcopy(corpus)
    if mutation == 'training': data['records'][0]['prompt'] = 'changed replay prompt'
    elif mutation == 'evaluation': data['records'] = [r for r in data['records'] if r['record_id'] != next(r['record_id'] for r in diverse['records'] if r['split'] == 'regression')]
    elif mutation == 'state': audit['compact_states'][0]['state']['priority'] = None
    elif mutation == 'source': audit['record_sources'][data['records'][0]['record_id']] = 'independent'
    else: audit['human_reviews'] = 2
    with pytest.raises(ValueError): validate(data, audit, diverse, old)
