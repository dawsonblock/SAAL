import json
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'research/diverse-learning-v1/src'))
from saal_rsi.compositional_corpus_v5 import build, validate
from saal_rsi.structured_choice import plan
from saal_rsi.spec import sha


@pytest.fixture(scope='module')
def corpus():
    parent = json.loads((ROOT/'research/behavior-closure-v4/dataset.json').read_text())
    data, audit = build(parent)
    return parent, data, audit


def test_balanced_new_supervision_preserves_old_public_data(corpus):
    from collections import Counter
    parent, data, audit = corpus
    assert validate(data, audit, parent)['new_records'] == 1152
    assert data['records'][:len(parent['records'])] == parent['records']
    old_ids = {r['record_id'] for r in parent['records']}
    for split, count in [('train', 192), ('development', 48), ('regression', 48)]:
        labels = Counter(r['expected_decision'] for r in data['records']
                         if r['record_id'] not in old_ids and r['split'] == split)
        assert labels == dict.fromkeys(('CHEAP', 'FAST', 'ASK', 'REFUSE'), count)
    assert audit['human_reviews'] == 0 and data['independent_population'] is False


def test_every_new_prompt_matches_state_and_independent_planning_control(corpus):
    _, data, audit = corpus
    rows = {r['record_id']: r for r in data['records']}
    for item in audit['records']:
        row = rows[item['record_id']]
        assert item['state_digest'] == sha(item['state']) and item['prompt_digest'] == sha(row['prompt'])
        assert plan(item['state'])['decision'] == row['expected_decision']


def test_semantic_or_lineage_substitution_is_rejected(corpus):
    import copy
    parent, data, audit = corpus
    changed = copy.deepcopy(audit)
    changed['records'][0]['state']['previous_priority'] = 'changed'
    with pytest.raises(ValueError): validate(data, changed, parent)
