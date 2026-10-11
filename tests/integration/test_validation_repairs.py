import copy
import pytest
from saal_rsi.sequence_replay_v2 import prediction_equal
from saal_rsi.research_selection import select_research_snapshots
from saal_rsi.spec import sha
from saal_rsi.evidence_chain import verify_chain
from saal_rsi.model_contract import sign
from test_corrective_evidence_chain import make, resign_tail

KIND = 'modernbert-seqcls-v1'
GOOD = {'decision': 'CHEAP', 'probabilities': [.7, .1, .1, .1]}

@pytest.mark.parametrize('values', [None, [], [.7], 'abcd', [1., 1., 1., 1.], [-.1, .5, .3, .3], [float('nan'), .1, .1, .1], [True, 0., 0., 0.]])
def test_malformed_replay_probabilities_fail_closed(values):
    malformed = dict(GOOD, probabilities=values)
    assert not prediction_equal([malformed], [malformed], KIND, 1e-4)
    assert not prediction_equal([GOOD], [malformed], KIND, 1e-4)

@pytest.mark.parametrize('label', ['FAST', 'FORBIDDEN', None, ['CHEAP']])
def test_replay_invalid_or_inconsistent_decision(label):
    p = dict(GOOD, decision=label)
    assert not prediction_equal([p], [p], KIND, 1e-4)

@pytest.mark.parametrize('tolerance', [float('nan'), float('inf'), -.1, .1, True, None])
def test_replay_invalid_tolerance(tolerance):
    assert not prediction_equal([GOOD], [GOOD], KIND, tolerance)

def test_replay_probability_agreement_and_disagreement():
    assert prediction_equal([GOOD], [copy.deepcopy(GOOD)], KIND, 1e-4)
    near = dict(GOOD, probabilities=[.70001, .09999, .1, .1])
    assert prediction_equal([GOOD], [near], KIND, 1e-4)
    far = dict(GOOD, probabilities=[.71, .09, .1, .1])
    assert not prediction_equal([GOOD], [far], KIND, 1e-4)

def snapshots():
    return [{'step': s, 'checkpoint_digest': sha(s), 'development': {'family_macro_accuracy': .5, 'worst_family_accuracy': .5, 'accuracy': .5, 'complete_pair_accuracy': None, 'invalid_outputs': 0, 'unauthorized_proposals': 0}} for s in (0, 10)]

def test_selection_missing_pair_metric_rejected_explicitly():
    rows = snapshots(); del rows[0]['development']['complete_pair_accuracy']
    with pytest.raises(ValueError, match='pair metric required'):
        select_research_snapshots(rows)

def test_selection_mixed_pair_measurement_rejected():
    rows = snapshots(); rows[1]['development']['complete_pair_accuracy'] = .5
    with pytest.raises(ValueError, match='pair-measurement coverage'):
        select_research_snapshots(rows)

def test_selection_unmeasured_pairs_supported():
    result = select_research_snapshots(snapshots())
    assert result['best_public_learning']['step'] == 0
    assert result['terminal']['step'] == 10

@pytest.mark.parametrize('attempt,seq,alpha', [(2, 3, .025), (30, 31, .001), (2, 2, .0125), (True, 3, .0125), ('2', 3, .0125)])
def test_receipt_alpha_and_counter_must_match_attempt(attempt, seq, alpha):
    keys, pins, identity, receipt, chain = make()
    receipt = sign(receipt['payload'] | {'attempt': attempt, 'seq': seq, 'alpha': alpha}, keys['custodian'])
    chain[2]['payload']['details']['consumption_digest'] = sha(receipt)
    resign_tail(chain, keys, 2)
    with pytest.raises(PermissionError):
        verify_chain(chain, pins, identity, now=100, consumption=receipt)

def test_receipt_later_attempt_with_lower_alpha_is_valid_evidence_only():
    keys, pins, identity, receipt, chain = make()
    receipt = sign(receipt['payload'] | {'attempt': 2, 'seq': 3, 'alpha': .0125}, keys['custodian'])
    chain[2]['payload']['details']['consumption_digest'] = sha(receipt)
    resign_tail(chain, keys, 2)
    assert verify_chain(chain, pins, identity, now=100, consumption=receipt)['runtime_capability_issued'] is False
