import copy
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.attempt_authority import Witness, CustodianMirror, evaluate_once, ZERO
from saal_rsi.spec import sha


def setup(tmp_path):
    key = Ed25519PrivateKey.generate()
    enrollment = {'tenant': 'lab', 'family': 'fixed_question', 'question_digest': sha('fixed'), 'total_alpha': .05, 'max_attempts': 3}
    witness = Witness.initialize(tmp_path / 'witness.db', key, enrollment)
    client = lambda op, args: witness.transact('custodian', op, args)
    mirror = CustodianMirror.initialize(tmp_path / 'mirror.db', client, enrollment, witness.pin)
    bindings = {k: sha(k) for k in ('candidate_digest', 'baseline_digest', 'protocol_digest', 'holdout_digest', 'sampling_review_digest', 'question_digest')}
    bindings['question_digest'] = enrollment['question_digest']
    return witness, mirror, bindings


def test_consumed_before_real_inference_and_no_refund(tmp_path):
    w, m, b = setup(tmp_path); reservation = m.reserve(b)
    assert reservation['payload']['alpha'] == .025
    invoked = []
    def inference(alpha):
        assert w.head()['payload']['seq'] == 2
        invoked.append(alpha)
        raise RuntimeError('interrupted evaluator')
    client = lambda op, args: w.transact('evaluator', op, args)
    with pytest.raises(RuntimeError): evaluate_once(reservation, b, client, w.pin, w.enrollment, inference)
    with pytest.raises(PermissionError): evaluate_once(reservation, b, client, w.pin, w.enrollment, inference)
    assert invoked == [.025] and w.head()['payload']['attempt_count'] == 1


def test_rollback_and_new_registry_do_not_reset_alpha(tmp_path):
    w, m, b = setup(tmp_path)
    shutil.copy2(m.path, tmp_path / 'old.db')
    m.reserve(b)
    shutil.copy2(tmp_path / 'old.db', m.path)
    b['candidate_digest'] = sha('new')
    with pytest.raises(PermissionError): m.reserve(b)
    with pytest.raises(PermissionError): CustodianMirror.initialize(tmp_path / 'new.db', m.client, w.enrollment, w.pin)
    assert w.head()['payload']['attempt_count'] == 1


def test_consumption_ack_and_spent_interrupted_reservation(tmp_path):
    w, m, b = setup(tmp_path); reservation = m.reserve(b)
    result = evaluate_once(reservation, b, lambda op, args: w.transact('evaluator', op, args), w.pin, w.enrollment, lambda alpha: {'failed': True})
    m.acknowledge_consumption(result['consumption'])
    b['candidate_digest'] = sha('second')
    second = m.reserve(b)
    assert second['payload']['attempt'] == 2 and second['payload']['alpha'] == .0125
    # Abandoned reserved attempt still consumes alpha; next allocation is .00625.
    b['candidate_digest'] = sha('third'); assert m.reserve(b)['payload']['alpha'] == .00625
    b['candidate_digest'] = sha('fourth')
    with pytest.raises(PermissionError): m.reserve(b)


@pytest.mark.parametrize('field', ['candidate_digest', 'holdout_digest', 'protocol_digest', 'baseline_digest', 'sampling_review_digest'])
def test_identity_substitution_prevents_inference(tmp_path, field):
    w, m, b = setup(tmp_path); r = m.reserve(b); changed = b | {field: sha('substituted')}
    with pytest.raises(PermissionError): evaluate_once(r, changed, lambda op, args: w.transact('evaluator', op, args), w.pin, w.enrollment, lambda _: pytest.fail('must never enter inference'))
    assert w.head()['payload']['seq'] == 1


@pytest.mark.parametrize('role,operation', [('controller','reserve'),('controller','head'),('custodian','consume'),('evaluator','reserve')])
def test_role_acl(tmp_path, role, operation):
    w, m, b = setup(tmp_path)
    with pytest.raises(PermissionError): w.transact(role, operation, {})


def test_concurrent_compare_and_swap(tmp_path):
    w, m, b = setup(tmp_path)
    def reserve(i):
        try:
            return w.transact('custodian', 'reserve', {'expected_head': ZERO, 'bindings': b | {'candidate_digest': sha(i)}})
        except PermissionError: return None
    with ThreadPoolExecutor(max_workers=8) as executor: result = list(executor.map(reserve, range(8)))
    assert sum(x is not None for x in result) == 1
    assert w.head()['payload']['attempt_count'] == 1


def test_corrupted_journal_and_question_reset_rejected(tmp_path):
    w, m, b = setup(tmp_path)
    with pytest.raises(PermissionError): m.reserve(b | {'question_digest': sha('reset')})
    m.reserve(b)
    with sqlite3.connect(w.path) as db: db.execute("UPDATE events SET body='{}'")
    with pytest.raises(ValueError): w.head()


def test_missing_witness_cannot_bootstrap_itself(tmp_path):
    w, m, b = setup(tmp_path); w.path.unlink()
    with pytest.raises(PermissionError): Witness(w.path, w.key, w.enrollment)


def test_unavailable_witness_blocks_reservation(tmp_path):
    w, m, b = setup(tmp_path)
    def down(*_): raise TimeoutError('offline witness')
    m.client = down
    with pytest.raises(TimeoutError): m.reserve(b)
    assert w.head()['payload']['attempt_count'] == 0
