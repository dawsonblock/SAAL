from dataclasses import asdict
from pathlib import Path
import sqlite3
import tempfile

import pytest

from dream_rsi_governed.canonical import sha256_digest
from dream_rsi_governed.canary import compare_live_canary
from dream_rsi_governed.activation import authorize_activation, ActivationDenied
from dream_rsi_governed.calibration import replay_live_calibration
from dream_rsi_governed.experiment import replicated_score
from dream_rsi_governed.datasets import DatasetSplit, partition
from dream_rsi_governed.demo import make_demo_worlds
from dream_rsi_governed.evaluator import ReplayEvaluator
from dream_rsi_governed.ledger import AppendOnlyLedger
from dream_rsi_governed.models import CandidatePolicy
from dream_rsi_governed.policy import FixedBreadthPolicy, RiskAwareUCBPolicy
from dream_rsi_governed.promotion import PromotionAuthority, PromotionDenied
from dream_rsi_governed.qualification import Qualifier, QualificationError
from dream_rsi_governed.replay import ReplayEngine
from dream_rsi_governed.sandbox import inspect_policy_source
from dream_rsi_governed.signing import generate_private_key
from dream_rsi_governed.store import ArtifactStore


def test_canonical_digest_stable():
    assert sha256_digest({"b": 2, "a": 1}) == sha256_digest({"a": 1, "b": 2})


def test_replay_is_seed_deterministic():
    w = make_demo_worlds(1)[0]
    p = RiskAwareUCBPolicy()
    a = ReplayEngine(seed=11).run(w, p)
    b = ReplayEngine(seed=11).run(w, p)
    assert a == b


def test_different_policy_id_changes_rng_stream():
    w = make_demo_worlds(1)[0]
    a = ReplayEngine(seed=11).run(w, RiskAwareUCBPolicy(beta=1.0))
    b = ReplayEngine(seed=11).run(w, RiskAwareUCBPolicy(beta=2.0))
    assert a.policy_id != b.policy_id


def test_partition_is_deterministic():
    worlds = make_demo_worlds(20)
    p1 = {k: [w.world_id for w in v] for k,v in partition(worlds).items()}
    p2 = {k: [w.world_id for w in v] for k,v in partition(worlds).items()}
    assert p1 == p2


def test_store_roundtrip_and_digest_enforcement(tmp_path):
    s = ArtifactStore(tmp_path / "cas")
    d = s.put(b"hello")
    assert s.get(d) == b"hello"
    with pytest.raises(ValueError):
        s.get("sha256:../bad")


def test_ledger_is_hash_chained_and_append_only(tmp_path):
    l = AppendOnlyLedger(tmp_path / "ledger.sqlite3")
    l.append("one", {"x": 1})
    l.append("two", {"x": 2})
    assert l.verify()
    with sqlite3.connect(tmp_path / "ledger.sqlite3") as c:
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("DELETE FROM ledger WHERE seq=1")


def test_source_inspector_rejects_process_access():
    r = inspect_policy_source("import subprocess\nsubprocess.run(['x'])")
    assert not r.safe
    assert any("subprocess" in x for x in r.violations)


def test_source_inspector_allows_math():
    assert inspect_policy_source("import math\nx=math.sqrt(4)").safe


def test_qualifier_requires_holdout():
    evaluator = ReplayEvaluator(ReplayEngine(seed=1))
    q = Qualifier(evaluator, min_holdout_worlds=3)
    c = RiskAwareUCBPolicy()
    b = FixedBreadthPolicy()
    meta = CandidatePolicy(c.policy_id, "1", sha256_digest({"x": 1}))
    with pytest.raises(QualificationError):
        q.qualify(meta, c, b, make_demo_worlds(1), make_demo_worlds(2))


def test_failed_record_cannot_promote():
    worlds = make_demo_worlds(8)
    evaluator = ReplayEvaluator(ReplayEngine(seed=2))
    c = RiskAwareUCBPolicy(beta=1)
    b = FixedBreadthPolicy()
    meta = CandidatePolicy(c.policy_id, "1", sha256_digest({"x": 1}))
    # absurdly strict support threshold forces failure
    q = Qualifier(evaluator, min_holdout_worlds=3, min_support_ratio=1.01)
    rec = q.qualify(meta, c, b, worlds[:3], worlds[3:])
    assert not rec.passed
    with pytest.raises(PromotionDenied):
        PromotionAuthority(generate_private_key()).issue_archival(rec, compare_live_canary(meta.policy_id, b.policy_id, lambda:(1,1), lambda:(1,1), meta.source_digest), meta.source_digest, 1)


def test_signed_manifest_verifies_when_qualified():
    worlds = make_demo_worlds(12)
    evaluator = ReplayEvaluator(ReplayEngine(seed=3))
    # use same baseline implementation as candidate to make no-regression deterministic
    c = FixedBreadthPolicy(); b = FixedBreadthPolicy()
    meta = CandidatePolicy("candidate-fixed", "1", sha256_digest({"policy": "fixed"}))
    q = Qualifier(evaluator, min_holdout_worlds=3, min_support_ratio=0.0, max_uncertainty=99)
    rec = q.qualify(meta, c, b, worlds[:4], worlds[4:])
    assert rec.passed
    key = generate_private_key()
    auth = PromotionAuthority(key)
    canary = compare_live_canary(meta.policy_id, b.policy_id, lambda:(1.0,1.0), lambda:(1.0,1.0), meta.source_digest)
    manifest = auth.issue_archival(rec, canary, meta.source_digest, 1)
    assert auth.verify(manifest, key.public_key(), rec.evidence_digest)
    with pytest.raises(ActivationDenied, match="External activation prohibited"):
        authorize_activation(manifest, rec, canary, key.public_key(), meta.source_digest)


def test_replay_live_calibration():
    r = replay_live_calibration(["a","b","c"], [1,2,3], [10,20,30], threshold=0.9)
    assert r.passed and r.spearman_rho == pytest.approx(1.0)

def test_replicated_score_returns_interval():
    est = replicated_score(make_demo_worlds(4), FixedBreadthPolicy(), [1,2,3,4], bootstrap_samples=100)
    assert est.ci_low <= est.mean <= est.ci_high

def test_activation_rejects_wrong_artifact_digest():
    worlds = make_demo_worlds(12)
    evaluator = ReplayEvaluator(ReplayEngine(seed=3))
    c = FixedBreadthPolicy(); b = FixedBreadthPolicy()
    meta = CandidatePolicy("candidate-fixed", "1", sha256_digest({"policy": "fixed"}))
    q = Qualifier(evaluator, min_holdout_worlds=3, min_support_ratio=0.0, max_uncertainty=99)
    rec = q.qualify(meta, c, b, worlds[:4], worlds[4:])
    key = generate_private_key()
    canary = compare_live_canary(meta.policy_id, b.policy_id, lambda:(1,1), lambda:(1,1), meta.source_digest)
    manifest = PromotionAuthority(key).issue_archival(rec, canary, meta.source_digest, 1)
    with pytest.raises(ActivationDenied):
        authorize_activation(manifest, rec, canary, key.public_key(), sha256_digest({"wrong": True}))
