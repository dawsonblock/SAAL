import pytest
from dream_rsi_governed.demo import make_demo_worlds
from dream_rsi_governed.models import ReplayDecision
from dream_rsi_governed.replay import ReplayEngine, ReplayError


class TooWide:
    policy_id = "too-wide"
    def decide(self, c):
        return ReplayDecision(tuple(a.id for a in c.legal_actions))


class Unknown:
    policy_id = "unknown"
    def decide(self, c):
        return ReplayDecision(("does-not-exist",))


def test_parallelism_cap_enforced():
    with pytest.raises(ReplayError):
        ReplayEngine().run(make_demo_worlds(1)[0], TooWide())


def test_unsupported_action_is_penalized_not_materialized():
    t = ReplayEngine().run(make_demo_worlds(1)[0], Unknown())
    assert t.support_ratio == 0.0
    assert t.probes == 0
