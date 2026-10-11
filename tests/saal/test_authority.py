import tempfile
import time
import unittest
from pathlib import Path

from saal.authority import Authority, SimulatedExecutor
from saal.types import Action, Delegation


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a = Authority(b"A"*32, b"H"*32, replay_db=Path(self.tmp.name)/"replay.sqlite")
        self.now = 1000
        self.action = Action("schedule.propose", {"slot_id": "example", "cost_cents": 0}, "user", "task")
        self.d = Delegation("user", "task", "propose only", ("schedule.propose",), 2000)

    def tearDown(self):
        self.a.close()
        self.tmp.cleanup()

    def test_authorize_and_one_use(self):
        decision = self.a.decide(self.action, self.d, self.now)
        self.assertEqual(decision.disposition, "ALLOW")
        self.assertTrue(self.a.consume(decision.token, self.action, self.now))
        self.assertFalse(self.a.consume(decision.token, self.action, self.now))

    def test_mismatched_action_and_forged_token(self):
        decision = self.a.decide(self.action, self.d, self.now)
        other = Action("schedule.propose", {"slot_id": "other", "cost_cents": 0}, "user", "task")
        self.assertFalse(self.a.consume(decision.token, other, self.now))
        tampered = {"claims": {**decision.token["claims"], "exp": 9_999_999},
                    "signature": decision.token["signature"]}
        self.assertFalse(self.a.consume(tampered, self.action, self.now))

    def test_denies_expiration_and_privilege_escalation(self):
        self.assertEqual(self.a.decide(self.action, self.d, 2000).disposition, "DENY")
        self.assertEqual(self.a.decide(Action("model.promote", {}, "user", "task"), self.d, self.now).disposition, "DENY")
        self.assertEqual(self.a.decide(Action("schedule.commit", {"slot_id": "x", "cost_cents": 0}, "user", "task"), self.d, self.now).disposition, "DENY")
        self.assertEqual(self.a.decide(Action("schedule.propose", {"slot_id": "x", "cost_cents": True}, "user", "task"), self.d, self.now).disposition, "DENY")
        self.assertEqual(self.a.decide(Action("schedule.propose", {"slot_id": "x", "cost_cents": 1}, "user", "task"), self.d, self.now).disposition, "DENY")

    def test_high_risk_requires_out_of_band_approval(self):
        committed = Action("schedule.commit", {"slot_id": "x", "cost_cents": 0}, "user", "task")
        d = Delegation("user", "task", "book with approval", ("schedule.commit",), 2000, may_affect_others=True)
        self.assertEqual(self.a.decide(committed, d, self.now).disposition, "APPROVAL_REQUIRED")
        with self.assertRaises(PermissionError):
            self.a.approve(committed, b"bad credential")
        self.a.approve(committed, b"H"*32, now=self.now)
        self.assertEqual(self.a.decide(committed, d, self.now).disposition, "ALLOW")
        self.assertEqual(self.a.decide(committed, d, self.now).disposition, "APPROVAL_REQUIRED")

    def test_revocation_invalidates_issued_capability(self):
        issued = self.a.decide(self.action, self.d, self.now)
        self.a.revoke("user", "task", b"H"*32)
        self.assertFalse(self.a.consume(issued.token, self.action, self.now))
        self.assertEqual(self.a.decide(self.action, self.d, self.now).disposition, "DENY")

    def test_strict_scope_and_no_external_effects(self):
        committed = Action("schedule.commit", {"slot_id": "x", "cost_cents": 0}, "user", "task")
        d = Delegation("user", "task", "no external effects", ("schedule.commit",), 2000)
        self.assertEqual(self.a.decide(committed, d, self.now).disposition, "DENY")
        self.assertEqual(self.a.decide(Action("schedule.propose", {"slot_id": "x", "cost_cents": 0}, "other", "task"), self.d, self.now).disposition, "DENY")

    def test_replay_survives_authority_restart(self):
        granted = self.a.decide(self.action, self.d, self.now)
        self.assertTrue(self.a.consume(granted.token, self.action, self.now))
        again = Authority(b"A"*32, b"H"*32, replay_db=Path(self.tmp.name)/"replay.sqlite")
        self.assertFalse(again.consume(granted.token, self.action, self.now))
        again.close()


if __name__ == "__main__":
    unittest.main()
