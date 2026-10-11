import os
import tempfile
import unittest
from pathlib import Path

from saal.authority import Authority
from saal.types import Action, Delegation


class BudgetV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "authority.sqlite"
        self.a = Authority(b"Z" * 32, b"Y" * 32, self.db)
        self.d = Delegation("person", "buy-demo", "synthetic purchases", ("sim.purchase",),
                            3000, max_cost_cents=100, may_affect_others=True)

    def tearDown(self):
        self.a.close()
        self.tmp.cleanup()

    def purchase(self, item, cost):
        return Action("sim.purchase", {"item_id": item, "cost_cents": cost}, "person", "buy-demo")

    def issued(self, item, cost, now=1000):
        action = self.purchase(item, cost)
        self.a.approve(action, b"Y" * 32, now=now)
        ticket = self.a.decide(action, self.d, now=now)
        return action, ticket

    def test_cumulative_budget_blocks_small_sequential_actions(self):
        first, t1 = self.issued("first", 60)
        self.assertEqual(t1.disposition, "ALLOW")
        self.assertTrue(self.a.consume(t1.token, first, 1000))
        second, t2 = self.issued("second", 41)
        self.assertEqual(t2.disposition, "DENY")
        self.assertIn("cumulative", t2.reasons[0])
        third, t3 = self.issued("third", 40)
        self.assertEqual(t3.disposition, "ALLOW")
        self.assertTrue(self.a.consume(t3.token, third, 1000))
        fourth, t4 = self.issued("fourth", 1)
        self.assertEqual(t4.disposition, "DENY")

    def test_outstanding_ticket_reserves_money(self):
        _, ticket = self.issued("pending", 80)
        self.assertEqual(ticket.disposition, "ALLOW")
        _, second = self.issued("overdraw", 30)
        self.assertEqual(second.disposition, "DENY")
        _, third = self.issued("after-expiry", 30, now=1061)
        self.assertEqual(third.disposition, "ALLOW")
        self.assertEqual(self.a.consume(ticket.token, self.purchase("pending", 80), 1061), False)

    def test_restart_keeps_consumed_budget(self):
        first, t1 = self.issued("spent", 70)
        self.assertTrue(self.a.consume(t1.token, first, 1000))
        self.a.close()
        self.a = Authority(b"Z" * 32, b"Y" * 32, self.db)
        _, second = self.issued("new", 31, 1001)
        self.assertEqual(second.disposition, "DENY")

    def test_cost_tamper_cannot_reduce_reservation(self):
        action, issued = self.issued("genuine", 70)
        token = issued.token
        forged = {"claims": {**token["claims"], "cost_cents": 0},
                  "signature": token["signature"]}
        self.assertFalse(self.a.consume(forged, action, 1000))
        self.assertTrue(self.a.consume(token, action, 1000))
        self.assertFalse(self.a.consume(token, action, 1000))

    def test_revocation_disables_pending_grant(self):
        action, issued = self.issued("pending", 30)
        self.a.revoke("person", "buy-demo", b"Y" * 32)
        self.assertFalse(self.a.consume(issued.token, action, 1000))
        self.assertEqual(self.a.decide(action, self.d, 1000).disposition, "DENY")

    def test_invalid_spend_types(self):
        a = self.purchase("item", True)
        self.assertEqual(self.a.decide(a, self.d, 1000).disposition, "DENY")
        a = self.purchase("item", -1)
        self.assertEqual(self.a.decide(a, self.d, 1000).disposition, "DENY")
        a = self.purchase("item", 5001)
        self.assertEqual(self.a.decide(a, self.d, 1000).disposition, "DENY")


if __name__ == "__main__":
    unittest.main()
