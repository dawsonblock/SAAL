import tempfile
import unittest
from pathlib import Path
from saal.memory import PreferenceMemory
from saal.runtime import run_simulated_session
from saal.sim import make_scenario


class MemoryRuntimeTests(unittest.TestCase):
    def test_memory_consent_ttl_update_erase(self):
        with tempfile.TemporaryDirectory() as td:
            m = PreferenceMemory(Path(td)/"mem.sqlite")
            with self.assertRaises(PermissionError):
                m.record_explicit("u", "calendar", "afternoon", consent=False)
            m.record_explicit("u", "calendar", "budget", consent=True, now=1000, ttl=60)
            self.assertEqual(m.recall("u", "calendar", now=1020).probs["budget"], 1)
            self.assertIsNone(m.recall("u", "calendar", now=1070))
            m.record_explicit("u", "calendar", "afternoon", consent=True, now=1100, ttl=60)
            self.assertEqual(m.erase("u"), 1)
            self.assertIsNone(m.recall("u", "calendar", now=1110))
            m.close()

    def test_simulated_full_session_does_not_commit_real_booking(self):
        with tempfile.TemporaryDirectory() as td:
            sc = make_scenario(81013, 0)
            result = run_simulated_session(sc, td, consent_to_memory=True)
            self.assertEqual(result["status"], "SIMULATED_SUCCESS")
            self.assertTrue(result["ledger"]["valid"])
            self.assertTrue(result["simulated_action"]["simulated"])
            self.assertEqual(result["simulated_action"]["tool"], "schedule.propose")

    def test_untrusted_note_causes_escalation_not_execution(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_simulated_session(make_scenario(81013, 2), td)
            self.assertEqual(result["status"], "ESCALATE")
            self.assertTrue(result["ledger"]["valid"])
