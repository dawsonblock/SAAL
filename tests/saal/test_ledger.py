import sqlite3
import tempfile
import unittest
from pathlib import Path
from saal.canonical import digest, sign, verify
from saal.ledger import EvidenceLedger


class LedgerTests(unittest.TestCase):
    def test_chain_and_external_anchor(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"ledger.sqlite"
            db = EvidenceLedger(path)
            db.append("task", {"success": True})
            db.append("verification", {"source": "sim_oracle"})
            anchor = db.verify()["final_hash"]
            self.assertTrue(db.verify(anchor)["valid"])
            self.assertFalse(db.verify("f"*64)["valid"])
            db._db.execute("UPDATE events SET category='corrupt' WHERE sequence=1")
            db._db.commit()
            self.assertFalse(db.verify()["valid"])
            db.close()

    def test_signature_and_serialization(self):
        obj = {"z": 1, "a": [True, None]}
        self.assertEqual(digest(obj), digest({"a": [True, None], "z": 1}))
        sig = sign(obj, b"z"*32)
        self.assertTrue(verify(obj, sig, b"z"*32))
        self.assertFalse(verify({"z": 2}, sig, b"z"*32))
        with self.assertRaises(ValueError):
            digest({"nan": float("nan")})


if __name__ == "__main__":
    unittest.main()
