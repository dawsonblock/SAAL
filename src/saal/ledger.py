"""Append-only *logical* hash chain. Storage immutability requires external WORM anchoring."""
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any
from .canonical import encode

GENESIS = "0" * 64


class EvidenceLedger:
    def __init__(self, path: str | Path):
        self._db = sqlite3.connect(str(path), timeout=10)
        self._db.execute("PRAGMA busy_timeout=10000")
        self._db.execute("""CREATE TABLE IF NOT EXISTS events(
            sequence INTEGER PRIMARY KEY, timestamp_ns INTEGER NOT NULL,
            category TEXT NOT NULL, payload BLOB NOT NULL,
            previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL)""")
        self._db.commit()

    @staticmethod
    def hash_event(seq: int, stamp: int, category: str, payload: bytes, prev: str) -> str:
        return hashlib.sha256(encode({"sequence": seq, "timestamp_ns": stamp,
            "category": category, "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "previous_hash": prev})).hexdigest()

    def append(self, category: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(category, str) or not category or len(category) > 100:
            raise ValueError("invalid event category")
        body = encode(payload)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            last = self._db.execute("SELECT sequence,event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
            seq = last[0] + 1 if last else 1
            prev = last[1] if last else GENESIS
            stamp = time.time_ns()
            eh = self.hash_event(seq, stamp, category, body, prev)
            self._db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)",
                             (seq, stamp, category, body, prev, eh))
        return {"sequence": seq, "event_hash": eh}

    def verify(self, expected_final_hash: str | None = None) -> dict[str, Any]:
        expected_previous = GENESIS
        expected_sequence = 1
        for seq, stamp, category, payload, prev, actual in self._db.execute(
                "SELECT * FROM events ORDER BY sequence"):
            if seq != expected_sequence or prev != expected_previous:
                return {"valid": False, "reason": f"broken chain at sequence {seq}"}
            target = self.hash_event(seq, stamp, category, payload, prev)
            if target != actual:
                return {"valid": False, "reason": f"hash mismatch at sequence {seq}"}
            expected_previous = actual
            expected_sequence += 1
        if expected_final_hash is not None and expected_final_hash != expected_previous:
            return {"valid": False, "reason": "external anchor mismatch"}
        return {"valid": True, "count": expected_sequence - 1, "final_hash": expected_previous}

    def close(self) -> None:
        self._db.close()
