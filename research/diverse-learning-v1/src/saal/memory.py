"""Explicit-consent, minimal preference memory; never stores raw chat by default."""
import sqlite3
import time
from pathlib import Path
from .beliefs import BeliefState, GOALS

MAX_TTL_SECONDS = 30 * 24 * 3600


class PreferenceMemory:
    def __init__(self, db: str | Path):
        self._db = sqlite3.connect(str(db))
        self._db.execute("""CREATE TABLE IF NOT EXISTS preferences(
            principal TEXT NOT NULL, domain TEXT NOT NULL, preference TEXT NOT NULL,
            source TEXT NOT NULL, expires_at INTEGER NOT NULL, revised_at INTEGER NOT NULL,
            PRIMARY KEY(principal,domain))""")
        self._db.commit()

    def record_explicit(self, principal: str, domain: str, preference: str,
                        consent: bool, ttl: int = 86400, now: int | None = None) -> None:
        if consent is not True:
            raise PermissionError("consent required to persist preference")
        if preference not in GOALS or not principal or not domain:
            raise ValueError("invalid preference memory")
        if len(principal) > 128 or len(domain) > 80 or ttl < 1 or ttl > MAX_TTL_SECONDS:
            raise ValueError("invalid principal, domain, or retention limit")
        now = int(time.time()) if now is None else int(now)
        with self._db:
            self._db.execute("""INSERT INTO preferences VALUES(?,?,?,?,?,?)
              ON CONFLICT(principal,domain) DO UPDATE SET preference=excluded.preference,
                source=excluded.source, expires_at=excluded.expires_at,
                revised_at=excluded.revised_at""",
                (principal, domain, preference, "explicit_authenticated_user", now+ttl, now))

    def recall(self, principal: str, domain: str, now: int | None = None) -> BeliefState | None:
        now = int(time.time()) if now is None else int(now)
        record = self._db.execute("""SELECT preference FROM preferences
            WHERE principal=? AND domain=? AND expires_at>?""",
            (principal, domain, now)).fetchone()
        if record is None:
            return None
        state = BeliefState()
        state.explicit_correction(record[0])
        state.provenance = "consented_memory:explicit_user"
        return state

    def erase(self, principal: str, domain: str | None = None) -> int:
        with self._db:
            if domain is None:
                cur = self._db.execute("DELETE FROM preferences WHERE principal=?", (principal,))
            else:
                cur = self._db.execute("DELETE FROM preferences WHERE principal=? AND domain=?",
                                       (principal, domain))
            return cur.rowcount

    def close(self):
        self._db.close()
