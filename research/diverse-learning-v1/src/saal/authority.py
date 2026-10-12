"""Deny-by-default external authorization. Deploy isolated from untrusted actor."""
import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from functools import wraps
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import digest, sign, verify
from .types import Action, Decision, Delegation

RISK_ORDER = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "BLOCKED": 4}


def serialized(method):
    @wraps(method)
    def _wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return _wrapped


@dataclass(frozen=True)
class ToolRule:
    risk: str
    require_approval: bool
    allow_external_effects: bool
    max_cost_cents: int = 0
    def __post_init__(self) -> None:
        if self.risk not in RISK_ORDER:
            raise ValueError("unknown risk")
        if self.max_cost_cents < 0:
            raise ValueError("negative tool budget")


DEFAULT_RULES = {
    "schedule.read": ToolRule("LOW", False, False),
    "schedule.propose": ToolRule("LOW", False, False),
    "schedule.commit": ToolRule("HIGH", True, True, 0),
    "filesystem.delete": ToolRule("BLOCKED", True, True),
    "system.change_policy": ToolRule("BLOCKED", True, True),
    "model.promote": ToolRule("BLOCKED", True, True),
    # Synthetic ONLY: there is no external adapter for this tool.
    "sim.purchase": ToolRule("HIGH", True, True, 5000),
}


class Authority:
    """Policy authority; key/approval credential must not be visible to the actor.

    This library class is a reference: in production run it under a distinct UID,
    disallow actor access to its files, and put an authenticated interface in front.
    """

    def __init__(self, key: bytes, approvals_key: bytes, replay_db: str | Path = ":memory:",
                 rules: dict[str, ToolRule] | None = None):
        if len(key) < 32 or len(approvals_key) < 32:
            raise ValueError("independent secrets must have >=32 bytes")
        if hmac.compare_digest(key, approvals_key):
            raise ValueError("signing key and approval credential must be distinct")
        self._key = key
        self._approval_key = approvals_key
        self.rules = rules if rules is not None else dict(DEFAULT_RULES)
        self._lock = threading.RLock()
        # UnixStreamServer and a trusted admin interface may run on distinct threads.
        self._db = sqlite3.connect(str(replay_db), check_same_thread=False)
        # v3 issue/consume lifecycle also binds the per-task action-count cap.
        # Older v2 replay state must be migrated offline; fail closed on schemas.
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(tokens)")}
        if columns and columns != {"nonce", "used", "action_hash", "principal", "task_id",
                                   "cost_cents", "limit_cents", "action_limit", "expires_at"}:
            raise RuntimeError("legacy replay DB schema: migrate offline, fail closed")
        self._db.execute("""CREATE TABLE IF NOT EXISTS tokens(
            nonce TEXT PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0,
            action_hash TEXT NOT NULL, principal TEXT NOT NULL, task_id TEXT NOT NULL,
            cost_cents INTEGER NOT NULL, limit_cents INTEGER NOT NULL,
            action_limit INTEGER NOT NULL, expires_at INTEGER NOT NULL)""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS approvals(
            action_hash TEXT PRIMARY KEY, principal TEXT NOT NULL, at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0)""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS revoked(
            principal TEXT NOT NULL, task_id TEXT NOT NULL,
            PRIMARY KEY(principal,task_id))""")
        self._db.commit()
        self.policy_hash = digest({k: vars(v) for k, v in sorted(self.rules.items())})

    @serialized
    def approve(self, action: Action, human_credential: bytes, now: int | None = None) -> None:
        """Trusted UI ONLY. Credential is NOT to be offered to the model."""
        if not hmac.compare_digest(human_credential, self._approval_key):
            raise PermissionError("invalid approval credential")
        now = int(time.time()) if now is None else int(now)
        self._db.execute("INSERT OR REPLACE INTO approvals VALUES(?,?,?,?,0)",
                         (digest(action.as_dict()), action.principal, now, now + 300))
        self._db.commit()

    @serialized
    def revoke(self, principal: str, task_id: str, human_credential: bytes) -> None:
        """Trusted-host operation: invalidate both pending and already issued grants."""
        if not hmac.compare_digest(human_credential, self._approval_key):
            raise PermissionError("invalid revocation credential")
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO revoked VALUES(?,?)", (principal, task_id))

    def _is_revoked(self, principal: str, task_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM revoked WHERE principal=? AND task_id=?",
                                (principal, task_id)).fetchone() is not None

    @serialized
    def decide(self, action: Action, delegation: Delegation, now: int | None = None) -> Decision:
        now = int(time.time()) if now is None else int(now)
        rule = self.rules.get(action.tool)
        if rule is None or rule.risk == "BLOCKED":
            return Decision("DENY", ("unknown or blocked tool",), "BLOCKED")
        if action.principal != delegation.principal or action.task_id != delegation.task_id:
            return Decision("DENY", ("delegation identity mismatch",), rule.risk)
        if self._is_revoked(action.principal, action.task_id):
            return Decision("DENY", ("delegation revoked",), rule.risk)
        if now >= delegation.expires_at:
            return Decision("DENY", ("delegation expired",), rule.risk)
        if action.tool not in delegation.allowed_tools:
            return Decision("DENY", ("tool outside delegated scope",), rule.risk)
        # Fail closed on malformed arguments: only recognized schemas may be authorized.
        if not validate_args(action):
            return Decision("DENY", ("invalid action schema",), rule.risk)
        cost = action.args.get("cost_cents", 0)
        if type(cost) is not int or cost < 0 or cost > min(delegation.max_cost_cents, rule.max_cost_cents):
            return Decision("DENY", ("cost exceeds budget",), rule.risk)
        if rule.allow_external_effects and not delegation.may_affect_others:
            return Decision("DENY", ("delegation prohibits external effects",), rule.risk)
        ah = digest(action.as_dict())
        if rule.require_approval:
            found = self._db.execute("""SELECT 1 FROM approvals WHERE action_hash=?
                AND principal=? AND used=0 AND at<=? AND expires_at>?""",
                                     (ah, action.principal, now, now)).fetchone()
            if not found:
                return Decision("APPROVAL_REQUIRED", ("trusted human approval required",), rule.risk)
        # Ticket issuance RESERVES both action count and monetary budget.
        # Repeated zero-cost calls cannot bypass a delegation's total action cap.
        # Ticket issuance RESERVES budget. Repeated calls cannot bypass the total
        # task cap by individually staying below a per-action limit. Expired,
        # *unconsumed* grants cease to reserve funds; consumed grants remain spent.
        # The true effect must still be implemented by a separate trusted executor.
        payload = {"version": 3, "action_hash": ah, "principal": action.principal,
                   "task_id": action.task_id, "nonce": secrets.token_hex(24),
                   "iat": now, "exp": min(now + 60, delegation.expires_at),
                   "cost_cents": cost, "limit_cents": delegation.max_cost_cents,
                   "action_limit": delegation.max_actions,
                   "policy_hash": self.policy_hash}
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            budget_used = self._db.execute("""SELECT COALESCE(SUM(cost_cents),0)
                FROM tokens WHERE principal=? AND task_id=?
                AND (used=1 OR expires_at>?)""",
                (action.principal, action.task_id, now)).fetchone()[0]
            if cost > delegation.max_cost_cents - budget_used:
                return Decision("DENY", ("cumulative task budget exhausted",), rule.risk)
            action_count = self._db.execute("""SELECT COUNT(*) FROM tokens
                WHERE principal=? AND task_id=? AND (used=1 OR expires_at>?)""",
                (action.principal, action.task_id, now)).fetchone()[0]
            if action_count >= delegation.max_actions:
                return Decision("DENY", ("cumulative task action limit exhausted",), rule.risk)
            if rule.require_approval:
                used = self._db.execute("""UPDATE approvals SET used=1 WHERE action_hash=?
                    AND principal=? AND used=0 AND at<=? AND expires_at>?""",
                    (ah, action.principal, now, now)).rowcount
                if used != 1:
                    return Decision("APPROVAL_REQUIRED", ("approval already used",), rule.risk)
            self._db.execute("""INSERT INTO tokens
                (nonce,used,action_hash,principal,task_id,cost_cents,limit_cents,action_limit,expires_at)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (payload["nonce"], 0, ah, action.principal, action.task_id, cost,
                 delegation.max_cost_cents, delegation.max_actions, payload["exp"]))
        return Decision("ALLOW", (), rule.risk,
                        {"claims": payload, "signature": sign(payload, self._key)})

    @serialized
    def consume(self, token: dict[str, Any], action: Action, now: int | None = None) -> bool:
        """Only a trusted executor may call; actor cannot execute via untrusted gateway."""
        now = int(time.time()) if now is None else int(now)
        try:
            payload = token["claims"]
            if not verify(payload, token["signature"], self._key):
                return False
            if payload["version"] != 3 or payload["policy_hash"] != self.policy_hash:
                return False
            if payload["action_hash"] != digest(action.as_dict()):
                return False
            if payload["principal"] != action.principal or payload["task_id"] != action.task_id:
                return False
            if type(payload["cost_cents"]) is not int or payload["cost_cents"] < 0:
                return False
            if type(payload["action_limit"]) is not int or payload["action_limit"] < 1:
                return False
            if now < payload["iat"] or now >= payload["exp"]:
                return False
            with self._db:
                if self._is_revoked(action.principal, action.task_id):
                    return False
                cur = self._db.execute("""UPDATE tokens SET used=1 WHERE nonce=?
                    AND used=0 AND action_hash=? AND principal=? AND task_id=?
                    AND cost_cents=? AND limit_cents=? AND action_limit=? AND expires_at=?""",
                    (payload["nonce"], payload["action_hash"], action.principal,
                     action.task_id, payload["cost_cents"], payload["limit_cents"],
                     payload["action_limit"], payload["exp"]))
                return cur.rowcount == 1
        except (KeyError, TypeError, ValueError, sqlite3.Error):
            return False

    @serialized
    def close(self) -> None:
        self._db.close()


def validate_args(action: Action) -> bool:
    """Whitelisted typed schemas. No command paths/URLs from model to executor."""
    a = action.args
    if not isinstance(a, dict):
        return False
    if action.tool == "schedule.read":
        return set(a) == {"calendar_id"} and isinstance(a["calendar_id"], str) and 0 < len(a["calendar_id"]) <= 80
    if action.tool == "sim.purchase":
        return (set(a) == {"item_id", "cost_cents"} and
                isinstance(a["item_id"], str) and 0 < len(a["item_id"]) <= 80 and
                type(a["cost_cents"]) is int and a["cost_cents"] >= 0)
    if action.tool in {"schedule.propose", "schedule.commit"}:
        return (set(a) == {"slot_id", "cost_cents"} and
                isinstance(a["slot_id"], str) and 0 < len(a["slot_id"]) <= 80 and
                type(a["cost_cents"]) is int and a["cost_cents"] >= 0)
    return False


class SimulatedExecutor:
    """Safe deterministic test adapter: no network, calendar access, or external writes."""

    def __init__(self, authority: Authority):
        self.authority = authority
        self.events: list[dict[str, Any]] = []

    def execute(self, action: Action, token: dict[str, Any], now: int | None = None) -> dict[str, Any]:
        if not self.authority.consume(token, action, now):
            raise PermissionError("invalid, expired, or replayed capability")
        event = {"tool": action.tool, "args": action.args, "simulated": True}
        self.events.append(event)
        return event
