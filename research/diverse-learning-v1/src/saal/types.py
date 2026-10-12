"""Explicit wire objects. Untrusted action risk labels never determine authorization."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Delegation:
    principal: str
    task_id: str
    goal: str
    allowed_tools: tuple[str, ...]
    expires_at: int
    max_cost_cents: int = 0
    may_affect_others: bool = False
    max_actions: int = 100
    def __post_init__(self) -> None:
        if not self.principal or not self.task_id or not self.goal:
            raise ValueError("principal/task_id/goal required")
        if (self.expires_at <= 0 or self.max_cost_cents < 0 or
                type(self.max_actions) is not int or not 1 <= self.max_actions <= 1_000_000):
            raise ValueError("invalid delegation limits")
        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("duplicate allowed tools")


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict[str, Any]
    principal: str
    task_id: str
    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args, "principal": self.principal,
                "task_id": self.task_id}


@dataclass(frozen=True)
class Decision:
    disposition: str  # ALLOW, APPROVAL_REQUIRED, DENY
    reasons: tuple[str, ...] = ()
    risk: str = "UNKNOWN"
    token: dict[str, Any] | None = None


@dataclass(frozen=True)
class Outcome:
    action_hash: str
    success: bool
    evidence_kind: str
    evidence: dict[str, Any] = field(default_factory=dict)
    uncertainty: str = "unknown"
    verifier_id: str = ""
