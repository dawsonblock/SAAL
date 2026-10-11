"""Simple secondary anomaly signal; NEVER acts as a permission grant."""
from dataclasses import dataclass
from .types import Action

SUSPICIOUS = ("ignore previous", "bypass policy", "ignore restrictions", "exfiltrate", "disable logging")


@dataclass(frozen=True)
class MonitorFinding:
    suspicious: bool
    indicators: tuple[str, ...]
    recommendation: str
    # Monitors never issue capability tokens.


def inspect(action: Action, untrusted_context: str = "") -> MonitorFinding:
    # Toy heuristic: production review needs its own independent policy+test set.
    signals = tuple(word for word in SUSPICIOUS if word in untrusted_context.lower())
    if action.tool in ("model.promote", "system.change_policy"):
        signals = signals + ("attempted control-plane modification",)
    return MonitorFinding(bool(signals), signals,
                          "BLOCK_OR_ESCALATE" if signals else "NO_ADDITIONAL_FINDINGS")
