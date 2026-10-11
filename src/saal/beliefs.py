"""Discrete explicit hypothesis model; not a claim of calibrated human preference inference."""
from dataclasses import dataclass, field
from typing import Mapping

GOALS = ("afternoon", "few_transfers", "budget", "short_travel")


def normalize(prior: Mapping[str, float]) -> dict[str, float]:
    if set(prior) != set(GOALS):
        raise ValueError("belief distribution has wrong support")
    if any(not isinstance(v, (int, float)) or v < 0 or v != v for v in prior.values()):
        raise ValueError("invalid belief mass")
    total = sum(prior.values())
    if total <= 0:
        raise ValueError("zero total probability")
    return {k: float(prior[k]) / total for k in GOALS}


@dataclass
class BeliefState:
    probs: dict[str, float] = field(default_factory=lambda: {g: 1/len(GOALS) for g in GOALS})
    revision: int = 0
    provenance: str = "unobserved"

    def __post_init__(self) -> None:
        self.probs = normalize(self.probs)

    def observe_model(self, predictions: Mapping[str, float], reliability: float = 0.75) -> None:
        if not 0 <= reliability <= 1:
            raise ValueError("reliability out of range")
        proposal = normalize(predictions)
        self.probs = normalize({g: (1-reliability)*self.probs[g] + reliability*proposal[g]
                                for g in GOALS})
        self.revision += 1
        self.provenance = "inferred:model"

    def explicit_correction(self, goal: str) -> None:
        if goal not in GOALS:
            raise ValueError("unknown preference")
        # Authoritative within the defined user's preferences; NOT an override of safety rules.
        self.probs = {g: float(g == goal) for g in GOALS}
        self.revision += 1
        self.provenance = "user:explicit"

    def uncertainty(self) -> float:
        import math
        return -sum(p*math.log(p) for p in self.probs.values() if p > 0) / math.log(len(GOALS))
