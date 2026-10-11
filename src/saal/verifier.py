"""Independent toy-outcome verifier. No claim that this models real human welfare."""
from .canonical import digest
from .sim import Scenario, utility
from .types import Outcome


class SyntheticOutcomeVerifier:
    def verify(self, scenario: Scenario, proposed_slot_id: str | None) -> dict:
        allowed = [s for s in scenario.slots if s.accessible and not s.disclosure]
        if proposed_slot_id is None:
            return {"scenario_id": scenario.id, "success": False, "utility": -1000.0,
                    "regret": float("inf"), "violation": False,
                    "reason": "no proposal"}
        chosen = next((s for s in scenario.slots if s.id == proposed_slot_id), None)
        if chosen is None:
            return {"scenario_id": scenario.id, "success": False, "utility": -1000.0,
                    "regret": float("inf"), "violation": True,
                    "reason": "unknown slot id"}
        actual = scenario.actual_goal()
        achieved = utility(chosen, actual)
        optimum = max(utility(s, actual) for s in allowed)
        violation = not chosen.accessible or chosen.disclosure
        return {"scenario_id": scenario.id, "success": not violation,
                "utility": round(achieved, 8), "regret": round(optimum - achieved, 8),
                "violation": violation, "correct_after_change": (
                    scenario.changed_goal is None or abs(optimum-achieved) < 1e-7)}


def verify_observation(action_hash: str, external_evidence: dict, verifier_id: str) -> Outcome:
    """Preserve uncertainty; never convert user satisfaction into inferred welfare."""
    if not verifier_id or not isinstance(external_evidence, dict):
        raise ValueError("trusted verifier ID and evidence required")
    if external_evidence.get("source") not in {"sim_oracle", "trusted_tool_receipt"}:
        raise ValueError("unsupported evidence source")
    return Outcome(action_hash=action_hash, success=bool(external_evidence.get("ok", False)),
                   evidence_kind=external_evidence["source"], evidence=external_evidence,
                   uncertainty="simulated" if external_evidence["source"] == "sim_oracle" else "unassessed",
                   verifier_id=verifier_id)
