"""All-stage SIMULATED orchestrator showing information-flow boundaries.

In real deployment these components must be split into independent processes.
A toy scenario oracle is held by the trusted harness, not passed to the actor.
"""
import time
from pathlib import Path
from .agent import AssistanceActor
from .authority import Authority, SimulatedExecutor
from .canonical import digest
from .ledger import EvidenceLedger
from .memory import PreferenceMemory
from .monitor import inspect
from .sim import Scenario
from .types import Action, Delegation
from .verifier import SyntheticOutcomeVerifier


def run_simulated_session(scenario: Scenario, directory: str | Path,
                          consent_to_memory: bool = False) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    actor = AssistanceActor(infer=True, clarify=True)
    memory = PreferenceMemory(directory / "preferences.sqlite")
    ledger = EvidenceLedger(directory / "evidence.sqlite")
    # In one-process demo, secrets are intentionally ephemeral; NOT a security boundary.
    import secrets
    authority = Authority(secrets.token_bytes(32), secrets.token_bytes(32),
                          directory / "replay.sqlite")
    try:
        visible = scenario.public_view()
        prior = memory.recall("demo_user", "appointments")
        candidate = actor.propose(visible, prior)
        asked = candidate["kind"] == "ASK"
        explicit = None
        if asked:
            # Synthetic human supplies a truthful answer; this is NOT an LLM hallucination.
            explicit = scenario.answer_clarification()
            candidate = actor.after_answer(visible, explicit)
        elif visible.get("correction"):
            explicit = visible["correction"]
        ledger.append("user_intent", {"task_id": scenario.id, "asked": asked,
                                      "has_explicit_answer": bool(explicit),
                                      "memory_consent": consent_to_memory})
        if candidate["kind"] != "PROPOSE":
            ledger.append("abstain", {"reason": candidate.get("reason", "no plan")})
            return {"status": "ABSTAIN", "ledger": ledger.verify()}
        action = Action("schedule.propose", {"slot_id": candidate["slot_id"], "cost_cents": 0},
                        "demo_user", scenario.id)
        ledger.append("proposal", {"action_hash": digest(action.as_dict())})
        finding = inspect(action, visible.get("untrusted_note", ""))
        if finding.suspicious:
            ledger.append("monitor_alert", {"indicators": list(finding.indicators)})
            return {"status": "ESCALATE", "alert": list(finding.indicators),
                    "ledger": ledger.verify()}
        delegation = Delegation("demo_user", scenario.id, "propose non-binding appointment",
                                ("schedule.propose",), int(time.time()) + 120)
        decision = authority.decide(action, delegation)
        ledger.append("authorization", {"disposition": decision.disposition,
                                        "policy_hash": authority.policy_hash})
        if decision.disposition != "ALLOW":
            return {"status": decision.disposition, "ledger": ledger.verify()}
        execution = SimulatedExecutor(authority).execute(action, decision.token)
        outcome = SyntheticOutcomeVerifier().verify(scenario, candidate["slot_id"])
        ledger.append("verified_outcome", {"source": "synthetic_oracle",
                      "action_hash": digest(action.as_dict()), "utility": outcome["utility"],
                      "regret": outcome["regret"], "constraint_violation": outcome["violation"]})
        if consent_to_memory and explicit is not None:
            memory.record_explicit("demo_user", "appointments", explicit, consent=True)
        return {"status": "SIMULATED_SUCCESS", "asked": asked, "simulated_action": execution,
                "verified_synthetic_outcome": outcome, "memory_saved": bool(consent_to_memory and explicit),
                "ledger": ledger.verify()}
    finally:
        authority.close()
        ledger.close()
        memory.close()
