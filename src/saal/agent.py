"""Actor API receives only public scenario view; no evaluator, scores or secret keys."""
import re
from .beliefs import BeliefState, GOALS
from .sim import Slot, utility
from .train import IntentClassifier


def parse_cue(text: str) -> dict[str, float]:
    cue = text.lower()
    keywords = {
        "afternoon": ("afternoon", "later", "evening", "morning", "late"),
        "few_transfers": ("transfer", "changes", "direct", "buses"),
        "budget": ("cost", "cheap", "afford", "money", "budget"),
        "short_travel": ("quick", "short", "fast", "travel time", "journey"),
    }
    hits = [goal for goal, words in keywords.items() if any(w in cue for w in words)]
    return {g: (0.85 if g == hits[0] else 0.05) if hits else .25 for g in GOALS}


def valid_slots(view: dict) -> list[Slot]:
    slots = [Slot(**slot) for slot in view["slots"]]
    return [s for s in slots if s.accessible and not s.disclosure]


def expected_value(slot: Slot, belief: BeliefState) -> float:
    return sum(p * utility(slot, goal) for goal, p in belief.probs.items())


class AssistanceActor:
    def __init__(self, infer: bool, clarify: bool, model: IntentClassifier | None = None,
                 ask_cost: float = 0.2):
        self.infer = infer
        self.clarify = clarify
        self.model = model
        self.ask_cost = ask_cost

    def propose(self, public_view: dict, prior: BeliefState | None = None) -> dict:
        """No hidden utility or simulator callback. Only opaque ASK, PROPOSE, ABSTAIN."""
        options = valid_slots(public_view)
        if not options:
            return {"kind": "ABSTAIN", "reason": "no accessible and non-disclosing slot"}
        state = BeliefState(dict(prior.probs), prior.revision, prior.provenance) if prior else BeliefState()
        if self.infer:
            predictions = self.model.predict(public_view["utterance"]) if self.model else parse_cue(public_view["utterance"])
            state.observe_model(predictions, reliability=0.85)
        if public_view.get("correction"):
            state.explicit_correction(public_view["correction"])
        if not self.infer:
            # Baseline intentionally fails to infer subjective user preference.
            selected = min(options, key=lambda s: (s.minutes, s.cost, s.id))
            return {"kind": "PROPOSE", "slot_id": selected.id, "asked": False}
        chosen = max(options, key=lambda s: (expected_value(s, state), s.id))
        current = expected_value(chosen, state)
        clairvoyant = sum(p * max(utility(s, goal) for s in options)
                            for goal, p in state.probs.items())
        if self.clarify and clairvoyant - current > self.ask_cost and state.uncertainty() > 0.04:
            return {"kind": "ASK", "question": "Which matters most: time of day, transfers, cost, or trip length?",
                    "current_choice": chosen.id}
        return {"kind": "PROPOSE", "slot_id": chosen.id, "asked": False}

    def after_answer(self, public_view: dict, explicit_goal: str) -> dict:
        """Trusted explicit user response supplied through interaction manager."""
        if explicit_goal not in GOALS:
            return {"kind": "ABSTAIN", "reason": "ambiguous clarification response"}
        options = valid_slots(public_view)
        state = BeliefState()
        state.explicit_correction(explicit_goal)
        chosen = max(options, key=lambda s: (expected_value(s, state), s.id))
        return {"kind": "PROPOSE", "slot_id": chosen.id, "asked": True}
