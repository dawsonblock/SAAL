import unittest
from saal.beliefs import BeliefState, GOALS, normalize
from saal.agent import AssistanceActor
from saal.sim import make_scenario, utility
from saal.train import dataset, fit_intent_model


class BeliefTests(unittest.TestCase):
    def test_correction_overrides_inference(self):
        b = BeliefState()
        b.observe_model({g: (0.97 if g == "budget" else 0.01) for g in GOALS})
        self.assertEqual(max(b.probs, key=b.probs.get), "budget")
        b.explicit_correction("afternoon")
        self.assertEqual(b.probs["afternoon"], 1)
        self.assertEqual(b.uncertainty(), 0)
        self.assertEqual(b.provenance, "user:explicit")

    def test_normalization_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            normalize({g: -1 for g in GOALS})
        with self.assertRaises(ValueError):
            normalize({"anything": 1})

    def test_synthetic_scenario_hides_goals(self):
        sc = make_scenario(101, 8)
        visible = sc.public_view()
        self.assertNotIn("hidden_goal", visible)
        self.assertNotIn("utility", str(visible))
        self.assertNotIn("oracle", visible)

    def test_offline_learning_on_training_distribution(self):
        model = fit_intent_model(seed=1201, epochs=35, per_class=30)
        examples = dataset(seed=29, per_class=12)
        accuracy = sum(max(model.predict(text), key=model.predict(text).get) == goal
                       for text, goal in examples) / len(examples)
        self.assertGreater(accuracy, 0.9)

    def test_clarification_returns_a_preference_matched_choice(self):
        sc = make_scenario(120, 1)
        actor = AssistanceActor(True, True)
        view = sc.public_view()
        result = actor.after_answer(view, sc.answer_clarification())
        options = [s for s in sc.slots if s.accessible and not s.disclosure]
        optimum = max(utility(s, sc.actual_goal()) for s in options)
        selected = next(s for s in options if s.id == result["slot_id"])
        self.assertAlmostEqual(utility(selected, sc.actual_goal()), optimum)


if __name__ == "__main__":
    unittest.main()
