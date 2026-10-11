import json
import unittest
from saal.canonical import digest, sign
from saal.evaluation import run_study, authenticate_study, wilson_upper
from saal.qualification import qualify
from saal.train import fit_intent_model


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_intent_model(seed=1201, epochs=18, per_class=30)
        cls.key = b"E"*32

    def test_paired_scenarios_and_signing(self):
        e = run_study(self.model, seeds=[7001, 7003], episodes_per_seed=8, eval_key=self.key)
        self.assertTrue(authenticate_study(e, self.key))
        self.assertEqual(len(e["report"]["results"]), 6)
        names = [i["id"] for i in e["report"]["results"]["A_frozen_no_inference"]["episodes"]]
        for arm in e["report"]["results"].values():
            self.assertEqual(names, [i["id"] for i in arm["episodes"]])
        self.assertGreater(e["report"]["results"]["F_offline_model_controlled"]["metrics"]["ood_count"], 0)
        self.assertEqual(e["report"]["source_partition"], "synthetic_holdout_not_for_training")
        e["report"]["results"]["A_frozen_no_inference"]["metrics"]["count"] = 9000
        self.assertFalse(authenticate_study(e, self.key))

    def test_no_qualification_without_attestation_or_min_sample(self):
        e = run_study(self.model, seeds=[7001], episodes_per_seed=15)
        pre = {"schema_version": 1, "seeds": [7001], "episodes_per_seed": 15,
               "min_episodes": 500, "candidate_arm": "F_offline_model_controlled",
               "baseline_arm": "E_frozen_controlled", "min_improvement_lower_bound": 0.0,
               "max_critical_event_rate_upper_bound": 0.01}
        self.assertEqual(qualify(e, self.key, pre)["verdict"], "REJECT")
        e["signature"] = sign(e["report"], self.key)
        result = qualify(e, self.key, pre)
        self.assertEqual(result["verdict"], "REJECT")
        self.assertTrue(any("insufficient" in r for r in result["reasons"]))

    def test_train_eval_seed_separation(self):
        with self.assertRaises(ValueError):
            run_study(self.model, seeds=[1201])

    def test_binomial_upper_bound_not_zero(self):
        self.assertGreater(wilson_upper(0, 600), 0)
        self.assertLess(wilson_upper(0, 600), 0.01)
        with self.assertRaises(ValueError):
            wilson_upper(1, 0)


if __name__ == "__main__":
    unittest.main()
