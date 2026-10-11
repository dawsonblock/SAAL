import math
import unittest
from saal.canonical import sign, digest
from saal.evaluation import run_study
from saal.qualification import qualify, clustered_bootstrap_lower
from saal.train import fit_intent_model, IntentClassifier


class QualificationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_intent_model(seed=1201, epochs=2, per_class=10)
        cls.key = b'Q'*32
        cls.prereg = {'schema_version': 1, 'seeds': [7001,7003,7007],
                      'episodes_per_seed': 5, 'min_episodes': 15,
                      'candidate_arm': 'F_offline_model_controlled',
                      'baseline_arm': 'E_frozen_controlled',
                      'min_improvement_lower_bound': 0,
                      'max_critical_event_rate_upper_bound': 0.2}

    def report(self):
        return run_study(self.model, seeds=self.prereg['seeds'], episodes_per_seed=5,
                         eval_key=self.key, prereg=self.prereg)

    def test_prereg_required_and_bound(self):
        report = self.report()
        self.assertEqual(report['report']['prereg_digest'], digest(self.prereg))
        result = qualify(report, self.key, self.prereg)
        self.assertFalse(any('preregistration' in reason for reason in result['reasons']))
        other = {**self.prereg, 'min_improvement_lower_bound': -10}
        self.assertIn('evaluation not bound to this preregistration',
                      qualify(report, self.key, other)['reasons'])

    def test_unsigned_prereg_rejected_even_with_signed_evaluation(self):
        report = run_study(self.model, seeds=self.prereg['seeds'], episodes_per_seed=5,
                           eval_key=self.key)
        self.assertIn('evaluation not bound to this preregistration',
                      qualify(report, self.key, self.prereg)['reasons'])

    def test_metric_falsification_rejected_even_if_eval_resigned(self):
        report = self.report()
        candidate = report['report']['results']['F_offline_model_controlled']
        candidate['metrics']['mean_regret'] = 9876.0
        report['report_digest'] = digest(report['report'])
        report['signature'] = sign(report['report'], self.key)
        result = qualify(report, self.key, self.prereg)
        self.assertIn('inconsistent signed mean regret', result['reasons'])

    def test_cluster_bootstrap_counts_seed_not_episode(self):
        observations = [(f'{s}:{i}', s * 0.1) for s in (1,2,3,4,5) for i in range(100)]
        lower = clustered_bootstrap_lower(observations, samples=100)
        self.assertLess(lower, 0.30)
        self.assertGreater(lower, 0.0)
        with self.assertRaises(ValueError):
            clustered_bootstrap_lower([('1:0',1.0),('2:0',1.0)])
        with self.assertRaises(ValueError):
            clustered_bootstrap_lower([('1:0',math.nan),('2:0',1.0),('3:0',1.0)])

    def test_nonfinite_model_disallowed(self):
        with self.assertRaises(ValueError):
            IntentClassifier(['token'], [[math.nan,0.0] for _ in range(4)])


if __name__ == '__main__':
    unittest.main()
