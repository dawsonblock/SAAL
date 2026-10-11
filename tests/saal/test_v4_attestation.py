"""Adversarial evaluator/qualifier trust-separation regressions for v4."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from saal.canonical import digest
from saal.evaluator_attestation import (
    EvaluatorPublicKey, create_evaluator_keys, sign_evaluation,
)
from saal.evaluation import run_study, authenticate_study
from saal.implementation import implementation_digest
from saal.prereg import private_prereg
from saal.qualification import qualify
from saal.replay import audit_replay
from saal.train import fit_intent_model


class EvaluatorV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_intent_model(seed=1201, epochs=1, per_class=4)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.private = self.root / 'eval.priv'
        self.public = self.root / 'eval.pub'
        create_evaluator_keys(self.private, self.public)
        self.verifier = EvaluatorPublicKey(self.public)
        self.prereg = private_prereg(10, 30, self.public)
        # Small but valid holdout generated privately for fast regression tests.
        self.prereg['episodes_per_seed'] = 2
        self.prereg['min_episodes'] = 20
        self.prereg['max_critical_event_rate_upper_bound'] = 1.0
        self.prereg['min_improvement_lower_bound'] = -100.0
        self.report = run_study(self.model, seeds=self.prereg['seeds'],
                                episodes_per_seed=2, prereg=self.prereg,
                                eval_private_key_file=str(self.private))
        self.external_commitment = digest(self.prereg)

    def check(self, report=None, prereg=None, verifier=None, commitment=None, model=None):
        return qualify(report if report is not None else self.report,
                       verifier if verifier is not None else self.verifier,
                       prereg if prereg is not None else self.prereg,
                       model=self.model if model is None else model,
                       expected_prereg_digest=(self.external_commitment if commitment is None
                                               else commitment))

    def test_ed25519_attestation_and_replay_and_separate_key(self):
        self.assertTrue(authenticate_study(self.report, self.verifier))
        self.assertTrue(audit_replay(self.report, self.prereg, self.model, self.verifier)['valid'])
        q = self.check()
        self.assertEqual(q['replay_audit']['valid'], True)
        self.assertFalse(any('evaluator' in x or 'commitment' in x for x in q['reasons']))
        self.assertEqual(self.report['report']['implementation_digest'], implementation_digest())
        self.assertEqual(len(self.verifier.raw()), 32)
        self.assertEqual(self.prereg['evaluator_public_key_sha256'], self.verifier.fingerprint())
        self.assertFalse((self.root/'qualifier-private-key').exists())

    def test_wrong_public_key_rejected(self):
        altpriv, altpub = self.root/'other.priv', self.root/'other.pub'
        create_evaluator_keys(altpriv, altpub)
        verifier = EvaluatorPublicKey(altpub)
        self.assertFalse(authenticate_study(self.report, verifier))
        self.assertEqual(self.check(verifier=verifier)['verdict'], 'REJECT')

    def test_altered_signature_or_report_rejected(self):
        forged = copy.deepcopy(self.report)
        forged['attestation']['signature_b64'] = 'A'*88
        self.assertFalse(authenticate_study(forged, self.verifier))
        forged = copy.deepcopy(self.report)
        forged['report']['results']['A_frozen_no_inference']['metrics']['count'] = 999
        self.assertFalse(authenticate_study(forged, self.verifier))

    def test_swapped_evaluator_key_cannot_be_substituted(self):
        altpriv, altpub = self.root/'other.priv', self.root/'other.pub'
        create_evaluator_keys(altpriv, altpub)
        forged = copy.deepcopy(self.report)
        forged['attestation'] = sign_evaluation(forged['report'], altpriv)
        self.assertTrue(authenticate_study(forged, EvaluatorPublicKey(altpub)))
        qualification = self.check(report=forged, verifier=EvaluatorPublicKey(altpub))
        self.assertEqual(qualification['verdict'], 'REJECT')
        self.assertTrue(any('trust anchor' in x for x in qualification['reasons']))

    def test_external_commitment_required_and_checked(self):
        self.assertEqual(self.check(commitment='0'*64)['verdict'], 'REJECT')
        q = qualify(self.report, self.verifier, self.prereg, model=self.model)
        self.assertEqual(q['verdict'], 'REJECT')
        self.assertTrue(any('externally witnessed' in x for x in q['reasons']))

    def test_hmac_downgrade_fails(self):
        legacy = run_study(self.model, seeds=self.prereg['seeds'], episodes_per_seed=2,
                           eval_key=b'H'*32, prereg={k:v for k,v in self.prereg.items()
                                                     if k not in ('signature_scheme',
                                                                  'evaluator_public_key_sha256')})
        self.assertFalse(authenticate_study(legacy, self.verifier))
        q = self.check(report=legacy)
        self.assertEqual(q['verdict'], 'REJECT')

    def test_corrupted_signed_report_caught_by_independent_replay(self):
        forged = copy.deepcopy(self.report)
        forged['report']['results']['E_frozen_controlled']['episodes'][0]['true_utility'] += 1
        forged['report']['results']['E_frozen_controlled']['metrics']['mean_true_utility'] += 1/20
        forged['report_digest'] = digest(forged['report'])
        forged['attestation'] = sign_evaluation(forged['report'], self.private)
        self.assertTrue(authenticate_study(forged, self.verifier))
        q = self.check(report=forged)
        self.assertEqual(q['verdict'], 'REJECT')
        self.assertTrue(any('replay mismatch' in x for x in q['reasons']))

    def test_unsigned_implementation_change_fails(self):
        forged = copy.deepcopy(self.report)
        forged['report']['implementation_digest'] = '0'*64
        forged['report_digest'] = digest(forged['report'])
        forged['attestation'] = sign_evaluation(forged['report'], self.private)
        self.assertTrue(authenticate_study(forged, self.verifier))
        q = self.check(report=forged)
        self.assertEqual(q['verdict'], 'REJECT')
        self.assertTrue(any('implementation differs' in x for x in q['reasons']))

    def test_missing_replay_model_rejected(self):
        q = qualify(self.report, self.verifier, self.prereg,
                    expected_prereg_digest=self.external_commitment)
        self.assertEqual(q['verdict'], 'REJECT')

    def test_key_reuse_and_symlink_rejected(self):
        with self.assertRaises(FileExistsError):
            create_evaluator_keys(self.root/'new.priv', self.public)
        link = self.root/'symlink.pub'
        link.symlink_to(self.public)
        with self.assertRaises(PermissionError):
            EvaluatorPublicKey(link).raw()
        self.assertFalse(authenticate_study(self.report, EvaluatorPublicKey(link)))

    def test_attestation_domain_and_fields_cannot_be_changed(self):
        forged = copy.deepcopy(self.report)
        forged['attestation']['statement']['purpose'] = 'saal.anything_else'
        self.assertFalse(authenticate_study(forged, self.verifier))
        forged = copy.deepcopy(self.report)
        forged['attestation']['statement']['extra'] = 'allowed?'
        self.assertFalse(authenticate_study(forged, self.verifier))


if __name__ == '__main__':
    unittest.main()
