"""Witnessed v5 trust-boundary tests; synthetic study only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from saal.canonical import digest
from saal.evaluation import authenticate_study, run_study
from saal.evaluator_attestation import EvaluatorPublicKey, create_evaluator_keys, sign_evaluation
from saal.prereg import private_prereg
from saal.qualification import qualify
from saal.replay import audit_replay
from saal.train import fit_intent_model
from saal.witness import create_witness_keys, create_witness_receipt, verify_witness_receipt


class WitnessedV5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_intent_model(seed=1201, epochs=1, per_class=4)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.eval_private = self.root/'eval.priv'
        self.eval_public = self.root/'eval.pub'
        self.witness_private = self.root/'witness.priv'
        self.witness_public = self.root/'witness.pub'
        create_evaluator_keys(self.eval_private, self.eval_public)
        create_witness_keys(self.witness_private, self.witness_public)
        self.prereg = private_prereg(10, 30, self.eval_public, self.witness_public)
        self.prereg['episodes_per_seed'] = 2
        self.prereg['min_episodes'] = 20
        self.prereg['min_improvement_lower_bound'] = -100.
        self.prereg['max_critical_event_rate_upper_bound'] = 1.
        self.receipt = create_witness_receipt(self.prereg, self.witness_private,
                                              self.witness_public)
        self.report = run_study(self.model, seeds=self.prereg['seeds'],
                                episodes_per_seed=2, prereg=self.prereg,
                                eval_private_key_file=str(self.eval_private),
                                witness_receipt=self.receipt,
                                witness_public_key=str(self.witness_public))

    def qualify(self, *, prereg=None, receipt=None, public=None, report=None, commitment=None):
        return qualify(report or self.report, EvaluatorPublicKey(self.eval_public),
                       prereg or self.prereg, model=self.model,
                       expected_prereg_digest=commitment or digest(self.prereg),
                       witness_receipt=receipt if receipt is not None else self.receipt,
                       witness_public_key=str(public or self.witness_public))

    def test_v5_signatures_and_exact_replay(self):
        self.assertTrue(verify_witness_receipt(self.prereg, self.receipt, self.witness_public))
        self.assertTrue(authenticate_study(self.report, EvaluatorPublicKey(self.eval_public)))
        replay = audit_replay(self.report, self.prereg, self.model,
                              EvaluatorPublicKey(self.eval_public),
                              witness_receipt=self.receipt,
                              witness_public_key=str(self.witness_public))
        self.assertTrue(replay['valid'], replay)
        out = self.qualify()
        self.assertTrue(out['replay_audit']['valid'])
        self.assertNotIn('witness', ' '.join(out['reasons']))
        self.assertEqual(out['witness_receipt_digest'], digest(self.receipt))
        self.assertEqual(self.report['attestation']['statement']['purpose'],
                         'saal.synthetic_evaluation.report.v5')

    def test_missing_witness_fails_closed(self):
        out = qualify(self.report, EvaluatorPublicKey(self.eval_public), self.prereg,
                      model=self.model, expected_prereg_digest=digest(self.prereg))
        self.assertEqual(out['verdict'], 'REJECT')
        self.assertTrue(any('witness' in r for r in out['reasons']))
        with self.assertRaises(PermissionError):
            run_study(self.model, seeds=self.prereg['seeds'], episodes_per_seed=2,
                      prereg=self.prereg, eval_private_key_file=str(self.eval_private))

    def test_unapproved_witness_key_fails(self):
        new_priv, new_pub = self.root/'extra.priv', self.root/'extra.pub'
        create_witness_keys(new_priv, new_pub)
        self.assertFalse(verify_witness_receipt(self.prereg, self.receipt, new_pub))
        self.assertEqual(self.qualify(public=new_pub)['verdict'], 'REJECT')
        with self.assertRaises(PermissionError):
            create_witness_receipt(self.prereg, new_priv, new_pub)

    def test_witness_is_not_evaluator(self):
        with self.assertRaises(ValueError):
            private_prereg(10, 30, self.eval_public, self.eval_public)
        change = copy.deepcopy(self.prereg)
        change['witness_public_key_sha256'] = change['evaluator_public_key_sha256']
        self.assertFalse(verify_witness_receipt(change, self.receipt, self.witness_public))

    def test_prereg_edit_invalidates_witness(self):
        altered = copy.deepcopy(self.prereg)
        altered['max_critical_event_rate_upper_bound'] = 0.2
        self.assertFalse(verify_witness_receipt(altered, self.receipt, self.witness_public))
        self.assertEqual(self.qualify(prereg=altered)['verdict'], 'REJECT')

    def test_receipt_tamper_rejected(self):
        altered = copy.deepcopy(self.receipt)
        altered['statement']['issued_by'] = 'untrusted_trainer'
        self.assertFalse(verify_witness_receipt(self.prereg, altered, self.witness_public))
        self.assertEqual(self.qualify(receipt=altered)['verdict'], 'REJECT')

    def test_attestation_extra_field_rejected(self):
        altered = copy.deepcopy(self.receipt)
        altered['statement']['authorized_to_deploy'] = True
        self.assertFalse(verify_witness_receipt(self.prereg, altered, self.witness_public))

    def test_witness_signature_downgrade_rejected(self):
        altered = copy.deepcopy(self.receipt)
        altered['statement']['purpose'] = 'saal.preregistration.witness.v4'
        self.assertFalse(verify_witness_receipt(self.prereg, altered, self.witness_public))
        altered = copy.deepcopy(self.report)
        del altered['report']['witness_digest']
        altered['report_digest'] = digest(altered['report'])
        altered['attestation'] = sign_evaluation(altered['report'], self.eval_private)
        self.assertTrue(authenticate_study(altered, EvaluatorPublicKey(self.eval_public)))
        self.assertEqual(self.qualify(report=altered)['verdict'], 'REJECT')

    def test_replay_missing_witness_rejects(self):
        audit = audit_replay(self.report, self.prereg, self.model,
                             EvaluatorPublicKey(self.eval_public))
        self.assertFalse(audit['valid'])

    def test_study_id_substitution_rejected(self):
        altered = copy.deepcopy(self.receipt)
        altered['statement']['study_id'] = 'f'*32
        self.assertFalse(verify_witness_receipt(self.prereg, altered, self.witness_public))

    def test_symlink_witness_key_rejected(self):
        link = self.root/'witness-sym.pub'
        link.symlink_to(self.witness_public)
        self.assertFalse(verify_witness_receipt(self.prereg, self.receipt, link))
        with self.assertRaises(PermissionError):
            create_witness_receipt(self.prereg, link, self.witness_public)

    def test_legacy_v4_still_accepted_by_existing_verifier(self):
        old_prereg = private_prereg(10, 30, self.eval_public)
        old_prereg['episodes_per_seed'] = 2
        old_prereg['min_episodes'] = 20
        old_prereg['min_improvement_lower_bound'] = -100.
        old_prereg['max_critical_event_rate_upper_bound'] = 1.
        report = run_study(self.model, seeds=old_prereg['seeds'], episodes_per_seed=2,
                           prereg=old_prereg, eval_private_key_file=str(self.eval_private))
        self.assertTrue(authenticate_study(report, EvaluatorPublicKey(self.eval_public)))
        self.assertEqual(report['attestation']['statement']['purpose'],
                         'saal.synthetic_evaluation.report.v4')
        result = qualify(report, EvaluatorPublicKey(self.eval_public), old_prereg,
                         model=self.model, expected_prereg_digest=digest(old_prereg))
        self.assertNotIn('witness', ' '.join(result['reasons']))


if __name__ == '__main__':
    unittest.main()
