import json
import os
import tempfile
import unittest
from pathlib import Path

from saal.admission import keygen
from saal.canonical import digest
from saal.release import (attest_qualification, authorize_candidate,
                          model_semantic_digest, stage_candidate,
                          verify_candidate_authorization, verify_qualification)
from saal.train import fit_intent_model


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.qual_private, self.qual_public = self.root/'qual.secret', self.root/'qual.pub'
        self.human_private, self.human_public = self.root/'human.secret', self.root/'human.pub'
        keygen(str(self.qual_private), str(self.qual_public))
        keygen(str(self.human_private), str(self.human_public))
        self.model_path = self.root/'model.json'
        fit_intent_model(epochs=1, per_class=3).save(self.model_path)
        self.prereg = {"schema_version":1,"seeds":[9,10,11], "episodes_per_seed": 10}
        self.qualification = {"verdict": "ELIGIBLE_FOR_HUMAN_REVIEW",
                              "report_digest": "a"*64,
                              "prereg_digest": digest(self.prereg),
                              "model_digest": model_semantic_digest(self.model_path),
                              "stats": {"count": 30}, "reasons": []}
        self.qualified = attest_qualification(self.qualification, self.qual_private, "Reviewer")

    def tearDown(self):
        self.tmp.cleanup()

    def authorize(self):
        return authorize_candidate(self.qualified, self.qual_public, self.model_path,
                                   self.prereg, self.human_private,
                                   "Human", "Synthetic-only staging approval")

    def test_human_authorization_and_staging_do_not_deploy(self):
        authorization = self.authorize()
        self.assertTrue(verify_qualification(self.qualified, self.qual_public))
        self.assertTrue(verify_candidate_authorization(authorization, self.human_public,
                          self.qualified, self.qual_public, self.model_path, self.prereg))
        result = stage_candidate(authorization, self.human_public, self.qualified,
                                 self.qual_public, self.model_path, self.prereg, self.root/'stage')
        self.assertEqual(result['status'], 'STAGED_NOT_DEPLOYED')
        saved = json.loads((Path(result['path'])/'release_record.json').read_text())
        self.assertFalse(saved['deployment_performed'])
        self.assertEqual(saved['authorized_for'], 'staging_only')
        self.assertFalse((self.root/'stage'/'current.json').exists())
        with self.assertRaises(FileExistsError):
            stage_candidate(authorization, self.human_public, self.qualified,
                            self.qual_public, self.model_path, self.prereg, self.root/'stage')

    def test_rejects_model_tamper_after_human_approval(self):
        authorization = self.authorize()
        self.model_path.write_text(self.model_path.read_text() + ' ')
        self.assertFalse(verify_candidate_authorization(authorization, self.human_public,
                         self.qualified, self.qual_public, self.model_path, self.prereg))
        with self.assertRaises(PermissionError):
            stage_candidate(authorization, self.human_public, self.qualified,
                            self.qual_public, self.model_path, self.prereg, self.root/'stage')

    def test_rejects_wrong_human_public_key_and_modified_prereg(self):
        authorization = self.authorize()
        self.assertFalse(verify_candidate_authorization(authorization, self.qual_public,
                         self.qualified, self.qual_public, self.model_path, self.prereg))
        modified = {**self.prereg, 'episodes_per_seed': 11}
        self.assertFalse(verify_candidate_authorization(authorization, self.human_public,
                         self.qualified, self.qual_public, self.model_path, modified))

    def test_rejected_qualification_cannot_be_signed(self):
        with self.assertRaises(ValueError):
            attest_qualification({**self.qualification, 'verdict': 'REJECT'},
                                  self.qual_private, 'reviewer')

    def test_actor_cannot_fake_qualifier_or_human_signatures(self):
        qualified = json.loads(json.dumps(self.qualified))
        qualified['statement']['qualification']['report_digest'] = 'b'*64
        self.assertFalse(verify_qualification(qualified, self.qual_public))
        auth = self.authorize()
        auth['statement']['permits_deployment'] = True
        self.assertFalse(verify_candidate_authorization(auth, self.human_public,
                         self.qualified, self.qual_public, self.model_path, self.prereg))

    def test_staging_root_symlink_rejected(self):
        if os.name == 'nt':
            self.skipTest('POSIX symlink example')
        (self.root/'stage').symlink_to(self.root/'somewhere')
        with self.assertRaises(PermissionError):
            stage_candidate(self.authorize(), self.human_public, self.qualified,
                            self.qual_public, self.model_path, self.prereg, self.root/'stage')


if __name__ == '__main__':
    unittest.main()
