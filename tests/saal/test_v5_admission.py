"""Training-data snapshot and independent-admission regressions."""
import json
import tempfile
import unittest
from pathlib import Path
from saal.admission import issue_manifest, keygen, verify_manifest


class AdmissionHardeningV5(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root/'train.jsonl'
        self.private, self.public = self.root/'admit.priv', self.root/'admit.pub'
        keygen(str(self.private), str(self.public))

    def sign(self, raw):
        self.data.write_text(raw)
        return issue_manifest(str(self.data), str(self.private), 'reviewer', 'consented synthetic only')

    def test_valid_snapshot(self):
        record = {"source_partition":"train", "admitted_by":"independent_data_authority",
                  "prompt":"Which choice?", "ideal_response":"Please clarify"}
        envelope = self.sign(json.dumps(record)+'\n')
        self.assertTrue(verify_manifest(str(self.data), envelope, str(self.public)))

    def test_duplicate_record_key_rejected(self):
        with self.assertRaises(ValueError):
            self.sign('{"source_partition":"train","source_partition":"holdout",'
                      '"admitted_by":"independent_data_authority",'
                      '"prompt":"hello","ideal_response":"hi"}\n')

    def test_hidden_extra_payload_rejected(self):
        row = {"source_partition":"train", "admitted_by":"independent_data_authority",
               "prompt":"hello", "ideal_response":"hi", "private_holdout": "leak"}
        with self.assertRaises(ValueError):
            self.sign(json.dumps(row)+'\n')

    def test_unauthorized_partition_rejected(self):
        row = {"source_partition":"sealed_holdout", "admitted_by":"independent_data_authority",
               "prompt":"hello", "ideal_response":"hi"}
        with self.assertRaises(ValueError):
            self.sign(json.dumps(row)+'\n')

    def test_symlinked_training_file_rejected(self):
        valid = self.root/'original.jsonl'
        valid.write_text(json.dumps({"source_partition":"train", "admitted_by":"independent_data_authority",
                                     "prompt":"x", "ideal_response":"y"})+'\n')
        self.data.symlink_to(valid)
        with self.assertRaises(OSError):
            issue_manifest(str(self.data), str(self.private), 'reviewer', 'rationale')

    def test_symlinked_public_key_rejected(self):
        record = {"source_partition":"train", "admitted_by":"independent_data_authority",
                  "prompt":"x", "ideal_response":"y"}
        envelope = self.sign(json.dumps(record)+'\n')
        link = self.root/'link.pub'
        link.symlink_to(self.public)
        self.assertFalse(verify_manifest(str(self.data), envelope, str(link)))

    def test_forced_manifest_boolean_rejected(self):
        record = {"source_partition":"train", "admitted_by":"independent_data_authority",
                  "prompt":"x", "ideal_response":"y"}
        env = self.sign(json.dumps(record)+'\n')
        env['statement']['not_authorized_for_promotion'] = 1
        self.assertFalse(verify_manifest(str(self.data), env, str(self.public)))

    def test_modified_after_signing_fails(self):
        record = {"source_partition":"train", "admitted_by":"independent_data_authority",
                  "prompt":"x", "ideal_response":"y"}
        env = self.sign(json.dumps(record)+'\n')
        self.data.write_text(self.data.read_text().replace('"x"','"changed"'))
        self.assertFalse(verify_manifest(str(self.data), env, str(self.public)))


if __name__ == '__main__':
    unittest.main()
