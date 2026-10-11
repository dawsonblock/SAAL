import json
import tempfile
import unittest
from pathlib import Path
from saal.admission import issue_manifest, keygen, verify_manifest


class AttestationTests(unittest.TestCase):
    def test_signed_training_data_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            private, public = p/'private.key', p/'public.key'
            data = p/'train.jsonl'
            data.write_text(json.dumps({"source_partition": "train", "admitted_by": "independent_data_authority",
                      "prompt": "Ask first", "ideal_response": "What is your preference?"})+'\n')
            keygen(str(private), str(public))
            issued = issue_manifest(str(data), str(private), "independent human reviewer", "synthetic non-sensitive")
            self.assertTrue(verify_manifest(str(data), issued, str(public)))
            data.write_text(data.read_text()+'\n')
            self.assertFalse(verify_manifest(str(data), issued, str(public)))


if __name__ == '__main__':
    unittest.main()
