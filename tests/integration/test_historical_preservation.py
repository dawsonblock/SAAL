from pathlib import Path
import hashlib,json

def test_frozen_historical_worker_and_negative_evidence():
    root=Path(__file__).resolve().parents[2]
    freeze=json.loads((root/'provenance/v0.9.3-historical-freeze.json').read_text())
    for relative,digest in freeze['protected_files'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
