from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import pytest


def _load_tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "verify_recovered_checkpoints.py"
    spec = importlib.util.spec_from_file_location("verify_recovered_checkpoints", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pointer_parser_requires_canonical_shape_and_full_oid():
    tool = _load_tool()
    oid = "a" * 64
    parsed = tool.parse_lfs_pointer(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 123\n".encode()
    )
    assert parsed == {"oid_sha256": oid, "size": 123}
    with pytest.raises(ValueError, match="canonical"):
        tool.parse_lfs_pointer(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\n".encode())


def test_classifier_tensor_inventory_matches_expected_conversion():
    tool = _load_tool()
    upstream = {
        "embeddings.weight": {"shape": [8, 4], "dtype": "F32"},
        "decoder.bias": {"shape": [8], "dtype": "F32"},
    }
    actual = {
        "embeddings.weight": {"shape": [8, 4], "dtype": "F32"},
        "classifier.weight": {"shape": [4, 4], "dtype": "F32"},
        "classifier.bias": {"shape": [4], "dtype": "F32"},
    }
    extra = {"classifier.weight": {"shape": [4, 4], "dtype": "F32"},
             "classifier.bias": {"shape": [4], "dtype": "F32"}}
    report = tool.compare_sequence_classifier_inventory(actual, upstream, ["decoder.bias"], extra)
    assert report["base_tensors_matched"] == 1
    altered = dict(actual)
    altered["embeddings.weight"] = {"shape": [8, 5], "dtype": "F32"}
    with pytest.raises(ValueError, match="base tensor inventory mismatch"):
        tool.compare_sequence_classifier_inventory(altered, upstream, ["decoder.bias"], extra)


def test_recovery_refuses_pointer_bytes_as_a_model_payload(tmp_path):
    tool = _load_tool()
    path = tmp_path / "model.safetensors"
    oid = "a" * 64
    path.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 3\n")
    assert path.stat().st_size < 1024
    with pytest.raises(Exception):
        with tool.safe_open(str(path), framework="pt", device="cpu") as _:
            pass


def test_substituted_checkpoint_bytes_are_rejected(tmp_path):
    tool = _load_tool()
    path = tmp_path / "checkpoint.safetensors"
    path.write_bytes(b"substituted tensors")
    expected = hashlib.sha256(b"original tensors").hexdigest()
    with pytest.raises(ValueError, match="hash/size mismatch"):
        tool.verify_payload_identity(path, expected, len(b"original tensors"))
