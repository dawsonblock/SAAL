from __future__ import annotations

import hashlib
import json
import zipfile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.authority.archive_audit_v2 import audit_zip
from saal_rsi.authority.source_closure_v6 import (
    build_source_snapshot,
    verify_signed_source_snapshot,
    verify_source_snapshot,
)
from saal_rsi.model_contract import sign


def _write_source(root, *, with_dependency=True):
    package = root / "src" / "sample"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("from .worker import run\n", encoding="utf-8")
    worker = package / "worker.py"
    worker.write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    files = ["src/sample/__init__.py", "src/sample/worker.py"]
    if with_dependency:
        files.append("pyproject.toml")
        (root / "pyproject.toml").write_text('[project]\nname="sample"\n', encoding="utf-8")
    return files


def test_source_snapshot_checks_import_closure_and_detects_changed_source(tmp_path):
    paths = _write_source(tmp_path)
    snapshot = build_source_snapshot(tmp_path, paths, ["src/sample/__init__.py"])
    checked = verify_source_snapshot(snapshot, tmp_path)
    assert checked["files_verified"] == len(paths)
    assert snapshot["review_status"] == "UNREVIEWED_DEVELOPMENT"
    assert snapshot["release_eligible"] is False

    (tmp_path / "src/sample/worker.py").write_text("def run(): return 'changed'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source content mismatch"):
        verify_source_snapshot(snapshot, tmp_path)


def test_source_snapshot_rejects_omitted_local_dependency_and_unsafe_path(tmp_path):
    _write_source(tmp_path)
    with pytest.raises(ValueError, match="local source imports omitted"):
        build_source_snapshot(tmp_path, ["src/sample/__init__.py"], ["src/sample/__init__.py"])
    with pytest.raises(ValueError, match="unsafe source path"):
        build_source_snapshot(tmp_path, ["../outside.py"], ["../outside.py"])


def test_source_snapshot_marks_lfs_payload_unresolved(tmp_path):
    pointer = tmp_path / "src" / "weights.safetensors"
    pointer.parent.mkdir()
    oid = hashlib.sha256(b"missing model bytes").hexdigest()
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{oid}\nsize 19\n",
        encoding="ascii",
    )
    snapshot = build_source_snapshot(tmp_path, ["src/weights.safetensors"], ["src/weights.safetensors"])
    assert snapshot["files"][0]["unresolved_lfs_pointer"] == {"oid_sha256": oid, "declared_size": 19}
    assert verify_source_snapshot(snapshot, tmp_path)["release_eligible"] is False


def test_signed_source_snapshot_requires_external_matching_trust_pin(tmp_path):
    paths = _write_source(tmp_path)
    snapshot = build_source_snapshot(tmp_path, paths, ["src/sample/__init__.py"])
    trusted = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    pin = trusted.public_key().public_bytes_raw().hex()
    envelope = sign(snapshot, trusted)
    signed_result = verify_signed_source_snapshot(envelope, pin, tmp_path)
    assert signed_result["signature_verified_against_external_pin"] is True
    assert signed_result["release_eligible"] is False
    with pytest.raises(ValueError, match="invalid signature|pinned identity mismatch"):
        verify_signed_source_snapshot(envelope, other.public_key().public_bytes_raw().hex(), tmp_path)


def test_archive_audit_reports_lfs_duplicates_unsafe_paths_and_version_drift(tmp_path):
    archive = tmp_path / "candidate.zip"
    payload = b"not the model bytes"
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(payload).hexdigest()}\nsize {len(payload)}\n"
    )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("weights/model.safetensors", pointer)
        zf.writestr("../outside.py", "bad")
        zf.writestr("src/saal_rsi/__init__.py", '__version__ = "0.9.4.3"\n')
        zf.writestr("pyproject.toml", '[project]\nversion="0.9.7.dev0"\n')
        zf.writestr("duplicate.txt", "first")
        zf.writestr("duplicate.txt", "second")
    report = audit_zip(str(archive))
    json.dumps(report, sort_keys=True)
    assert len(report["lfs_pointers"]) == 1
    assert report["unsafe_paths"] == ["../outside.py"]
    assert report["exact_duplicate_paths"] == ["duplicate.txt"]
    assert report["package_version"]["status"] == "MISMATCH"
    assert report["model_payloads"]["status"] == "MODEL_LFS_PAYLOADS_UNRESOLVED"


def test_archive_audit_resolves_nested_manifest_paths_relative_to_manifest(tmp_path):
    archive = tmp_path / "nested.zip"
    data = b"nested evidence"
    digest = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("release/evidence.bin", data)
        zf.writestr("release/MANIFEST.sha256", f"{digest}  evidence.bin\n")
    report = audit_zip(str(archive))
    result = report["original_manifest"]
    assert result["status"] == "MATCH"
    assert result["manifests"][0]["missing_members"] == []


def test_archive_audit_stops_at_configured_expansion_limit(tmp_path):
    archive = tmp_path / "bounded.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("large.bin", b"x" * 4096)
    report = audit_zip(str(archive), max_member_uncompressed=1024)
    assert report["completeness"] == "INCOMPLETE"
    assert report["resource_limits"]["limit_events"] == [{"path": "large.bin", "declared_size": 4096}]
    assert report["members"][0]["status"] == "RESOURCE_LIMIT_EXCEEDED"


def test_archive_closure_coverage_is_never_reported_as_authenticated(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("src/app.py", "print('ok')\n")
    report = audit_zip(str(archive), {"files": [{"path": "src/app.py"}]})
    assert report["source_closure_coverage"]["status"] == "PATH_COVERAGE_COMPLETE_UNAUTHENTICATED"
    assert report["source_closure_coverage"]["coverage_authentication"] == "NOT_VERIFIED"


def test_source_snapshot_rejects_symlinked_paths(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    source = tmp_path / "src" / "sample"
    source.mkdir(parents=True)
    (source / "linked.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink in source path"):
        build_source_snapshot(tmp_path, ["src/sample/linked.py"], ["src/sample/linked.py"])


def test_project_and_runtime_version_contract_match():
    import re
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    project_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    init = (root / "src/saal_rsi/__init__.py").read_text()
    runtime_version = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", init).group(1)
    assert project_version == runtime_version == "0.9.7.dev0"
