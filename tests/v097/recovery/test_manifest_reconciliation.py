from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import zipfile
from pathlib import Path


def _load_tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "reconcile_release_manifests.py"
    spec = importlib.util.spec_from_file_location("reconcile_release_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_source_comparator():
    path = Path(__file__).resolve().parents[3] / "tools" / "compare_v096_sources.py"
    spec = importlib.util.spec_from_file_location("compare_v096_sources", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_scope_is_not_silently_inferred():
    tool = _load_tool()
    payload = b"current release file"
    digest = hashlib.sha256(payload).hexdigest()
    report = tool.reconcile_manifests(
        {"bundle/README.md": {"sha256": digest, "size": len(payload)}},
        {"bundle/MANIFEST.sha256": f"{digest}  bundle/README.md\n".encode()},
    )
    row = report["manifests"][0]
    assert row["resolution"] == "SCOPE_UNRESOLVED"
    assert row["likely_scope_for_review"] == {"type": "archive_root"}
    assert report["overall_status"] == "UNRESOLVED"


def test_declared_scope_and_digest_are_verified():
    tool = _load_tool()
    payload = b"current release file"
    digest = hashlib.sha256(payload).hexdigest()
    report = tool.reconcile_manifests(
        {"bundle/README.md": {"sha256": digest, "size": len(payload)}},
        {"bundle/MANIFEST.sha256": f"{digest}  bundle/README.md\n".encode()},
        {"bundle/MANIFEST.sha256": {"type": "archive_root"}},
    )
    assert report["manifests"][0]["resolution"] == "MATCH"
    assert report["manifests"][0]["disposition"] == "VALID_HISTORICAL_SCOPE"
    assert report["publisher_authentication"] == "NOT_ESTABLISHED_BY_HASH_RECONCILIATION"


def test_standalone_package_root_scope_must_be_explicitly_declared():
    tool = _load_tool()
    payload = b"standalone package file"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = "provenance/package/MANIFEST.sha256"
    raw = f"{digest}  provenance/package/README.md\n".encode()
    unresolved = tool.reconcile_manifests({"README.md": {"sha256": digest, "size": len(payload)}},
                                          {manifest: raw})
    assert unresolved["manifests"][0]["resolution"] == "SCOPE_UNRESOLVED"
    resolved = tool.reconcile_manifests(
        {"README.md": {"sha256": digest, "size": len(payload)}}, {manifest: raw},
        {manifest: {"type": "strip_manifest_parent"}},
    )
    assert resolved["manifests"][0]["resolution"] == "MATCH"
    assert resolved["manifests"][0]["disposition"] == "VALID_HISTORICAL_SCOPE"


def test_declared_bad_scope_and_substituted_digest_fail():
    tool = _load_tool()
    report = tool.reconcile_manifests(
        {"bundle/README.md": {"sha256": "0" * 64, "size": 1}},
        {"bundle/MANIFEST.sha256": f"{'1' * 64}  README.md\n".encode()},
        {"bundle/MANIFEST.sha256": {"type": "manifest_directory"}},
    )
    row = report["manifests"][0]
    assert row["resolution"] == "MISMATCH"
    assert row["selected_result"]["digest_mismatches"] == ["bundle/README.md"]


def test_manifest_rejects_duplicate_and_escaping_entries():
    tool = _load_tool()
    digest = "a" * 64
    parsed = tool.parse_sha256_manifest(f"{digest}  ../../../outside\n{digest}  ../../../outside\n".encode())
    report = tool.reconcile_manifests(
        {}, {"nested/MANIFEST.sha256": f"{digest}  ../../../outside\n{digest}  ../../../outside\n".encode()},
        {"nested/MANIFEST.sha256": {"type": "manifest_directory"}},
    )
    assert parsed["duplicate_paths"] == ["../../../outside"]
    selected = report["manifests"][0]["selected_result"]
    assert selected["invalid_paths"] == ["../../../outside"]
    assert selected["duplicate_paths"] == ["../../../outside"]
    assert report["overall_status"] == "UNRESOLVED"


def test_archive_inventory_reports_versions_duplicates_and_lfs(tmp_path):
    tool = _load_tool()
    archive = tmp_path / "input.zip"
    payload = b"weights"
    pointer = ("version https://git-lfs.github.com/spec/v1\n"
               f"oid sha256:{hashlib.sha256(payload).hexdigest()}\nsize {len(payload)}\n")
    digest = hashlib.sha256(b"readme").hexdigest()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("weights/model.safetensors", pointer)
        zf.writestr("README.md", "readme")
        zf.writestr("MANIFEST.sha256", f"{digest}  README.md\n")
        zf.writestr("pyproject.toml", '[project]\nversion="0.9.7.dev0"\n')
        zf.writestr("src/saal_rsi/__init__.py", '__version__ = "0.9.6"\n')
        zf.writestr("duplicate.txt", "first")
        zf.writestr("duplicate.txt", "second")
    report = tool.audit_archive(str(archive), {"MANIFEST.sha256": {"type": "archive_root"}})
    assert report["archive"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert report["duplicate_paths"] == ["duplicate.txt"]
    assert report["package_versions"] == {"status": "MISMATCH", "pyproject_version": "0.9.7.dev0",
                                           "runtime_version": "0.9.6"}
    assert report["manifest_reconciliation"]["manifests"][0]["resolution"] == "MATCH"
    assert report["publisher_authentication"] == "NOT_ESTABLISHED_BY_ARCHIVE_CONTENTS"
    assert report["physical_integrity_status"] == "FAILED"


def test_archive_without_manifest_is_not_reported_as_resolved(tmp_path):
    tool = _load_tool()
    archive = tmp_path / "no-manifest.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "target")
    report = tool.audit_archive(str(archive))
    assert report["manifest_reconciliation"]["overall_status"] == "NO_MANIFESTS"
    assert report["symlink_members"] == ["link"]
    assert report["physical_integrity_status"] == "FAILED"


def test_source_comparison_requires_expected_archive_hash_and_never_imports(tmp_path):
    tool = _load_source_comparator()
    baseline = tmp_path / "baseline"
    local = baseline / "src/saal_rsi/adaptive_coordinator_v5.py"
    local.parent.mkdir(parents=True)
    local.write_text("same source\n")
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("checkout/src/saal_rsi/adaptive_coordinator_v5.py", "same source\n")
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = tool.compare_sources(archive, baseline, expected)
    row = next(x for x in report["files"] if x["archive_suffix"] == "src/saal_rsi/adaptive_coordinator_v5.py")
    assert report["archive_hash_match"] is True
    assert row["status"] == "UNCHANGED"
    assert report["imports_performed"] is False
    wrong = tool.compare_sources(archive, baseline, "0" * 64)
    assert wrong["comparison_status"] == "REJECTED_ARCHIVE_HASH_MISMATCH"
    assert wrong["files"] == []


def test_source_comparison_rejects_duplicate_candidate_paths(tmp_path):
    tool = _load_source_comparator()
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("one/src/saal_rsi/adaptive_coordinator_v5.py", "first")
        zf.writestr("two/src/saal_rsi/adaptive_coordinator_v5.py", "second")
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = tool.compare_sources(archive, tmp_path, expected)
    row = next(x for x in report["files"] if x["archive_suffix"] == "src/saal_rsi/adaptive_coordinator_v5.py")
    assert row["status"] == "REJECTED_DUPLICATE_CANDIDATE"
