#!/usr/bin/env python3
"""Hash-gated, non-importing comparison of allowlisted v0.9.6 ZIP sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED_ARCHIVE_SHA256 = "d5f42a0705d9711bdb3ce04953a928e96aa5a4762f2022ad5cfd2a2ff4c858f1"
CANDIDATES = [
    {"archive_suffix": "src/saal_rsi/adaptive_coordinator_v5.py", "baseline_path": "src/saal_rsi/adaptive_coordinator_v5.py", "treatment": "IMPORT_AND_HARDEN"},
    {"archive_suffix": "src/saal_rsi/authority/budget_authority_v5.py", "baseline_path": "src/saal_rsi/authority/budget_authority_v5.py", "treatment": "IMPORT_WITH_COMPATIBILITY_REVIEW"},
    {"archive_suffix": "src/saal_rsi/authority/source_closure_v5.py", "baseline_path": "src/saal_rsi/authority/source_closure_v5.py", "treatment": "REFERENCE_ONLY_KEEP_V6_AUTHORITATIVE"},
    {"archive_suffix": "src/saal_rsi/evaluation/safety_feasibility.py", "baseline_path": "src/saal_rsi/evaluation/safety_feasibility.py", "treatment": "IMPORT_AND_TEST"},
    {"archive_suffix": "src/saal_rsi/evaluation/bundle_feasibility_v5.py", "baseline_path": "src/saal_rsi/evaluation/bundle_feasibility_v5.py", "treatment": "DEVELOPMENT_ONLY_EVALUATOR"},
    {"archive_suffix": "src/saal_rsi/evaluation/independent_evidence.py", "baseline_path": "src/saal_rsi/evaluation/independent_evidence.py", "treatment": "IMPORT_WITH_SIGNER_TRUST_REVIEW"},
    {"archive_suffix": "src/saal_rsi/evaluation/replay_receipt_v5.py", "baseline_path": "src/saal_rsi/evaluation/replay_receipt_v5.py", "treatment": "IMPORT_WITH_PROTOCOL_COMPATIBILITY_CHECKS"},
    {"archive_suffix": "src/saal_rsi/evaluation/learning_curves.py", "baseline_path": "src/saal_rsi/evaluation/learning_curves.py", "treatment": "IMPORT_AND_TEST"},
    {"archive_suffix": "src/saal_rsi/learning/failure_diagnoser.py", "baseline_path": "src/saal_rsi/learning/failure_diagnoser.py", "treatment": "EXPERIMENTAL_MODULE"},
    {"archive_suffix": "src/saal_rsi/learning/intervention_controller.py", "baseline_path": "src/saal_rsi/learning/intervention_controller.py", "treatment": "PROPOSAL_ONLY_CONTROLLER"},
    {"archive_suffix": "src/saal_rsi/learning/skill_registry.py", "baseline_path": "src/saal_rsi/learning/skill_registry.py", "treatment": "IMPORT_AND_VALIDATE_STORAGE_SEMANTICS"},
    {"archive_suffix": "tests/v096/test_v096_upgrade.py", "baseline_path": None, "treatment": "ADAPT_AS_V097_REGRESSION_TESTS"},
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    if not name or "\\" in name or "\x00" in name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return False
    return not any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))


def _baseline_file(root: Path, relative: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink in baseline source path")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("baseline source path is not a regular in-root file")
    return resolved


def compare_sources(archive_path: str | Path, baseline_root: str | Path,
                    expected_archive_sha256: str = EXPECTED_ARCHIVE_SHA256) -> dict:
    archive = Path(archive_path)
    baseline = Path(baseline_root).resolve(strict=True)
    observed_archive_sha256 = _sha256_file(archive)
    result = {
        "schema": "saal-v097-source-comparison-v1",
        "archive_name": archive.name,
        "expected_archive_sha256": expected_archive_sha256,
        "observed_archive_sha256": observed_archive_sha256,
        "archive_hash_match": observed_archive_sha256 == expected_archive_sha256,
        "publisher_authentication": "NOT_ESTABLISHED_BY_ARCHIVE_HASH",
        "archive_structural_audit": "NOT_PERFORMED_USE_AUDIT_RELEASE_INTEGRITY_V2",
        "files": [],
        "imports_performed": False,
    }
    if not result["archive_hash_match"]:
        result["comparison_status"] = "REJECTED_ARCHIVE_HASH_MISMATCH"
        return result
    with zipfile.ZipFile(archive) as zf:
        names: dict[str, list[zipfile.ZipInfo]] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            for candidate in CANDIDATES:
                suffix = candidate["archive_suffix"]
                if info.filename == suffix or info.filename.endswith("/" + suffix):
                    names.setdefault(suffix, []).append(info)
        for candidate in CANDIDATES:
            suffix = candidate["archive_suffix"]
            matches = names.get(suffix, [])
            row = {**candidate}
            if not matches:
                row.update({"status": "MISSING_FROM_ARCHIVE", "archive_path": None, "archive_sha256": None})
            elif len(matches) != 1:
                row.update({"status": "REJECTED_DUPLICATE_CANDIDATE", "archive_path": None, "archive_sha256": None})
            else:
                info = matches[0]
                mode = info.external_attr >> 16
                if not _safe_member(info.filename) or stat.S_ISLNK(mode):
                    row.update({"status": "REJECTED_UNSAFE_CANDIDATE_PATH", "archive_path": info.filename,
                                "archive_sha256": None})
                elif info.file_size > 8 * 1024 * 1024:
                    row.update({"status": "REJECTED_CANDIDATE_TOO_LARGE", "archive_path": info.filename,
                                "archive_sha256": None})
                else:
                    try:
                        source_bytes = zf.read(info)
                    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                        row.update({"status": "REJECTED_UNREADABLE_CANDIDATE", "archive_path": info.filename,
                                    "archive_sha256": None, "error": type(exc).__name__})
                    else:
                        archive_digest = hashlib.sha256(source_bytes).hexdigest()
                        row["archive_path"] = info.filename
                        row["archive_sha256"] = archive_digest
                        local_rel = candidate["baseline_path"]
                        if local_rel is None:
                            row["status"] = "NEW"
                        else:
                            try:
                                local = _baseline_file(baseline, local_rel)
                            except FileNotFoundError:
                                row["status"] = "NEW"
                                row["baseline_sha256"] = None
                            except ValueError as exc:
                                row["status"] = "REJECTED_UNSAFE_BASELINE_PATH"
                                row["baseline_sha256"] = None
                                row["baseline_error"] = str(exc)
                            else:
                                row["baseline_sha256"] = _sha256_file(local)
                                row["status"] = "UNCHANGED" if row["baseline_sha256"] == archive_digest else "MODIFIED"
            result["files"].append(row)
    result["comparison_status"] = "COMPARISON_COMPLETE_REQUIRES_COMPATIBILITY_REVIEW"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--expected-sha256", default=EXPECTED_ARCHIVE_SHA256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_sources(args.archive, args.baseline, args.expected_sha256)
    Path(args.output).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison_status": report["comparison_status"],
                      "archive_hash_match": report["archive_hash_match"],
                      "files_classified": len(report["files"]), "imports_performed": False}))
    return 0 if report["archive_hash_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
