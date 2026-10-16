#!/usr/bin/env python3
"""Reconcile physical ZIP contents against embedded SHA-256 manifests.

The tool deliberately does not infer publisher trust or silently choose a
manifest's path scope. A unique matching interpretation is a clue for review,
not a declared historical fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import tomllib
import zipfile
from pathlib import PurePosixPath

LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
LFS_OID = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.M)
LFS_SIZE = re.compile(rb"^size ([0-9]+)$", re.M)
MANIFEST_NAMES = {"manifest.sha256", "sha256sums", "sha256sum.txt"}
MANIFEST_LINE = re.compile(r"^([0-9a-fA-F]{64})[ \t]+(.*)$")
MAX_MANIFEST_BYTES = 8 * 1024 * 1024


def _json_digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _safe_relative(value: str, base: tuple[str, ...] = ()) -> str | None:
    if not value or "\\" in value or "\x00" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return None
    stack = list(base)
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not stack:
                return None
            stack.pop()
        else:
            stack.append(part)
    return "/".join(stack) if stack else None


def parse_sha256_manifest(raw: bytes) -> dict:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return {"entries": [], "malformed_lines": [0], "duplicate_paths": []}
    entries: list[dict] = []
    malformed: list[int] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        match = MANIFEST_LINE.fullmatch(line)
        if not match:
            malformed.append(line_number)
            continue
        digest, target = match.groups()
        # sha256sum permits a '*' marker before a binary filename.
        if target.startswith("*"):
            target = target[1:]
        if not target or target.startswith("\\"):
            malformed.append(line_number)
            continue
        if target in seen:
            duplicates.append(target)
        seen.add(target)
        entries.append({"sha256": digest.lower(), "target": target, "line": line_number})
    return {"entries": entries, "malformed_lines": malformed, "duplicate_paths": sorted(set(duplicates))}


def _resolve_target(manifest_path: str, target: str, scope: dict) -> str | None:
    kind = scope.get("type")
    parent = PurePosixPath(manifest_path).parent.parts
    if kind == "archive_root":
        return _safe_relative(target)
    if kind == "manifest_directory":
        return _safe_relative(target, tuple(parent))
    if kind == "strip_manifest_parent":
        prefix = "/".join(parent)
        if not prefix or not target.startswith(prefix + "/"):
            return None
        return _safe_relative(target[len(prefix) + 1 :])
    if kind == "strip_prefix":
        prefix = scope.get("prefix")
        if type(prefix) is not str or not prefix or not target.startswith(prefix.rstrip("/") + "/"):
            return None
        return _safe_relative(target[len(prefix.rstrip("/")) + 1 :])
    return None


def _evaluate_scope(manifest_path: str, parsed: dict, inventory: dict[str, dict], scope: dict) -> dict:
    missing, mismatches, invalid = [], [], []
    for entry in parsed["entries"]:
        target = _resolve_target(manifest_path, entry["target"], scope)
        if target is None:
            invalid.append(entry["target"])
            continue
        member = inventory.get(target)
        if member is None:
            missing.append(target)
        elif member.get("sha256") != entry["sha256"]:
            mismatches.append(target)
    clean = not (parsed["malformed_lines"] or parsed["duplicate_paths"] or missing or mismatches or invalid)
    return {
        "scope": scope,
        "entries": len(parsed["entries"]),
        "missing_members": sorted(set(missing)),
        "digest_mismatches": sorted(set(mismatches)),
        "invalid_paths": sorted(set(invalid)),
        "malformed_lines": parsed["malformed_lines"],
        "duplicate_paths": parsed["duplicate_paths"],
        "status": "MATCH" if clean else "MISMATCH",
    }


def reconcile_manifests(
    inventory: dict[str, dict],
    manifest_contents: dict[str, bytes],
    declared_scopes: dict[str, dict] | None = None,
) -> dict:
    declared_scopes = declared_scopes or {}
    results = []
    for manifest_path in sorted(manifest_contents):
        raw = manifest_contents[manifest_path]
        parsed = parse_sha256_manifest(raw)
        candidates = [{"type": "archive_root"}, {"type": "manifest_directory"}]
        if PurePosixPath(manifest_path).parent.parts:
            candidates.append({"type": "strip_manifest_parent"})
        candidate_results = [_evaluate_scope(manifest_path, parsed, inventory, item) for item in candidates]
        matches = [item for item in candidate_results if item["status"] == "MATCH"]
        selected = declared_scopes.get(manifest_path)
        selected_result = _evaluate_scope(manifest_path, parsed, inventory, selected) if selected else None
        results.append({
            "path": manifest_path,
            "physical_sha256": hashlib.sha256(raw).hexdigest(),
            "declared_scope": selected,
            "likely_scope_for_review": matches[0]["scope"] if len(matches) == 1 else None,
            "candidate_interpretations": candidate_results,
            "resolution": ("MATCH" if selected_result and selected_result["status"] == "MATCH"
                           else "MISMATCH" if selected_result else "SCOPE_UNRESOLVED"),
            "selected_result": selected_result,
            "disposition": ("VALID_HISTORICAL_SCOPE" if selected_result and selected_result["status"] == "MATCH"
                            else "REQUIRED_UNRESOLVED"),
        })
    return {
        "schema": "saal-manifest-reconciliation-v1",
        "manifest_count": len(results),
        "manifests": results,
        "external_artifact_references": "NOT_INFERRED_REQUIRES_EXPLICIT_ARTIFACT_LOCK_REVIEW",
        "historical_manifest_bytes_modified": False,
        "publisher_authentication": "NOT_ESTABLISHED_BY_HASH_RECONCILIATION",
        "overall_status": ("NO_MANIFESTS" if not results else
                           "RESOLVED" if all(item["resolution"] == "MATCH" for item in results) else "UNRESOLVED"),
    }


def audit_archive(path: str, declared_scopes: dict[str, dict] | None = None,
                  max_total_uncompressed: int = 64 * 1024**3,
                  max_member_uncompressed: int = 16 * 1024**3) -> dict:
    archive_hash = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            archive_hash.update(chunk)
    inventory: dict[str, dict] = {}
    manifest_contents: dict[str, bytes] = {}
    duplicates, unsafe, collisions, symlinks, read_errors, limits = [], [], [], [], [], []
    seen_casefold: dict[str, str] = {}
    total_declared = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            is_directory = info.is_dir()
            if not name or "\\" in name or name.startswith("/") or "\x00" in name or re.match(r"^[A-Za-z]:", name):
                unsafe.append(name)
                continue
            parts = name.rstrip("/").split("/")
            if any(part in ("", ".", "..") for part in parts):
                unsafe.append(name)
                continue
            if is_directory:
                continue
            if name in inventory:
                duplicates.append(name)
                continue
            fold = name.casefold()
            if fold in seen_casefold and seen_casefold[fold] != name:
                collisions.append({"first": seen_casefold[fold], "second": name})
            else:
                seen_casefold[fold] = name
            mode = info.external_attr >> 16
            symlink = stat.S_ISLNK(mode)
            if symlink:
                symlinks.append(name)
            total_declared += info.file_size
            if info.file_size > max_member_uncompressed or total_declared > max_total_uncompressed:
                limits.append(name)
                inventory[name] = {"size": info.file_size, "status": "RESOURCE_LIMIT_EXCEEDED"}
                continue
            digest = hashlib.sha256()
            prefix = b""
            size = 0
            try:
                with archive.open(info) as member:
                    while chunk := member.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        if len(prefix) < 512:
                            prefix += chunk[:512 - len(prefix)]
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                read_errors.append({"path": name, "error": type(exc).__name__})
                continue
            record = {"size": size, "sha256": digest.hexdigest(), "status": "VERIFIED", "symlink": symlink}
            if prefix.startswith(LFS_HEADER):
                oid, declared = LFS_OID.search(prefix), LFS_SIZE.search(prefix)
                record["lfs_pointer"] = ({"oid_sha256": oid.group(1).decode(), "declared_size": int(declared.group(1))}
                                          if oid and declared else {"malformed": True})
            inventory[name] = record
            if PurePosixPath(name).name.lower() in MANIFEST_NAMES:
                if size > MAX_MANIFEST_BYTES:
                    read_errors.append({"path": name, "error": "MANIFEST_TOO_LARGE"})
                else:
                    try:
                        manifest_contents[name] = archive.read(info)
                    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                        read_errors.append({"path": name, "error": type(exc).__name__})
    reconciliation = reconcile_manifests(inventory, manifest_contents, declared_scopes)
    physical = [{"path": path, "sha256": record.get("sha256"), "size": record.get("size")}
                for path, record in sorted(inventory.items())]
    package_versions = _package_versions(path, inventory)
    integrity_ok = not any((duplicates, unsafe, collisions, symlinks, read_errors, limits))
    return {
        "schema": "saal-release-manifest-audit-v1",
        "archive": {"name": path.rsplit("/", 1)[-1], "size": __import__("os").path.getsize(path),
                    "sha256": archive_hash.hexdigest()},
        "physical_inventory_digest": _json_digest(physical),
        "physical_member_count": len(inventory),
        "duplicate_paths": sorted(set(duplicates)),
        "normalized_path_collisions": collisions,
        "unsafe_paths": sorted(set(unsafe)),
        "symlink_members": sorted(set(symlinks)),
        "resource_limit_members": sorted(set(limits)),
        "read_errors": read_errors,
        "manifest_reconciliation": reconciliation,
        "package_versions": package_versions,
        "physical_integrity_status": "CLEAN" if integrity_ok else "FAILED",
        "publisher_authentication": "NOT_ESTABLISHED_BY_ARCHIVE_CONTENTS",
    }


def _package_versions(path: str, inventory: dict[str, dict]) -> dict:
    project_path, runtime_path = "pyproject.toml", "src/saal_rsi/__init__.py"
    if project_path not in inventory or runtime_path not in inventory:
        return {"status": "INCOMPLETE", "pyproject_version": None, "runtime_version": None}
    if inventory[project_path]["size"] > 1024 * 1024 or inventory[runtime_path]["size"] > 1024 * 1024:
        return {"status": "METADATA_TOO_LARGE", "pyproject_version": None, "runtime_version": None}
    try:
        with zipfile.ZipFile(path) as archive:
            project = tomllib.loads(archive.read(project_path).decode("utf-8"))
            runtime = archive.read(runtime_path).decode("utf-8")
        project_version = project.get("project", {}).get("version")
        match = re.search(r"(?m)^__version__\s*=\s*['\"]([^'\"]+)['\"]", runtime)
        runtime_version = match.group(1) if match else None
    except (UnicodeDecodeError, KeyError, ValueError, zipfile.BadZipFile, tomllib.TOMLDecodeError) as exc:
        return {"status": "PARSE_ERROR", "error": type(exc).__name__,
                "pyproject_version": None, "runtime_version": None}
    return {"status": "MATCH" if project_version and project_version == runtime_version else "MISMATCH",
            "pyproject_version": project_version, "runtime_version": runtime_version}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="ZIP archive to inventory without extraction")
    parser.add_argument("--scopes-json", help="JSON object mapping manifest paths to reviewed scope objects")
    parser.add_argument("--output", required=True, help="Write JSON reconciliation report")
    args = parser.parse_args()
    scopes = json.load(open(args.scopes_json, encoding="utf-8")) if args.scopes_json else None
    report = audit_archive(args.archive, scopes)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["manifest_reconciliation"]["overall_status"],
                      "manifests": report["manifest_reconciliation"]["manifest_count"],
                      "output": args.output}))
    integrity_ok = report["physical_integrity_status"] == "CLEAN" and report["package_versions"]["status"] == "MATCH" and not any((
                            report["duplicate_paths"], report["normalized_path_collisions"],
                            report["unsafe_paths"], report["symlink_members"],
                            report["resource_limit_members"], report["read_errors"]))
    return 0 if integrity_ok and report["manifest_reconciliation"]["overall_status"] == "RESOLVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
