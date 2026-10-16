"""Read-only ZIP audit for unsafe paths, duplicate members, LFS pointers and drift."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tomllib
import zipfile
from pathlib import PurePosixPath

_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
_LFS_OID = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.M)
_LFS_SIZE = re.compile(rb"^size ([0-9]+)$", re.M)
_MANIFEST_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+[* ]?(.+?)\s*$")
_SOURCE_SUFFIXES = {".py", ".sh", ".bash", ".zsh", ".ps1", ".command", ".rb", ".pl"}
_MODEL_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".model", ".tflite"}


def _unsafe(name: str) -> bool:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        return True
    if re.match(r"^[A-Za-z]:", name):
        return True
    parts = name.rstrip("/").split("/")
    return any(part in ("", ".", "..") for part in parts)


def _pointer(prefix: bytes) -> dict | None:
    if not prefix.startswith(_LFS_HEADER):
        return None
    oid, size = _LFS_OID.search(prefix), _LFS_SIZE.search(prefix)
    if not oid or not size:
        return {"malformed": True}
    return {"oid_sha256": oid.group(1).decode("ascii"), "declared_size": int(size.group(1))}


def _manifest_member(manifest_name: str, target: str) -> str | None:
    if not target or "\\" in target or target.startswith("/") or re.match(r"^[A-Za-z]:", target):
        return None
    stack = list(PurePosixPath(manifest_name).parent.parts)
    for part in target.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not stack:
                return None
            stack.pop()
        else:
            stack.append(part)
    return "/".join(stack) if stack else None


def _archive_manifest(members: dict[str, dict], manifest_bytes: dict[str, bytes]) -> dict:
    candidates = [n for n in members if PurePosixPath(n).name.lower() in {"manifest.sha256", "sha256sums", "sha256sum.txt"}]
    reports = []
    for name in sorted(candidates):
        raw = manifest_bytes.get(name)
        if raw is None:
            reports.append({"path": name, "status": "UNREADABLE_OR_DUPLICATE"})
            continue
        expected: dict[str, str] = {}
        malformed = []
        for lineno, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = _MANIFEST_LINE.match(line)
            if not match:
                malformed.append(lineno)
                continue
            digest, target = match.groups()
            target = _manifest_member(name, target)
            if target is None:
                malformed.append(lineno)
                continue
            if target in expected:
                malformed.append(lineno)
            expected[target] = digest.lower()
        mismatches = []
        missing = []
        for target, digest in expected.items():
            actual = members.get(target)
            if actual is None:
                missing.append(target)
            elif actual.get("sha256") != digest:
                mismatches.append(target)
        reports.append({"path": name, "entries": len(expected), "malformed_lines": malformed,
                        "missing_members": sorted(missing), "digest_mismatches": sorted(mismatches),
                        "status": "MATCH" if not (malformed or missing or mismatches) else "MISMATCH"})
    return {"manifests": reports, "status": "NOT_PRESENT" if not candidates else
            ("MATCH" if all(item["status"] == "MATCH" for item in reports) else "MISMATCH")}


def audit_zip(
    path: str,
    closure: dict | None = None,
    *,
    max_total_uncompressed: int = 64 * 1024**3,
    max_member_uncompressed: int = 16 * 1024**3,
) -> dict:
    if type(max_total_uncompressed) is not int or max_total_uncompressed <= 0:
        raise ValueError("positive total uncompressed-byte limit required")
    if type(max_member_uncompressed) is not int or max_member_uncompressed <= 0:
        raise ValueError("positive per-member uncompressed-byte limit required")
    archive_hash = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            archive_hash.update(chunk)
    members: dict[str, dict] = {}
    rows = []
    exact_duplicates = []
    normalized: dict[str, str] = {}
    normalized_collisions = []
    unsafe_paths = []
    lfs_pointers = []
    source_files = []
    executable_files = []
    model_lfs_pointers = []
    manifest_bytes: dict[str, bytes] = {}
    limit_events = []
    read_errors = []
    total_uncompressed_read = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename
            if _unsafe(name):
                unsafe_paths.append(name)
            key = name.casefold()
            if name in members:
                exact_duplicates.append(name)
            elif key in normalized and normalized[key] != name:
                normalized_collisions.append({"first": normalized[key], "second": name})
            else:
                normalized[key] = name
            mode = info.external_attr >> 16
            is_link = stat.S_ISLNK(mode)
            if info.file_size > max_member_uncompressed or total_uncompressed_read + info.file_size > max_total_uncompressed:
                record = {"path": name, "declared_size": info.file_size, "compressed_size": info.compress_size,
                          "directory": info.is_dir(), "symlink": is_link, "status": "RESOURCE_LIMIT_EXCEEDED"}
                limit_events.append({"path": name, "declared_size": info.file_size})
                if name not in members:
                    members[name] = record
                rows.append(record)
                continue
            digest = hashlib.sha256()
            prefix = b""
            size = 0
            try:
                with zf.open(info) as item:
                    while chunk := item.read(1024 * 1024):
                        if size + len(chunk) > max_member_uncompressed or total_uncompressed_read + size + len(chunk) > max_total_uncompressed:
                            limit_events.append({"path": name, "bytes_read_before_limit": size})
                            size = -1
                            break
                        if len(prefix) < 512:
                            prefix += chunk[:512 - len(prefix)]
                        digest.update(chunk)
                        size += len(chunk)
            except Exception as exc:
                error_record = {"path": name, "status": "READ_ERROR", "error": type(exc).__name__}
                rows.append(error_record)
                read_errors.append(error_record)
                continue
            if size < 0:
                limit_record = {"path": name, "declared_size": info.file_size, "compressed_size": info.compress_size,
                                "directory": info.is_dir(), "symlink": is_link, "status": "RESOURCE_LIMIT_EXCEEDED"}
                rows.append(limit_record)
                if name not in members:
                    members[name] = limit_record
                continue
            total_uncompressed_read += size
            record = {"path": name, "sha256": digest.hexdigest(), "size": size,
                      "compressed_size": info.compress_size, "directory": info.is_dir(), "symlink": is_link,
                      "status": "VERIFIED"}
            pointer = _pointer(prefix)
            if pointer is not None:
                record["lfs_pointer"] = pointer
                lfs_pointers.append({"path": name, **pointer})
                if PurePosixPath(name).suffix.lower() in _MODEL_SUFFIXES:
                    model_lfs_pointers.append({"path": name, **pointer})
            if not info.is_dir() and PurePosixPath(name).suffix.lower() in _SOURCE_SUFFIXES:
                source_files.append(name)
            if not info.is_dir() and mode & 0o111:
                executable_files.append(name)
            if name not in members:
                members[name] = record
            rows.append(record)
        for name in list(members):
            if PurePosixPath(name).name.lower() in {"manifest.sha256", "sha256sums", "sha256sum.txt"}:
                if sum(1 for info in zf.infolist() if info.filename == name) == 1:
                    try:
                        with zf.open(name) as stream:
                            manifest_bytes[name] = stream.read(8 * 1024 * 1024 + 1)
                    except Exception:
                        pass
    manifest = _archive_manifest(members, manifest_bytes)
    version = {"status": "NOT_CHECKED", "pyproject_version": None, "runtime_version": None}
    project = members.get("pyproject.toml")
    runtime_init = members.get("src/saal_rsi/__init__.py")
    # Byte payloads are read directly from the ZIP only for these two small files.
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if "pyproject.toml" in names and "src/saal_rsi/__init__.py" in names:
            try:
                project_data = tomllib.loads(zf.read("pyproject.toml").decode("utf-8"))
                init_text = zf.read("src/saal_rsi/__init__.py").decode("utf-8")
                project_version = project_data.get("project", {}).get("version")
                runtime_match = re.search(r"(?m)^__version__\s*=\s*['\"]([^'\"]+)['\"]", init_text)
                runtime_version = runtime_match.group(1) if runtime_match else None
                version = {"status": "MATCH" if project_version and project_version == runtime_version else "MISMATCH",
                           "pyproject_version": project_version, "runtime_version": runtime_version}
            except Exception as exc:
                version = {"status": "PARSE_ERROR", "error": type(exc).__name__}
    candidates = set(source_files) | set(executable_files)
    closure_result = {"status": "NOT_CHECKED", "coverage_authentication": "NOT_VERIFIED",
                      "uncovered_source_files": sorted(candidates)}
    if closure is not None:
        try:
            if type(closure) is not dict or type(closure.get("files")) is not list:
                raise ValueError("closure schema")
            covered = {entry["path"] for entry in closure["files"]
                       if type(entry) is dict and type(entry.get("path")) is str and not _unsafe(entry["path"])}
            uncovered = sorted(candidates - covered)
            closure_result = {"status": "PATH_COVERAGE_COMPLETE_UNAUTHENTICATED" if not uncovered else "PATH_COVERAGE_GAPS_UNAUTHENTICATED",
                              "coverage_authentication": "NOT_VERIFIED", "uncovered_source_files": uncovered}
        except Exception:
            closure_result = {"status": "INVALID_COVERAGE_MANIFEST", "coverage_authentication": "NOT_VERIFIED",
                              "uncovered_source_files": sorted(candidates)}
    return {
        "schema": "saal-archive-audit-v2", "archive": {"name": path.rsplit("/", 1)[-1],
            "size": os.path.getsize(path), "sha256": archive_hash.hexdigest()},
        "members": rows, "member_count": len(rows), "exact_duplicate_paths": sorted(exact_duplicates),
        "normalized_path_collisions": normalized_collisions, "unsafe_paths": sorted(unsafe_paths),
        "lfs_pointers": lfs_pointers, "original_manifest": manifest,
        "completeness": "COMPLETE" if not (limit_events or read_errors) else "INCOMPLETE",
        "resource_limits": {"max_total_uncompressed_bytes": max_total_uncompressed,
                             "max_member_uncompressed_bytes": max_member_uncompressed,
                             "bytes_read": total_uncompressed_read, "limit_events": limit_events,
                             "read_errors": read_errors},
        "source_closure_coverage": closure_result, "package_version": version,
        "model_payloads": {"status": "MODEL_LFS_PAYLOADS_UNRESOLVED" if model_lfs_pointers else "NO_MODEL_LFS_POINTERS_FOUND",
                            "requiredness": "MUST_BE_REVIEWED_AGAINST_RUNTIME_CLOSURE",
                            "model_lfs_pointer_count": len(model_lfs_pointers), "pointers": model_lfs_pointers},
        "release_provenance": {"status": "NOT_AUTHENTICATED_BY_ARCHIVE_AUDIT",
                               "note": "A same-archive manifest or signing key is not an independent trust anchor."},
    }


def render_recovery_report(report: dict) -> str:
    return "\n".join([
        "# Archive recovery report", "",
        f"Archive: `{report['archive']['name']}` ({report['archive']['size']} bytes)",
        f"SHA-256: `{report['archive']['sha256']}`", "",
        f"Members: {report['member_count']}",
        f"Unsafe paths: {len(report['unsafe_paths'])}",
        f"Duplicate paths: {len(report['exact_duplicate_paths'])}",
        f"Git LFS pointers: {len(report['lfs_pointers'])}",
        f"Manifest: {report['original_manifest']['status']}",
        f"Source closure coverage: {report['source_closure_coverage']['status']}",
        f"Package/runtime version: {report['package_version']['status']}", "",
        "This audit inventories archive bytes and detects structural inconsistencies. It does not authenticate the publisher, recover missing Git LFS objects, certify dynamic-import completeness, or establish scientific qualification.",
        "",
    ])
