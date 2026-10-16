"""Explicit, reviewable source snapshots with static local-import closure checks.

This module establishes content integrity, not publisher provenance. A snapshot
is development-only until its envelope is signed by a separately trusted key
and the reviewed path set is approved outside the source tree.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Iterable

from ..model_contract import verify
from ..spec import canonical

SCHEMA = "saal-source-closure-v6"
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
_LFS_OID = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.M)
_LFS_SIZE = re.compile(rb"^size ([0-9]+)$", re.M)


def _relative_path(value: object) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError("source path must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("unsafe source path")
    return path.as_posix()


def _regular_file(root: Path, relative: str) -> Path:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink in source path: {relative}")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"source path is not a regular in-root file: {relative}")
    return resolved


def _lfs_pointer(prefix: bytes) -> dict | None:
    if not prefix.startswith(_LFS_HEADER):
        return None
    oid = _LFS_OID.search(prefix)
    size = _LFS_SIZE.search(prefix)
    if not oid or not size:
        raise ValueError("malformed Git LFS pointer")
    return {"oid_sha256": oid.group(1).decode("ascii"), "declared_size": int(size.group(1))}


def _module_index(root: Path) -> dict[str, str]:
    source = root / "src"
    result: dict[str, str] = {}
    if not source.exists():
        return result
    for path in source.rglob("*.py"):
        if path.is_symlink():
            continue
        relative = path.relative_to(source).as_posix()
        parts = relative[:-3].split("/")
        if parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        module = ".".join(parts)
        if module in result:
            raise ValueError(f"ambiguous source module: {module}")
        result[module] = path.relative_to(root).as_posix()
    return result


def _import_targets(node: ast.AST, module: str, is_package: bool) -> set[str]:
    targets: set[str] = set()
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if not isinstance(node, ast.ImportFrom):
        return targets
    if node.level:
        package = module if is_package else module.rpartition(".")[0]
        parts = package.split(".") if package else []
        keep = max(0, len(parts) - (node.level - 1))
        base = parts[:keep]
        if node.module:
            base.extend(node.module.split("."))
        prefix = ".".join(base)
        if prefix:
            targets.add(prefix)
        if node.module is None:
            for alias in node.names:
                if alias.name != "*":
                    targets.add(".".join([*base, alias.name]))
    elif node.module:
        targets.add(node.module)
        for alias in node.names:
            if alias.name != "*":
                targets.add(node.module + "." + alias.name)
    return targets


def _check_import_closure(root: Path, paths: set[str], entry_points: list[str]) -> None:
    modules = _module_index(root)
    reverse = {path: name for name, path in modules.items()}
    missing: set[str] = set()
    for relative in sorted(paths):
        module = reverse.get(relative)
        if module is None:
            continue
        raw = _regular_file(root, relative).read_bytes()
        try:
            tree = ast.parse(raw, filename=relative)
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise ValueError(f"source cannot be statically inspected: {relative}") from exc
        is_package = relative.endswith("/__init__.py")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _import_targets(node, module, is_package):
                target_path = modules.get(target)
                if target_path and target_path not in paths:
                    missing.add(target_path)
    absent_entry_points = sorted(set(entry_points) - paths)
    if absent_entry_points:
        raise ValueError("entry points omitted from source closure: " + ", ".join(absent_entry_points))
    if missing:
        raise ValueError("local source imports omitted from source closure: " + ", ".join(sorted(missing)))


def build_source_snapshot(
    root: str | os.PathLike[str],
    reviewed_paths: Iterable[str],
    entry_points: Iterable[str],
) -> dict:
    """Create an explicitly scoped snapshot; does not sign or self-certify review."""
    base = Path(root).resolve(strict=True)
    paths = [_relative_path(item) for item in reviewed_paths]
    entries = [_relative_path(item) for item in entry_points]
    if not paths or not entries or len(set(paths)) != len(paths) or len(set(entries)) != len(entries):
        raise ValueError("source closure requires unique reviewed paths and entry points")
    path_set = set(paths)
    _check_import_closure(base, path_set, entries)
    files = []
    for relative in sorted(path_set):
        path = _regular_file(base, relative)
        digest = hashlib.sha256()
        size = 0
        prefix = b""
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if len(prefix) < 512:
                    prefix += chunk[:512 - len(prefix)]
                digest.update(chunk)
                size += len(chunk)
        pointer = _lfs_pointer(prefix)
        record = {"path": relative, "sha256": digest.hexdigest(), "size": size}
        if pointer is not None:
            record["unresolved_lfs_pointer"] = pointer
        files.append(record)
    return {
        "schema": SCHEMA,
        "closure_version": 6,
        "entry_points": sorted(entries),
        "files": files,
        "static_local_imports_checked": True,
        "dynamic_import_review": "REQUIRED_OUTSIDE_STATIC_ANALYSIS",
        "review_status": "UNREVIEWED_DEVELOPMENT",
        "review_attestations": [],
        "publisher_authenticated": False,
        "release_eligible": False,
    }


def verify_source_snapshot(snapshot: object, root: str | os.PathLike[str]) -> dict:
    """Verify bytes and local-import closure without asserting publisher trust."""
    if type(snapshot) is not dict or set(snapshot) != {
        "schema", "closure_version", "entry_points", "files", "static_local_imports_checked",
        "dynamic_import_review", "review_status", "review_attestations", "publisher_authenticated",
        "release_eligible",
    }:
        raise ValueError("source closure schema")
    if snapshot["schema"] != SCHEMA or snapshot["closure_version"] != 6:
        raise ValueError("source closure version")
    if snapshot["review_status"] != "UNREVIEWED_DEVELOPMENT" or snapshot["review_attestations"] != []:
        raise ValueError("unsupported review claim")
    if snapshot["publisher_authenticated"] is not False or snapshot["release_eligible"] is not False:
        raise ValueError("development snapshot cannot claim publisher authentication or release eligibility")
    if snapshot["static_local_imports_checked"] is not True or snapshot["dynamic_import_review"] != "REQUIRED_OUTSIDE_STATIC_ANALYSIS":
        raise ValueError("source closure analysis declaration")
    if type(snapshot["files"]) is not list or not snapshot["files"] or type(snapshot["entry_points"]) is not list:
        raise ValueError("source closure contents")
    base = Path(root).resolve(strict=True)
    seen: set[str] = set()
    for record in snapshot["files"]:
        if type(record) is not dict or set(record) not in (
            {"path", "sha256", "size"}, {"path", "sha256", "size", "unresolved_lfs_pointer"}
        ):
            raise ValueError("source file record")
        relative = _relative_path(record["path"])
        if relative in seen or type(record["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("duplicate path or invalid source hash")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError("source size")
        raw = _regular_file(base, relative).read_bytes()
        if len(raw) != record["size"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("source content mismatch: " + relative)
        pointer = _lfs_pointer(raw[:512])
        if pointer != record.get("unresolved_lfs_pointer"):
            raise ValueError("source LFS state mismatch: " + relative)
        seen.add(relative)
    _check_import_closure(base, seen, [_relative_path(x) for x in snapshot["entry_points"]])
    return {"files_verified": len(seen), "snapshot_digest": hashlib.sha256(canonical(snapshot)).hexdigest(),
            "publisher_authenticated": False, "release_eligible": False}


def verify_signed_source_snapshot(envelope: object, trusted_public_key_hex: str, root: str | os.PathLike[str]) -> dict:
    """Verify signature against an externally provisioned trust pin, then bytes.

    The public-key pin must come from an independent trust configuration. A key
    packaged next to the snapshot is not a trust anchor.
    """
    snapshot = verify(envelope, trusted_public_key_hex)
    result = verify_source_snapshot(snapshot, root)
    result["signature_verified_against_external_pin"] = True
    result["trusted_key_sha256"] = hashlib.sha256(bytes.fromhex(trusted_public_key_hex)).hexdigest()
    # A valid signature attributes bytes to the pinned key; it does not establish
    # that the key is an authorized release publisher or that review is complete.
    return result
