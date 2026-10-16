#!/usr/bin/env python3
"""Verify recovered Git LFS checkpoint bytes without claiming publisher trust."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from safetensors import safe_open

LFS_VERSION = "version https://git-lfs.github.com/spec/v1"
_OID = re.compile(r"^oid sha256:([0-9a-f]{64})$")
_SIZE = re.compile(r"^size ([0-9]+)$")


def _safe_repo_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value or value.startswith("/"):
        raise ValueError("unsafe repository-relative checkpoint path")
    path = PurePosixPath(value)
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("unsafe repository-relative checkpoint path")
    return path.as_posix()


def parse_lfs_pointer(raw: bytes) -> dict:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Git LFS pointer must be ASCII") from exc
    if len(lines) != 3 or lines[0] != LFS_VERSION:
        raise ValueError("not a canonical three-line Git LFS pointer")
    oid, size = _OID.fullmatch(lines[1]), _SIZE.fullmatch(lines[2])
    if not oid or not size:
        raise ValueError("malformed Git LFS pointer")
    return {"oid_sha256": oid.group(1), "size": int(size.group(1))}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_payload_identity(path: Path, expected_sha256: str, expected_size: int) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint payload missing or not a regular file")
    observed_hash, observed_size = _hash_file(path)
    if observed_hash != expected_sha256 or observed_size != expected_size:
        raise ValueError("checkpoint payload hash/size mismatch")
    return observed_hash, observed_size


def _inventory_digest(inventory: dict[str, dict]) -> str:
    rows = [[name, value["dtype"], value["shape"]] for name, value in sorted(inventory.items())]
    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def compare_sequence_classifier_inventory(actual: dict[str, dict], upstream: dict[str, dict],
                                           expected_missing: list[str], expected_extra: dict[str, dict]) -> dict:
    missing = sorted(set(upstream) - set(actual))
    extra_names = sorted(set(actual) - set(upstream))
    if missing != sorted(expected_missing) or extra_names != sorted(expected_extra):
        raise ValueError("checkpoint tensor names do not match the declared classifier conversion")
    for name in sorted(set(actual) & set(upstream)):
        left, right = actual[name], upstream[name]
        if left["shape"] != right["shape"] or left["dtype"] != right["dtype"]:
            raise ValueError(f"base tensor inventory mismatch: {name}")
    for name, expected in expected_extra.items():
        if actual[name] != expected:
            raise ValueError(f"classification tensor shape/dtype mismatch: {name}")
    return {"upstream_tensor_count": len(upstream), "checkpoint_tensor_count": len(actual),
            "base_tensors_matched": len(set(actual) & set(upstream)),
            "expected_upstream_head_replacements": missing, "expected_classifier_tensors": extra_names,
            "checkpoint_inventory_sha256": _inventory_digest(actual)}


def _read_upstream_inventory(repo: Path, architecture: dict) -> dict[str, dict]:
    metadata_relative = _safe_repo_path(architecture["upstream_metadata_path"])
    metadata_path = repo / metadata_relative
    metadata_hash, _ = _hash_file(metadata_path)
    if metadata_hash != architecture["upstream_metadata_sha256"]:
        raise ValueError("pinned upstream metadata digest mismatch")
    upstream_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (upstream_metadata.get("id") != architecture["model_id"]
            or upstream_metadata.get("sha") != architecture["model_revision"]):
        raise ValueError("upstream model/revision identity mismatch")
    relative = _safe_repo_path(architecture["upstream_inventory_path"])
    path = repo / relative
    observed_hash, _ = _hash_file(path)
    if observed_hash != architecture["upstream_inventory_sha256"]:
        raise ValueError("pinned upstream tensor inventory digest mismatch")
    raw = json.loads(path.read_text(encoding="utf-8"))
    model_key = architecture["upstream_inventory_model_key"]
    model = raw.get(model_key)
    if type(model) is not dict or not model:
        raise ValueError("upstream model tensor inventory missing")
    return {name: {"shape": value["shape"], "dtype": value["dtype"]} for name, value in model.items()}


def verify_checkpoints(repo_root: str | Path, requirements: dict) -> dict:
    repo = Path(repo_root).resolve(strict=True)
    if requirements.get("schema") != "saal-checkpoint-requirements-v1":
        raise ValueError("checkpoint requirements schema")
    commit = requirements.get("baseline_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("baseline commit must be a full Git SHA-1")
    architecture = requirements.get("architecture")
    if type(architecture) is not dict or architecture.get("kind") != "ModernBertForSequenceClassification":
        raise ValueError("unsupported checkpoint architecture contract")
    upstream = _read_upstream_inventory(repo, architecture)
    expected_missing = architecture.get("expected_upstream_head_replacements")
    expected_extra = architecture.get("classifier_tensors")
    if type(expected_missing) is not list or type(expected_extra) is not dict:
        raise ValueError("incomplete classifier inventory contract")
    records = requirements.get("checkpoints")
    if type(records) is not list or len(records) != requirements.get("expected_checkpoint_path_count"):
        raise ValueError("checkpoint path inventory is incomplete")
    if len({record.get("path") for record in records if type(record) is dict}) != len(records):
        raise ValueError("duplicate checkpoint path in requirements")
    results = []
    for record in records:
        if type(record) is not dict:
            raise ValueError("checkpoint record schema")
        relative = _safe_repo_path(record.get("path"))
        pointer_bytes = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{relative}"],
                                       check=True, capture_output=True).stdout
        pointer = parse_lfs_pointer(pointer_bytes)
        pointer_blob = _git(repo, "rev-parse", f"{commit}:{relative}")
        if pointer_blob != record.get("git_pointer_blob"):
            raise ValueError("baseline Git pointer blob changed: " + relative)
        if pointer["oid_sha256"] != record.get("oid_sha256") or pointer["size"] != record.get("size_bytes"):
            raise ValueError("baseline Git LFS pointer differs from requirement: " + relative)
        payload = repo / relative
        try:
            observed_hash, observed_size = verify_payload_identity(payload, pointer["oid_sha256"], pointer["size"])
        except ValueError as exc:
            raise ValueError(f"{exc}: {relative}") from exc
        actual: dict[str, dict] = {}
        with safe_open(str(payload), framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                item = tensors.get_slice(name)
                actual[name] = {"shape": list(item.get_shape()), "dtype": item.get_dtype()}
        structural = compare_sequence_classifier_inventory(actual, upstream, expected_missing, expected_extra)
        if len(actual) != architecture.get("expected_tensor_count"):
            raise ValueError("checkpoint tensor count mismatch: " + relative)
        results.append({
            "path": relative,
            "git_pointer_blob": pointer_blob,
            "pointer_oid_sha256": pointer["oid_sha256"],
            "observed_sha256": observed_hash,
            "declared_size_bytes": pointer["size"],
            "observed_size_bytes": observed_size,
            "tensor_count": len(actual),
            "tensor_inventory_sha256": structural["checkpoint_inventory_sha256"],
            "structural_checks": structural,
            "content_status": "HASH_VERIFIED",
            "provenance_status": "POINTER_BOUND_TO_BASELINE_COMMIT; PUBLISHER_TRUST_UNVERIFIED",
        })
    groups: dict[str, list[str]] = {}
    for result in results:
        groups.setdefault(result["observed_sha256"], []).append(result["path"])
    if len(groups) != requirements.get("expected_unique_payload_count"):
        raise ValueError("unique payload count differs from checkpoint recovery contract")
    return {
        "schema": "saal-checkpoint-verification-v1",
        "baseline_commit": commit,
        "checkpoint_path_count": len(results),
        "unique_payload_count": len(groups),
        "payload_groups": [{"sha256": digest, "paths": paths} for digest, paths in sorted(groups.items())],
        "checkpoints": results,
        "content_integrity": "VERIFIED" if results else "NO_CHECKPOINTS",
        "publisher_provenance": "NOT_AUTHENTICATED_BY_THIS_VERIFIER",
        "historical_reproduction_status": "PROVENANCE_PENDING_TRUSTED_PUBLISHER_OR_SOURCE_REVIEW",
        "generated_at_utc": requirements.get("verification_time_utc"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    requirements = json.loads(Path(args.requirements).read_text(encoding="utf-8"))
    result = verify_checkpoints(args.repo, requirements)
    Path(args.output).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"content_integrity": result["content_integrity"],
                      "publisher_provenance": result["publisher_provenance"],
                      "checkpoint_paths": result["checkpoint_path_count"],
                      "unique_payloads": result["unique_payload_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
