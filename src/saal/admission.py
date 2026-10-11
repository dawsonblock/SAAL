"""External signed training-data admission. Requires optional 'cryptography' package.

Signer and private key must live OUTSIDE learner/train/evaluator trust domains.
Trainer holds ONLY the public verification key; HMAC would be insufficient here.
"""
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from .canonical import encode


def _crypto():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:
        raise RuntimeError("Training-data attestation needs pip install 'saal-assistance[attestation]'") from exc
    return Ed25519PrivateKey, Ed25519PublicKey, serialization


def keygen(private_path: str, public_path: str) -> None:
    import os
    Private, Public, ser = _crypto()
    private = Private.generate()
    key_bytes = private.private_bytes(ser.Encoding.Raw, ser.PrivateFormat.Raw, ser.NoEncryption())
    path = Path(private_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(key_bytes)
    Path(public_path).write_bytes(private.public_key().public_bytes(ser.Encoding.Raw, ser.PublicFormat.Raw))


MAX_TRAIN_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate training data JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-JSON numeric constant: {value}")


def _read_train_snapshot(path: Path) -> bytes:
    """Read one exact no-follow immutable-in-memory data snapshot.

    This does not defend against a malicious co-tenant on the same host; process
    isolation and signed immutable candidate files are still required.
    """
    import os
    import stat
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("no-follow reads unavailable")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_TRAIN_BYTES or st.st_size == 0:
            raise ValueError("invalid training file size or type")
        data = b""
        while len(data) <= MAX_TRAIN_BYTES:
            part = os.read(fd, min(1024 * 1024, MAX_TRAIN_BYTES + 1 - len(data)))
            if not part:
                break
            data += part
        if len(data) > MAX_TRAIN_BYTES:
            raise ValueError("training file exceeds byte limit")
        return data
    finally:
        os.close(fd)


def _validate_train_bytes(data: bytes) -> int:
    if not data or len(data) > MAX_TRAIN_BYTES:
        raise ValueError("empty or oversize training snapshot")
    lines = data.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("empty training record")
    for idx, line in enumerate(lines):
        if len(line) > MAX_LINE_BYTES:
            raise ValueError(f"line {idx + 1}: oversized training record")
        row = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_pairs,
                         parse_constant=_reject_constant)
        if not isinstance(row, dict) or set(row) != {
                "source_partition", "admitted_by", "prompt", "ideal_response"}:
            raise ValueError(f"line {idx+1}: unexpected training fields")
        if (row["source_partition"] != "train" or
                row["admitted_by"] != "independent_data_authority" or
                not isinstance(row["prompt"], str) or
                not isinstance(row["ideal_response"], str)):
            raise ValueError(f"line {idx+1}: invalid provenance or data schema")
        if (not row["prompt"].strip() or not row["ideal_response"].strip() or
                "\x00" in row["prompt"] or "\x00" in row["ideal_response"]):
            raise ValueError(f"line {idx+1}: empty or invalid text")
    return len(lines)


def _validate_train_jsonl(path: Path) -> int:
    return _validate_train_bytes(_read_train_snapshot(path))


def issue_manifest(data_path: str, private_key_path: str, reviewer: str,
                   rationale: str) -> dict[str, Any]:
    Private, _, ser = _crypto()
    path = Path(data_path)
    data = _read_train_snapshot(path)
    count = _validate_train_bytes(data)
    if not reviewer or not rationale:
        raise ValueError("reviewer and evidence rationale are required")
    from .gate_server import secure_key
    private = Private.from_private_bytes(secure_key(private_key_path))
    statement = {"schema_version": 1, "purpose": "candidate_training_only",
                 "sha256": hashlib.sha256(data).hexdigest(),
                 "records": count, "reviewer": reviewer, "review_rationale": rationale,
                 "partition": "train", "not_authorized_for_promotion": True}
    signature = base64.b64encode(private.sign(encode(statement))).decode("ascii")
    return {"statement": statement, "ed25519_signature_b64": signature}


def verify_manifest(data_path: str, manifest: dict, public_key_path: str) -> bool:
    _, Public, _ = _crypto()
    try:
        statement = manifest["statement"]
        if not isinstance(statement, dict) or set(statement) != {
                "schema_version", "purpose", "sha256", "records", "reviewer",
                "review_rationale", "partition", "not_authorized_for_promotion"}:
            return False
        if (statement["schema_version"] != 1 or
                statement["purpose"] != "candidate_training_only"):
            return False
        if (statement["partition"] != "train" or
                statement["not_authorized_for_promotion"] is not True or
                type(statement["records"]) is not int or statement["records"] <= 0 or
                not isinstance(statement["reviewer"], str) or not statement["reviewer"].strip() or
                not isinstance(statement["review_rationale"], str) or
                not statement["review_rationale"].strip()):
            return False
        data = _read_train_snapshot(Path(data_path))
        if statement["sha256"] != hashlib.sha256(data).hexdigest():
            return False
        if statement["records"] != _validate_train_bytes(data):
            return False
        from .evaluator_attestation import EvaluatorPublicKey
        public = Public.from_public_bytes(EvaluatorPublicKey(public_key_path).raw())
        public.verify(base64.b64decode(manifest["ed25519_signature_b64"], validate=True),
                      encode(statement))
        return True
    except (KeyError, TypeError, ValueError, OSError, PermissionError, UnicodeError, json.JSONDecodeError):
        return False
    except Exception as exc:
        # cryptography's InvalidSignature is not part of Python's built-in errors.
        from cryptography.exceptions import InvalidSignature
        if isinstance(exc, InvalidSignature):
            return False
        raise
