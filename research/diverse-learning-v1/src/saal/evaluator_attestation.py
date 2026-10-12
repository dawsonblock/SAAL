"""Domain-separated public-key evaluator attestations for synthetic studies.

The qualifier needs *only* a pinned public key. The private key never crosses
from the evaluator into the learner, actor or qualification authority.
This authenticates the evaluation statement, not the truth of real outcomes.
"""
from __future__ import annotations

import base64
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .admission import _crypto, keygen
from .canonical import digest, encode
from .gate_server import secure_key


PURPOSE = "saal.synthetic_evaluation.report.v4"
PURPOSE_V5 = "saal.synthetic_evaluation.report.v5"


@dataclass(frozen=True)
class EvaluatorPublicKey:
    path: str | Path

    def raw(self) -> bytes:
        if not hasattr(os, "O_NOFOLLOW"):
            raise PermissionError("secure public-key reading unavailable on this system")
        try:
            fd = os.open(str(self.path), os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
        except OSError as exc:
            raise PermissionError("cannot open pinned public key") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != 32:
                raise ValueError("evaluator public key must be a 32-byte regular file")
            blob = os.read(fd, 33)
            if len(blob) != 32:
                raise ValueError("invalid Ed25519 evaluator public key length")
            return blob
        finally:
            os.close(fd)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.raw()).hexdigest()


def create_evaluator_keys(private_path: str | Path, public_path: str | Path) -> None:
    private_path, public_path = Path(private_path), Path(public_path)
    if private_path.resolve() == public_path.resolve():
        raise ValueError("private and public key paths must be different")
    if public_path.exists() or public_path.is_symlink():
        raise FileExistsError("public key already exists")
    keygen(str(private_path), str(public_path))


def sign_evaluation(report: dict[str, Any], private_path: str | Path) -> dict[str, Any]:
    Private, _, _ = _crypto()
    private = Private.from_private_bytes(secure_key(str(private_path)))
    statement = {
        "schema_version": 1,
        "purpose": PURPOSE_V5 if "witness_digest" in report else PURPOSE,
        "report_digest": digest(report),
        "prereg_digest": report.get("prereg_digest"),
        "model_digest": report.get("model_digest"),
        "implementation_digest": report.get("implementation_digest"),
        "source_partition": report.get("source_partition"),
    }
    if "witness_digest" in report:
        statement["witness_digest"] = report["witness_digest"]
    return {"statement": statement,
            "signature_b64": base64.b64encode(private.sign(encode(statement))).decode("ascii")}


def verify_evaluation(report: dict[str, Any], attestation: dict,
                      public_key: EvaluatorPublicKey) -> bool:
    try:
        _, Public, _ = _crypto()
        statement = attestation["statement"]
        fields = {"schema_version", "purpose", "report_digest", "prereg_digest",
                  "model_digest", "implementation_digest", "source_partition"}
        v5 = "witness_digest" in report
        if not isinstance(statement, dict) or set(statement) != (fields | ({"witness_digest"} if v5 else set())):
            return False
        if statement["schema_version"] != 1 or statement["purpose"] != (PURPOSE_V5 if v5 else PURPOSE):
            return False
        if v5 and statement["witness_digest"] != report["witness_digest"]:
            return False
        for field in ("prereg_digest", "model_digest", "implementation_digest", "source_partition"):
            if statement[field] != report.get(field):
                return False
        if statement["report_digest"] != digest(report):
            return False
        signature = base64.b64decode(attestation["signature_b64"], validate=True)
        if len(signature) != 64:
            return False
        Public.from_public_bytes(public_key.raw()).verify(signature, encode(statement))
        return True
    except (KeyError, ValueError, TypeError, OSError, PermissionError):
        return False
    except Exception as exc:
        from cryptography.exceptions import InvalidSignature
        if isinstance(exc, InvalidSignature):
            return False
        raise
