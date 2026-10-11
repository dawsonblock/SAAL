"""Independently signed preregistration witness for synthetic v5 studies.

A receipt proves possession of a separately pinned witness signing key, not
wall-clock ordering by itself. To prove chronology, a third-party append-only
journal must record the receipt before any evaluator sees candidate weights.
"""
from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path
from typing import Any

from .admission import _crypto
from .canonical import digest, encode
from .evaluator_attestation import EvaluatorPublicKey, create_evaluator_keys
from .gate_server import secure_key
from .implementation import implementation_digest

PURPOSE = "saal.preregistration.witness.v5"
STATEMENT_FIELDS = frozenset({
    "schema_version", "purpose", "study_id", "prereg_digest",
    "evaluator_public_key_sha256", "witness_public_key_sha256",
    "implementation_digest", "issued_at_ns", "issued_by",
})


def create_witness_keys(private_path: str | Path, public_path: str | Path) -> None:
    """Generate a separate keypair on the independent witness machine."""
    create_evaluator_keys(private_path, public_path)


def create_witness_receipt(prereg: dict[str, Any], private_path: str | Path,
                           witness_public_key: str | Path, *,
                           witness_identity: str = "independent_witness") -> dict[str, Any]:
    """Only the witness authority may issue; never give this key to evaluator/trainer.

    The signed timestamp is asserted by the witness, not proof of chronology.
    External journaling of the receipt is required for rigorous preregistration.
    """
    if (prereg.get("witness_scheme") != "ed25519" or
            prereg.get("signature_scheme") != "ed25519" or
            prereg.get("study") != "synthetic_assistance_v5_witnessed"):
        raise ValueError("v5 witnessed preregistration required")
    witness_fp = EvaluatorPublicKey(witness_public_key).fingerprint()
    if witness_fp != prereg.get("witness_public_key_sha256"):
        raise PermissionError("witness key not pinned in preregistration")
    if witness_fp == prereg.get("evaluator_public_key_sha256"):
        raise PermissionError("witness must not share evaluator key")
    if not isinstance(witness_identity, str) or not witness_identity.strip():
        raise ValueError("witness identity is required")
    study_id = prereg.get("study_id")
    if not isinstance(study_id, str) or len(study_id) != 32:
        raise ValueError("missing or invalid random study identifier")
    if any(ch not in "0123456789abcdef" for ch in study_id):
        raise ValueError("invalid study identifier encoding")
    Private, _, serial = _crypto()
    private = Private.from_private_bytes(secure_key(str(private_path)))
    actual_pub = private.public_key().public_bytes(serial.Encoding.Raw, serial.PublicFormat.Raw)
    if hashlib.sha256(actual_pub).hexdigest() != witness_fp:
        raise PermissionError("private witness key does not match pinned public key")
    statement = {
        "schema_version": 1,
        "purpose": PURPOSE,
        "study_id": study_id,
        "prereg_digest": digest(prereg),
        "evaluator_public_key_sha256": prereg["evaluator_public_key_sha256"],
        "witness_public_key_sha256": witness_fp,
        "implementation_digest": implementation_digest(),
        "issued_at_ns": time.time_ns(),
        "issued_by": witness_identity.strip(),
    }
    return {"statement": statement, "signature_b64": base64.b64encode(
        private.sign(encode(statement))).decode("ascii")}


def verify_witness_receipt(prereg: dict, receipt: dict,
                           witness_public_key: str | Path) -> bool:
    """Exact digest, purpose, study, source and independently pinned key check."""
    try:
        _, Public, _ = _crypto()
        if not isinstance(receipt, dict) or set(receipt) != {"statement", "signature_b64"}:
            return False
        statement = receipt["statement"]
        if not isinstance(statement, dict) or set(statement) != STATEMENT_FIELDS:
            return False
        fp = EvaluatorPublicKey(witness_public_key).fingerprint()
        if (prereg.get("witness_scheme") != "ed25519" or
                prereg.get("signature_scheme") != "ed25519" or
                prereg.get("study") != "synthetic_assistance_v5_witnessed" or
                fp != prereg.get("witness_public_key_sha256") or
                fp == prereg.get("evaluator_public_key_sha256") or
                statement["schema_version"] != 1 or statement["purpose"] != PURPOSE or
                statement["prereg_digest"] != digest(prereg) or
                statement["study_id"] != prereg.get("study_id") or
                statement["evaluator_public_key_sha256"] != prereg.get("evaluator_public_key_sha256") or
                statement["witness_public_key_sha256"] != fp or
                statement["implementation_digest"] != implementation_digest() or
                type(statement["issued_at_ns"]) is not int or statement["issued_at_ns"] <= 0 or
                not isinstance(statement["issued_by"], str) or
                not statement["issued_by"].strip()):
            return False
        signature = base64.b64decode(receipt["signature_b64"], validate=True)
        if len(signature) != 64:
            return False
        Public.from_public_bytes(EvaluatorPublicKey(witness_public_key).raw()).verify(
            signature, encode(statement))
        return True
    except (KeyError, ValueError, TypeError, OSError, PermissionError):
        return False
    except Exception as exc:
        from cryptography.exceptions import InvalidSignature
        if isinstance(exc, InvalidSignature):
            return False
        raise
