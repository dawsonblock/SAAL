"""Offline, independently signed candidate qualification and human release staging.

Explicitly NOT runtime model promotion. Distinct Ed25519 public keys verify
qualifier and human authorization. Neither the learner nor actor gets keys.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .admission import _crypto
from .canonical import digest, encode
from .train import IntentClassifier


def _require_private_file(path: str | Path) -> bytes:
    from .gate_server import secure_key
    return secure_key(str(path))


def _signed(statement: dict, private_key_path: str | Path) -> dict:
    Private, _, _ = _crypto()
    private = Private.from_private_bytes(_require_private_file(private_key_path))
    return {"statement": statement,
            "signature_b64": base64.b64encode(private.sign(encode(statement))).decode("ascii")}


def _verify(envelope: dict, public_key_path: str | Path) -> bool:
    try:
        _, Public, _ = _crypto()
        statement = envelope["statement"]
        pub = Public.from_public_bytes(Path(public_key_path).read_bytes())
        pub.verify(base64.b64decode(envelope["signature_b64"], validate=True), encode(statement))
        return True
    except (ValueError, TypeError, KeyError, OSError):
        return False
    except Exception as exc:
        from cryptography.exceptions import InvalidSignature
        if isinstance(exc, InvalidSignature):
            return False
        raise


def model_semantic_digest(model_path: str | Path) -> str:
    model = IntentClassifier.load(model_path)
    return digest({"vocab": model.vocab, "weights": model.weights,
                   "metadata": model.metadata})


def attest_qualification(qualification: dict, qualifier_private_key: str | Path,
                         reviewer: str) -> dict:
    """Call only on isolated qualification host AFTER signature + gates passed."""
    if qualification.get("verdict") != "ELIGIBLE_FOR_HUMAN_REVIEW":
        raise ValueError("cannot attest rejected or unreviewed qualification")
    for field in ("report_digest", "prereg_digest", "model_digest"):
        if not isinstance(qualification.get(field), str) or len(qualification[field]) != 64:
            raise ValueError(f"missing qualified {field}")
    if not reviewer.strip():
        raise ValueError("named independent qualifier required")
    return _signed({"schema_version": 1, "purpose": "candidate_qualification_attestation",
                    "reviewer": reviewer, "qualification": qualification}, qualifier_private_key)


def verify_qualification(envelope: dict, qualifier_public_key: str | Path) -> bool:
    try:
        record = envelope["statement"]
        q = record["qualification"]
        return bool(_verify(envelope, qualifier_public_key) and
                    record["schema_version"] == 1 and
                    record["purpose"] == "candidate_qualification_attestation" and
                    q["verdict"] == "ELIGIBLE_FOR_HUMAN_REVIEW" and
                    all(len(q[key]) == 64 for key in
                        ("report_digest", "prereg_digest", "model_digest")))
    except (KeyError, TypeError):
        return False


def authorize_candidate(qualification_envelope: dict,
                        qualifier_public_key: str | Path, model_path: str | Path,
                        prereg: dict, human_private_key: str | Path,
                        approver: str, explanation: str) -> dict:
    """Authorizes only staging a precise immutable candidate, never deployment."""
    if not verify_qualification(qualification_envelope, qualifier_public_key):
        raise PermissionError("invalid qualification attestation")
    q = qualification_envelope["statement"]["qualification"]
    if model_semantic_digest(model_path) != q["model_digest"]:
        raise ValueError("candidate model digest not evaluated")
    if digest(prereg) != q["prereg_digest"]:
        raise ValueError("wrong or modified evaluation preregistration")
    if not approver.strip() or not explanation.strip():
        raise ValueError("human approver and rationale required")
    model_bytes = Path(model_path).read_bytes()
    statement = {"schema_version": 1, "purpose": "candidate_staging_only",
                 "approver": approver, "rationale": explanation,
                 "candidate_file_sha256": hashlib.sha256(model_bytes).hexdigest(),
                 "model_semantic_digest": q["model_digest"],
                 "report_digest": q["report_digest"],
                 "prereg_digest": q["prereg_digest"],
                 "qualification_digest": digest(qualification_envelope["statement"]),
                 "permits_deployment": False}
    return _signed(statement, human_private_key)


def verify_candidate_authorization(candidate_authorization: dict,
                                    human_public_key: str | Path,
                                    qualification_envelope: dict,
                                    qualifier_public_key: str | Path,
                                    model_path: str | Path,
                                    prereg: dict) -> bool:
    try:
        if not (_verify(candidate_authorization, human_public_key) and
                verify_qualification(qualification_envelope, qualifier_public_key)):
            return False
        claims = candidate_authorization["statement"]
        q = qualification_envelope["statement"]["qualification"]
        if claims["schema_version"] != 1 or claims["purpose"] != "candidate_staging_only":
            return False
        if claims["permits_deployment"] is not False:
            return False
        raw = Path(model_path).read_bytes()
        return (claims["candidate_file_sha256"] == hashlib.sha256(raw).hexdigest() and
                claims["model_semantic_digest"] == q["model_digest"] == model_semantic_digest(model_path) and
                claims["prereg_digest"] == q["prereg_digest"] == digest(prereg) and
                claims["report_digest"] == q["report_digest"] and
                claims["qualification_digest"] == digest(qualification_envelope["statement"]))
    except (KeyError, TypeError, OSError, ValueError):
        return False


def stage_candidate(candidate_authorization: dict, human_public_key: str | Path,
                    qualification_envelope: dict, qualifier_public_key: str | Path,
                    model_path: str | Path, prereg: dict, stage_root: str | Path) -> dict:
    """Write immutable staging record; never switch a running model or launch code.

    Use an administrator-owned clean staging root. This research-grade function
    is not an adversarial filesystem boundary against a shared-UID attacker.
    """
    if not verify_candidate_authorization(candidate_authorization, human_public_key,
                                           qualification_envelope, qualifier_public_key,
                                           model_path, prereg):
        raise PermissionError("candidate not independently qualified and authorized")
    root = Path(stage_root)
    if root.is_symlink():
        raise PermissionError("staging root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise PermissionError("staging root changed to symlink")
    model_bytes = Path(model_path).read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != candidate_authorization["statement"]["candidate_file_sha256"]:
        raise ValueError("candidate changed after authorization")
    slot_id = hashlib.sha256(model_bytes).hexdigest()
    candidate_dir = root / slot_id
    # Fail on an existing slot rather than overwriting a previous signed artifact.
    candidate_dir.mkdir(mode=0o700, exist_ok=False)
    record = {"schema_version": 1, "candidate_id": slot_id,
              "authorized_for": "staging_only", "deployment_performed": False,
              "authorization": candidate_authorization,
              "qualification": qualification_envelope}
    for name, data in (("candidate.json", model_bytes),
                       ("release_record.json", encode(record) + b"\n")):
        path = candidate_dir / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
    return {"status": "STAGED_NOT_DEPLOYED", "candidate_id": slot_id,
            "path": str(candidate_dir), "model_digest": record["authorization"]["statement"]["model_semantic_digest"]}
