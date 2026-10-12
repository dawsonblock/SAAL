"""Independent deterministic simulator replay. Not a proof of human welfare.

Re-run on an evaluator-controlled host holding the pristine simulator code,
sealed preregistration and exact model. Does not trust episode statistics alone.
"""
from .canonical import digest
from .evaluation import authenticate_study, run_study
from .evaluator_attestation import EvaluatorPublicKey
from .train import IntentClassifier


def audit_replay(envelope: dict, prereg: dict, model: IntentClassifier,
                 evaluator_key: bytes | EvaluatorPublicKey,
                 *, witness_receipt: dict | None = None,
                 witness_public_key: str | None = None) -> dict:
    if not authenticate_study(envelope, evaluator_key):
        return {"valid": False, "reason": "evaluator signature invalid"}
    if envelope["report"].get("prereg_digest") != digest(prereg):
        return {"valid": False, "reason": "preregistration digest mismatch"}
    try:
        expected = run_study(model, seeds=prereg["seeds"],
                             episodes_per_seed=prereg["episodes_per_seed"],
                             prereg=prereg, replay_only=True,
                             witness_receipt=witness_receipt,
                             witness_public_key=witness_public_key)
    except (KeyError, ValueError, TypeError, PermissionError) as exc:
        return {"valid": False, "reason": f"replay cannot complete: {type(exc).__name__}"}
    equal = (digest(envelope["report"]) == digest(expected["report"]))
    return {"valid": equal,
            "reason": "exact synthetic study replay matched" if equal else
                      "signed report differs from independently recomputed outcomes",
            "actual_report_digest": envelope["report_digest"],
            "expected_report_digest": expected["report_digest"],
            "model_digest": expected["report"]["model_digest"],
            "prereg_digest": digest(prereg)}
