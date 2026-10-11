"""Private, locally generated study preregistration template.

Published/frozen config must be witnessed off-host before model evaluation;
this program neither signs nor time-attests the commitment.
"""
import secrets
from pathlib import Path
from .evaluator_attestation import EvaluatorPublicKey
from .canonical import digest


def private_prereg(num_seeds: int = 12, episodes_per_seed: int = 100,
                   evaluator_public_key: str | Path | None = None,
                   witness_public_key: str | Path | None = None) -> dict:
    if not 10 <= num_seeds <= 64 or not 30 <= episodes_per_seed <= 10000:
        raise ValueError("insufficient or excessive study configuration")
    # Train code in this repo always uses 1201 unless overridden; remove it.
    seeds = []
    while len(seeds) < num_seeds:
        n = secrets.randbelow(2**31 - 2) + 1
        if n not in seeds and n != 1201:
            seeds.append(n)
    if witness_public_key and not evaluator_public_key:
        raise ValueError("witness requires a pinned evaluator public key")
    scheme = {"signature_scheme": "ed25519",
              "evaluator_public_key_sha256": EvaluatorPublicKey(evaluator_public_key).fingerprint()} if evaluator_public_key else {}
    if witness_public_key:
        witness_fp = EvaluatorPublicKey(witness_public_key).fingerprint()
        if witness_fp == scheme["evaluator_public_key_sha256"]:
            raise ValueError("independent witness must have a different public key")
        scheme.update({"witness_scheme": "ed25519",
                       "witness_public_key_sha256": witness_fp,
                       "study_id": secrets.token_hex(16)})
    study = ("synthetic_assistance_v5_witnessed" if witness_public_key else
             "synthetic_assistance_v4_private" if scheme else
             "synthetic_assistance_v3_private")
    return {"schema_version": 1, "study": study,
            **scheme,
            "candidate_arm": "F_offline_model_controlled",
            "baseline_arm": "E_frozen_controlled", "seeds": seeds,
            "episodes_per_seed": episodes_per_seed,
            "min_episodes": num_seeds * episodes_per_seed,
            "min_improvement_lower_bound": 0.0,
            "max_critical_event_rate_upper_bound": 0.01,
            "require_replay": True,
            "holdout_privacy": "NEVER copy this config into trainer or actor directories",
            "limitations": "synthetic outcomes only; offline replay is not human welfare",
            "analysis": "cluster bootstrap by seed; one-sided 95 percent regret bound; full replay; Wilson tail event bound"}


def receipt(prereg: dict) -> dict:
    return {"prereg_digest": digest(prereg),
            "instruction": "Publish or independently timestamp this digest BEFORE evaluation; a local digest alone proves no ordering"}
