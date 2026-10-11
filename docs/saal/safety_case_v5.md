# Bounded safety case (research prototype, 2026-10-08)

## Claimed boundaries

**C1 — Synthetic permission check.** The demonstration authority denies unrecognized and blocked actions, binds a single-use capability to an action/delegation, enforces per-delegation cumulative cost and action limits, and supports revocation. Evidence: `tests/test_authority.py`, `tests/test_budget_v2.py`, `tests/test_v3_security.py`. **Limitation:** real side effects absent; same-UID untrusted code may circumvent the demonstration; claims are not about real-world tool security.

**C2 — Independent signed v5 study.** A v5 study requires two distinct signed authorizations: a preregistration witness receipt and an evaluator attestation. Each is bound to the same frozen preregistration, installed source digest, candidate digest, and report. Evidence: `tests/test_v5_witness.py` and `tests/test_v4_attestation.py`. **Limitation:** independence is an operational property, not established by unit tests; witness timestamps require external anchoring.

**C3 — Independent numerical result audit.** The qualifier recomputes the entire simulator study on the exact model/preregistration and checks statistical and safety criteria. Evidence: v3 replay tests and `test_v5_signatures_and_exact_replay`. **Limitation:** replay of a simulator does not verify human benefits in the real world.

**C4 — Offline-only candidate admission.** Signed training-data manifests bind an exact byte snapshot to a specific train-only purpose. Signed staging permission does not authorize a model deployment. Evidence: `tests/test_v5_admission.py`, `tests/test_release_v2.py`. **Limitation:** an independent reviewer must really perform data checks; signature alone cannot establish consent or absence of dataset poisoning.

## Critical failure paths still open

- Trusted verifier, independent signer, or human-approval infrastructure could itself be compromised.
- No external independent executor validates real effects against an issued capability.
- No proven general human-welfare utility or causal attribution mechanism exists in this architecture.
- Agent could misgeneralize, deceive or reward-hack if an actual LLM/RL learner were introduced.
- Long-horizon autonomy, real memory privacy, adversarial distribution shift and supply-chain risk have not been qualified.

**Deployment decision: RESEARCH ONLY.** No high-impact autonomy authorization and no general alignment claim.
