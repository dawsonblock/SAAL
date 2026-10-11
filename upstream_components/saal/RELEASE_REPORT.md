# SAAL v0.5.0 — release verification

**Date:** 2026-10-08. **Release purpose:** research-grade upgrade, not safety certification or proof of AI alignment.

## Included functional upgrades

- Separate Ed25519 witness authority, independent of the evaluator, qualifier and actor by operational design.
- v5 preregistration binds independent witness public key fingerprint, evaluator fingerprint and unique study identifier.
- Witness signs preregistration and implementation source commitment; evaluator signs report with witness receipt digest included; qualifier verifies exact witness and performs deterministic full replay.
- Any missing, forged, replaced, tampered, unpinned or downgraded witness evidence rejects v5 qualification.
- Training-data admission validates one no-follow byte snapshot, exact train-only schema, field types, size limits, duplicate-key rejection and independent data authority signature.
- No external effectors, production model deployment, or online neural-weight modification included.

## Verification performed

1. In-tree unittest suite: **86/86 passed**.
2. End-to-end CLI evaluation: independent evaluator and witness key generation, v5 preregistration, witness receipt signing, offline small classifier training, Ed25519 evaluation, independent replay and qualification.
3. End-to-end qualifier returned `REJECT` for insufficient demonstrated improvement, with a successful replay. Rejection is expected evidence of a functioning promotion barrier.
4. Archive validation: SHA-256 inventory check, ZIP CRC/structure test, clean extraction and independent test execution; see terminal/release log.

## Qualification boundaries

Signatures prove authentic statements from the held key, not beneficial outcomes or trustworthiness of a signer. The synthetic verifier cannot causally establish human welfare. A claimed witness timestamp is not independent chronological proof absent an external append-only time-ordered register. A separate service, UID or host is still required for actual authority isolation. The optional LoRA adapter is not exercised by the core test suite, which trains and evaluates a small intent classifier.

Do **not** deploy this prototype for financial actions, medical decisions, communication sending, destructive file operations, robotics or other high-impact tasks.
