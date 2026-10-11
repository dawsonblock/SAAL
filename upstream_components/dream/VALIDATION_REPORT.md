# Validation report

Release: Dream-RSI-Governed-RSI-v1.0.0

Validation performed in the build environment:

- Python bytecode compilation: PASS
- Pytest suite: 16/16 PASS
- End-to-end demo qualification: PASS
- Fresh canary attestation generation: PASS
- Ed25519 promotion-manifest signing: PASS
- Append-only ledger generation and verification: PASS
- Editable package installation (`pip install -e . --no-deps --no-build-isolation`): PASS

Implementation size (excluding upstream reference and caches): approximately 1244 Python source/test lines across 22 Python files.

## Scope of the claim

This proves that the reference control plane and its local invariants execute in this environment. It does **not** reproduce the Dream-RSI paper's reported experimental results, because the official discovery agent/runtime/reproduction code was not present in the supplied repository. It also does not turn local Python process boundaries into a hardened security boundary. Production deployment still requires OS/container/VM isolation and independent key custody.
