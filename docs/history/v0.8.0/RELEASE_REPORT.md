# SAAL-RSI v0.8.0 release validation

Research prototype; no AI alignment guarantee, real-model learning, human welfare result, external effect authorization, or production sandbox qualification.

- Supplied v0.7 manifest: 143 entries, zero hash mismatches. Original test result on this host: 116 passed / 2 macOS peer-UID failures.
- Upgraded source: **146 tests passed**, Python 3.12.0, macOS 26.2 arm64. Includes inherited feature regressions, actor behavior and negative controls, actual child executor forgery/tamper/replay checks, signed plan anchoring, fresh-seed/alpha persistence, concurrency, interrupted evaluations, replay-dependent shortlisting, and independent human-approval binding/expiry/reuse.
- Editable installation and installed CLI startup verified in a separate local virtual environment. Standalone wheel qualification is intentionally refused without the complete source/test inventory.
- Exact validation dependency versions: cryptography 48.0.1, CFFI 2.0.0, pycparser 3.0, pytest 8.4.2. Additional pinned versions are in requirements-lock.txt; this is a version record, not a hash-locked dependency supply-chain attestation.
- End-to-end CLI: discover, distinct role key generation, plan-create, anchor, qualify, and pinned-key verification. Signed example candidate **failed qualification**; promotion is prohibited. See examples/validation_summary.json for the measured bound and examples/example_negative_qualification.json for full gates/identity.
- Clean-extraction test: all 146 tests passed from a fresh candidate ZIP, with complete manifest verification before and after tests. Portable cryptographic/source inspection also passed. Final delivery ZIP is independently rechecked after the documentation/provenance freeze; exact ZIP checksum and results are recorded in companion ZIP_VALIDATION.json.

The 118 inherited tests were preserved with intentional migrations for anchored qualification and disabled legacy activation; 28 new cases extend behavior/control and negative coverage. Positive human-approval checks use an explicitly constructed unit fixture with a mocked statistical bound, not evidence of a qualifying measured candidate.

No distinct-UID sandbox, remote custodian RPC, external timestamp/transparency witness, HSM, network isolation, Linux/Windows execution, Python 3.11/3.13 execution, real LLM training, upstream Dream-RSI benchmark, or production qualification was run. Local signed preregistration and source/runtime measurements depend on trusted custody; same-UID operators can reset stores or falsify measurements. See docs/INTEGRATION_SECURITY.md and docs/AUDIT_RESPONSE.md.

Private signing keys, sealed plans, mutable replay/holdout databases, virtual environments, caches, and generated installation metadata are excluded from the ZIP. Public demo evidence is not a confidential benchmark. Inherited references and historical documents remain bundled with provenance/license limitations.
