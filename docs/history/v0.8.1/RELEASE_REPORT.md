# SAAL-RSI v0.8.1 release validation

Research prototype; no AI alignment guarantee, real-model learning, human welfare result, external effect authorization, or production sandbox qualification.

- Supplied v0.7 manifest: 143 entries, zero hash mismatches. Original test result on this host: 116 passed / 2 macOS peer-UID failures; v0.8.0 repaired these and passed 146 tests.
- Repaired source: **175 tests passed in 26.45 seconds**, Python 3.12.0, macOS 26.2 arm64. The 29 added cases exercise re-signed inconsistent reports, receipt/key identity failures, malformed actor decisions, real-child blocked-write deadlines, forged child replies, and negative file-handling cases. All previous behavior, permission, holdout, selection, and approval tests remain included.
- Editable installation and installed CLI startup verified in a separate local virtual environment. Standalone wheel qualification is intentionally refused without the complete source/test inventory.
- Exact validation dependency versions: cryptography 48.0.1, CFFI 2.0.0, pycparser 3.0, pytest 8.4.2. requirements-lock.txt records versions, not a hash-locked dependency supply-chain attestation.
- End-to-end CLI: discovery, distinct role key generation, plan creation, custodian anchoring, qualification, and pinned-key verification. Regenerated schema-v3 example candidate **failed qualification** with exploratory lower bound -0.058055555466666665; promotion remains prohibited. Full verification recomputes statistics and consistency against signed evaluation rules. Signature/source inspection also passed.
- Clean-extraction test: 175 tests passed from a fresh candidate ZIP, with complete manifest verification before and after tests. The exact final delivery ZIP is independently rechecked after documentation/provenance freeze; checksum, counts, timing, and captured results are recorded in companion SAAL-RSI-v0.8.1-ZIP_VALIDATION.json.

The original 118 tests are retained with intentional qualification/activation migrations; v0.8.0 added 28 and v0.8.1 adds 29. Positive approval unit checks use deliberately constructed coherent statistics and a mocked bound during fixture creation, not measured evidence of a qualifying candidate.

No distinct-UID sandbox, remote custodian RPC, external timestamp/transparency witness, HSM, network isolation, Linux/Windows execution, Python 3.11/3.13 execution, real LLM training, upstream Dream-RSI benchmark, or production qualification was run. Same-UID operators can modify stores, keys, source, or measurements; signed consistency checks do not prove honest evaluation. SQLite file preflight does not eliminate parent-path replacement races. Bootstrap confidence coverage remains unestablished. See docs/REPAIRS_v0.8.1.md, docs/INTEGRATION_SECURITY.md, and docs/AUDIT_RESPONSE.md.

Private signing keys, sealed plans, mutable databases, virtual environments, caches, and installation metadata are excluded. Historical v0.8.0 evidence is retained under docs/history/v0.8.0 and is not current-runtime qualification. Public demo evidence uses public seeds. Inherited references remain bundled with provenance/license limitations.
