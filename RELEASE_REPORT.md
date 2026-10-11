# SAAL-RSI v0.9.4.3 — Reproducibility repair

This engineering repair addresses the P0 packaging and validation defects from the deep audit. It is not the proposed v0.9.5 learning/generalization milestone. Qualification NOT_EVALUATED; independent generalization NOT_ESTABLISHED; external deployment prohibited.

## Repairs

- Integrated the separately published Python3.13 certificate fixture patch: CA and leaf subject/authority key identifiers and explicit key usage. Production TLS transport is unchanged.
- Preserve real Git LFS payloads in complete physical release ZIPs. The physical verifier rejects pointer files, missing/unlisted files, hash substitution and invalid controller signatures. Default GitHub branch ZIPs may still contain pointers; use the complete assets or a hydrated checkout.
- Freeze the new physical inventory, exact parent change accounting, source/config identity, platform locks, installed package records and fresh test evidence. Parent is main commit413d167f4d85d58b9bda3a7c797611fbab4ac3d9. Older signatures are historical, not fresh proof of this edited source. The new controller key has no independently established trust identity.

## Fresh verification

| Scope | Result |
|---|---|
| Local macOS ARM64/Python3.12 combined suite |545 passed|
| Fresh Colab T4/Linux x86_64/Python3.13 combined suite |544 passed,1macOS-only skip,0failures/errors|
| Final packaging tools on separate Linux CPU/Python3.13 |15 passed plus signed archive/private-key boundary round-trip|
| CUDA numerical probe |Tesla T4,CUDA12.8,finite forward/backward/update; peak allocated18,094,592bytes|
| Fresh-process CPU restore of probe checkpoint |PASS|
| Corpus deterministic reconstruction |PASS; still zero human reviews|

The545combined cases comprise492inherited core tests,15physical-release tests and38corpus tests. The separate15-test packaging run is not added again to that total. These results are distinct from the historical542-case supplemented Colab suite and its two CUDA tests.

The Linux full run downloaded wheels with hash enforcement and installed offline from those wheels with hash enforcement using the existing pretrained-linux-x86_64-py313 lock. Wheel hashes and installed distribution inventory are included.225final source/config/test/tool components match the accepted retry input. Documentation and evidence were frozen afterward. The Mac environment was inspected against pinned versions; this turn did not perform a fresh Mac reinstall.

The first T4 attempt stopped before tests because the harness invoked pip freeze in a venv intentionally created without pip. The retry uses the host pip's --python option. A local importlib collection failure from a missing test-helper path was also corrected. Both failures remain in reproducibility-evidence. No production transport or learning defect is inferred from those harness failures.

The two dedicated T4 attempts and separate packaging CPU session were terminated with subsequent server-list checks. The pre-existing unrelated Colab session remained untouched. Logs and source identities establish recorded execution, not hostile-worker hardware attestation or independent operator custody.

## Scientific and assurance gates remain unmet

Human corpus reviews:zero. Seven synthetic generator families and17,262lexical flags still need substantive review. Public test labels remain exposed regression/development evidence. No pretrained model was trained in this repair; no fixed/dynamic LoRA advantage, transfer gain, retention result, independent assurance deployment, protected evaluation or recursive improvement was established. Historical model failures and signatures are retained with their original meanings. Hostile-code containment and independently administered trust domains remain unqualified.

Do not weaken behavioral screens or convert software-test success into qualification. The next learning milestone still requires reviewed/diverse data, matched baselines and genuinely independent evaluation. Research prototype; no guaranteed AI alignment.
