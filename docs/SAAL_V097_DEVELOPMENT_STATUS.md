# SAAL-RSI v0.9.7 development status

This branch is `repair/saal-v097-source-integrity`, based on the available `repair/dynamic-lora-v6-followup` source at commit `840c0da71dcf79cab28a690d3a1584c13c95bd46`. The cited `SAAL-RSI-v0.9.6-research-upgrade.zip` and a v0.9.6 source checkout were not found in the available Downloads or Codex workspace. The current predecessor identifies as `0.9.5.dev6`; this work does not relabel it as recovered v0.9.6 source.

## Implemented in this development slice

- Added a v6 source-snapshot contract that rejects unsafe and symlinked paths, hashes exact file bytes, checks static local Python imports against an explicit file set, and marks dynamic-import review as required.
- Snapshot verification distinguishes byte integrity from publisher identity. A signed snapshot is checked against an externally supplied public-key pin, but the generated development snapshot cannot claim release eligibility or completed review.
- Added a read-only ZIP auditor for archive/member hashes, unsafe and duplicate paths, Git LFS pointers, manifest discrepancies, package/runtime version drift, source-closure coverage, and resource limits.
- Aligned `pyproject.toml`, `saal_rsi.__version__`, adaptive CLI output, and newly generated capsule metadata to `0.9.7.dev0`.
- Guarded the legacy physical-release-v1 builder so it refuses to label this v0.9.7 development source as v0.9.4.3. The v0.9.4.3 verifier remains available for its original signed format.
- Audited the available 7,470,857,286-byte v0.9.5-dev6 evidence archive. Its ZIP had 6,692 readable members, no unsafe paths, duplicates, LFS pointers, or read errors. Its root manifest omitted six checkpoint files: `run-00-7f4563c3d645` from behavior-closure-v4 and `run-06-d007a32c8446` from reliable-learning-v3, each missing `initial`, `best`, and `terminal` safetensors. Several nested packaging manifests reference standalone package files absent from the outer archive. Its package metadata (`0.9.5.dev6`) and runtime module (`0.9.4.3`) disagreed. The audit does not authenticate a publisher key.
- Generated `recovery/source-closure-candidate-v6.json` over 325 source, test, tool, configuration, and dependency files. It passes static local-import closure and byte verification, while remaining explicitly `UNREVIEWED_DEVELOPMENT`, unauthenticated, and release-ineligible.

## Validation and limits

The final 648-test collection ran in the pinned Python 3.12 validation environment and completed with exit code 0: **647 passed, 1 skipped** in 110.39 seconds. The ten new integrity tests and two release-version guard tests also pass explicitly. An initial ambient-environment run produced six dependency-preflight failures because Transformers, PEFT, and Hugging Face Hub did not match the admitted versions; the affected tests passed in isolation with the pinned environment. The run's temporary directory was removed afterward. No GPU study or training campaign was launched.

WP0 remains incomplete until the intended v0.9.6 source/assets are supplied or located, the predecessor manifest omissions are resolved from authentic sources, the closure is independently reviewed, and an independently administered publisher trust root authenticates the final package. WP1–WP6, human adjudication, Colab execution, protected evaluation, independent qualification, promotion authority, and recursive learning improvement remain unimplemented or unestablished.
