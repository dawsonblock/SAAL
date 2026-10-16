# SAAL-RSI v0.9.7 WP0.2 completion report

Status: **PARTIAL — tooling, full regression and checkpoint content verification pass; source reconciliation and trusted provenance remain blocked.**

## Completed

- Created `repair/saal-v097-wp02-recovery` from the v0.9.7 development baseline at `fb2e7eb000fb2a4ca50e2b5ec5034e02d3b95f4c`; the remote source-integrity ref matched that commit at inspection.
- Recorded the expected v0.9.6 archive hash and the missing-input state without substituting another archive.
- Added a hash-gated, non-importing comparator for the exact v0.9.6 candidate allowlist and a manifest reconciler that requires explicit path-scope declarations.
- Built and independently re-verified a 332-file development source snapshot. Static local imports and file bytes verify; the snapshot explicitly remains unreviewed, unsigned and release-ineligible.
- Verified all six historical checkpoint paths in the baseline Git LFS checkout: four unique payloads, each 598,445,904 bytes, match their full pointer SHA-256 values. All six parse as Safetensors with 138 tensors and match the pinned ModernBERT base inventory plus the four-class classifier conversion.
- Recorded the prior v0.9.5-dev6 archive findings without changing its historical ZIP or manifests. The former archive is still physically incomplete even though the missing checkpoint payloads now exist in the current checkout.
- Added adversarial checks for archive hash mismatch, duplicate candidates, unsafe names, manifest traversal/scope, substituted checkpoint bytes and tensor inventory mismatch.

## Validation

Pinned macOS arm64 Python 3.12.0 environment, with the repository locks and versions recorded in `release/TEST_REPORT.json`: **661 passed, 0 failed, 0 errors, 0 skipped**, exit code 0, in 152.54 seconds. The dedicated WP0.2 recovery tests also passed (13 tests). Two expected `zipfile` warnings were emitted by duplicate-member test fixtures.

## Remaining blockers

The requested `SAAL-RSI-v0.9.6-research-upgrade.zip` was not found in Downloads or the Codex workspace, and no matching published GitHub release asset was listed. Its expected SHA-256 is recorded, but no observed archive digest exists. No v0.9.6 source was imported or reconstructed, so source comparison, compatibility review and target-archive manifest reconciliation remain incomplete.

Checkpoint content integrity is verified, but the baseline Git publisher identity is not pinned to an independently administered trust root. No external source reviewer or publisher signer is established. The predecessor's nested manifest scope discrepancies remain unresolved pending the original archive/raw manifests or an explicit package-scope decision.

No authenticated release package was built. WP0.2 is **not** `INTEGRITY_VERIFIED`; WP1 execution authority, independent replay, GPU execution, qualification and external deployment remain blocked.
