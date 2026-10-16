# SAAL-RSI v0.9.7.dev0 — Development work in progress

This development branch continues from the available v0.9.5.dev6 research checkout. The v0.9.6 research-upgrade archive referenced in the proposed plan was not present in the workspace, so this branch does not claim to have recovered or verified it. The package is a development build, not a release candidate.

Use the complete hydrated release archives for artifact reproduction. GitHub branch source ZIPs may contain Git LFS pointers. A pointer is not a checkpoint, PDF or nested archive. Clone with Git LFS and run `git lfs pull`, or use the separately verified release ZIPs. Physical release verification rejects pointers, missing files, changed hashes and unaccounted changes from the parent main commit.

Run `python tools/verify_reproducible_release.py --pin PUBLIC_KEY_HEX` using an independently authenticated pin; the bundled public key verifies self-consistency only. Read RELEASE_REPORT.md for exact runs, environment identities and pending gates. Older signed reports and manifests remain historical; do not run old verifiers against changed source and interpret their rejection as fresh validation.

The 600-record corpus remains a review candidate with zero completed human reviews and unresolved semantic split review. Historical fixed/dynamic LoRA learning failures are preserved. The current WP0 work adds source-integrity tooling; it does not establish publisher authentication, operational custody independence, independent generalization, model qualification, or external deployment authorization.
