# SAAL-RSI v0.9.4.3 — Reproducibility repair

This is a repair of v0.9.4.2, not the proposed v0.9.5 learning/generalization milestone. The Python3.13 mTLS test certificates now contain explicit subject/authority key identifiers and key usage. Production transport, model-learning algorithms and authority gates are unchanged.

Use the complete hydrated release archives for artifact reproduction. GitHub branch source ZIPs may contain Git LFS pointers. A pointer is not a checkpoint, PDF or nested archive. Clone with Git LFS and run `git lfs pull`, or use the separately verified release ZIPs. Physical release verification rejects pointers, missing files, changed hashes and unaccounted changes from the parent main commit.

Run `python tools/verify_reproducible_release.py --pin PUBLIC_KEY_HEX` using an independently authenticated pin; the bundled public key verifies self-consistency only. Read RELEASE_REPORT.md for exact runs, environment identities and pending gates. Older signed reports and manifests remain historical; do not run old verifiers against changed source and interpret their rejection as fresh validation.

The 600-record corpus remains a review candidate with zero completed human reviews and unresolved semantic split review. Historical fixed/dynamic LoRA learning failures are preserved. There is no new training, independent generalization or model qualification. External deployment remains prohibited.
