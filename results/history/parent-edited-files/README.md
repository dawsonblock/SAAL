# SAAL-RSI v0.9.4.2 — Validation repairs

This release fixes reproducible validation defects in v0.9.4.1. It preserves the historical model workers, checkpoints, training predictions and negative research results.

- Malformed or inconsistent classifier probabilities cannot pass sequence replay; invalid numerical tolerances fail closed.
- Public checkpoint selection explicitly rejects missing pair metrics and mixed measured/unmeasured pair coverage. Use null consistently when pairs are unmeasured.
- Artifact-chain receipts must satisfy attempt-specific maximum alpha and plausible event sequence. This is additional consistency checking, not proof of an independent witness deployment.

Read RELEASE_REPORT.md for actual validation results and limitations. Historical documentation and reproduction instructions remain in docs/. The previous README and report are preserved under corrective-evidence/history/v0.9.4.1/.

Candidates remain NOT_EVALUATED. Independent generalization is not established. External deployment remains prohibited. This patch performs no new training or Colab run and does not improve previously recorded model accuracy.

Validate with the pinned environment described in docs/CORRECTIVE_REPRODUCTION.md:

```sh
PYTHONPATH=src python -m pytest
python tools/verify_validation_repair_history.py
python tools/verify_corrective_history.py
python tools/verify_corrective_release.py --pin YOUR_PINNED_RELEASE_PUBLIC_KEY
```

Release signatures prove controller attribution and file integrity only. Authenticate the public-key pin independently. Heavy pretrained weights remain outside this source ZIP.
