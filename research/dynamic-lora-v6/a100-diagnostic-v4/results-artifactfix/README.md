# SAAL-RSI A100 diagnostic pilot — results handoff

This branch records one completed, single-seed engineering/diagnostic campaign. It does not qualify a model. The campaign terminal record says `COMPLETED`, cleanup was verified for session `saal-09c048d1ed6a4e129eca`, and deployment remains disabled. It used 4.49 Colab compute units under an 80-CU cap. Earlier interrupted attempts are preserved in `attempt-history/`; their budgets were not reset or imputed as completed.

## Results

| Method | Terminal accuracy | Complete pairs | Worst family | Unauthorized proposals | Selected step |
|---|---:|---:|---:|---:|---:|
| Fixed LoRA r8 | 93.1% | 86.5% | 79.2% | 21 | 0 — no trained checkpoint qualified |
| Fixed LoRA r12 | 89.0% | 79.2% | 67.7% | 15 | 0 — no trained checkpoint qualified |
| Static AdaLoRA r12 | 64.1% | 34.4% | 54.2% | 180 | 0 — no trained checkpoint qualified |
| Dynamic AdaLoRA 12→8 | 63.5% | 33.7% | 55.2% | 170 | 0 — no trained checkpoint qualified |

Fixed-r8 improved on public diagnostics, but all four terminal checkpoints failed the behavioral selection gate; step zero was retained with `NO_QUALIFYING_TRAINED_CHECKPOINT`, not counted as a learning win. Dynamic AdaLoRA performed 70 rank-allocation events and reduced aggregate active rank 264→176, yet its terminal pair score was far below fixed-r8. This is one seed on correlated public synthetic tasks, not independent generalization or evidence for changing the default adapter.

## Hardware and execution caveat

The original controller hardware-preflight log contains a probe exception and no accepted structured success observation. All four worker results report the signed target runtime (A100-SXM4-80GB and high RAM), but those are worker self-reports, not independent host attestation. The campaign remains recorded as completed; this package does not hide or rewrite the discrepancy. A new versioned controller now rejects an exception or missing/duplicate hardware observation before training. That fix needs a new admission before a future run.

## Package contents

`final/` contains the ordered post-hoc report, campaign and cleanup records, per-run metrics, fresh-process replay receipts, compressed training/rank telemetry, and compressed signed job manifests. `SHARED_INPUT_IDENTITY.json` records dataset/sampler identities. `PRESERVED_ATTEMPT_HISTORY.json` and `attempt-history/` preserve prior failures. `LOCAL_ARTIFACTS_NOT_PUBLISHED.json` lists every local checkpoint path, size, and SHA-256; tensor bytes are not included because Git LFS upload is unavailable, so this branch is not a standalone replay package.

The signing identity is a local development key; independent publisher trust is not established. Qualification is `NOT_EVALUATED`; retention and independent generalization remain unmeasured/unestablished; external deployment is false. No protected final data was used.
