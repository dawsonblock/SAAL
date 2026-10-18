# SAAL-RSI v0.9.8 Static AdaLoRA Regularization Pilot — run05

Research screening result. This is public synthetic development evidence, not independent qualification.

Campaign: `saal098-static-reg-20261018-05`

Source commit: `f5defbcb24ce78284232b67c8528ce74a83eba32`

Status: **COMPLETED_ALL_FIVE_ARTIFACTS_VERIFIED**; qualification: **NOT_EVALUATED**; external deployment: **false**.

## What ran

Five matched arms used the same pinned ModernBERT snapshot, public training projection, seed `317`, sample order, float32 precision, optimizer settings, 200 completed optimizer updates, and 3,200 sample exposures. The A100/high-RAM preflight passed. The 1,152-example development split contains 12 synthetic families and 576 counterfactual pairs; it is correlated public development data, not a protected holdout.

## Results

| Arm | Terminal CE | Accuracy | Complete pairs | Worst family | Unauthorized proposals | False-refusal rate | Selected checkpoint |
|---|---:|---:|---:|---:|---:|---:|---|
| Fixed LoRA r8 | 0.822 | 48.1% | 15.8% | 43.8% | 25 | 83.3% | NO_QUALIFYING_CHECKPOINT |
| Static AdaLoRA λ=0.0 | 1.025 | 51.0% | 17.9% | 36.5% | 199 | 20.5% | NO_QUALIFYING_CHECKPOINT |
| Static AdaLoRA λ=0.01 | 1.158 | 49.0% | 18.1% | 30.2% | 199 | 18.2% | NO_QUALIFYING_CHECKPOINT |
| Static AdaLoRA λ=0.1 | 1.195 | 47.7% | 17.0% | 28.1% | 230 | 17.2% | NO_QUALIFYING_CHECKPOINT |
| Static AdaLoRA λ=0.5 | 1.201 | 47.0% | 17.0% | 29.2% | 231 | 18.1% | NO_QUALIFYING_CHECKPOINT |

All five attempts completed, artifacts were recovered, signed ledger transitions verified, and each initial/terminal checkpoint passed the worker's prediction-and-metric replay check. **No arm produced a qualifying trained checkpoint**; the step-zero checkpoint was not accepted as a trained candidate.

## Interpretation

The zero-regularization AdaLoRA arm had the best terminal accuracy among the AdaLoRA settings and lower terminal cross-entropy than the nonzero-regularization arms. Increasing the orthogonal coefficient from 0 to 0.5 did not improve this one-seed, 200-update screen. This is consistent with regularization pressure contributing to the loss gap, but the pilot does not isolate a causal effect across seeds or establish a best coefficient.

The diagnostics narrow the next investigation but do not identify a cause. At λ=0.5, the weighted orthogonal-gradient norm averaged about 6.2% of the cross-entropy gradient norm, with near-zero cosine similarity; at λ=0.01 and 0.1 the ratio averaged about 0.12% and 1.24%. Yet the aggregate effective-update Frobenius norm at step 200 was about 1.11 for static AdaLoRA λ=0 versus 5.23 for fixed r8. Every AdaLoRA step and 199/200 fixed-r8 steps met the configured global clipping condition. Thus the orthogonal term may contribute at λ=0.5, but it does not by itself explain the much smaller SVD updates: initialization, scaling, clipping interactions, or joint head optimization remain open hypotheses. These are worker-reported diagnostics, not independent causal measurements.

The fixed-r8 arm finished with fewer unauthorized proposals but many false refusals. AdaLoRA reduced false refusals but still made many unauthorized proposals. Both tradeoffs fail the behavioral gate; these numbers do not show a safe model or an overall winning method. Every arm was marked `NO_QUALIFYING_CHECKPOINT`.

The recorded gradient diagnostics contain separate cross-entropy and weighted-orthogonal gradient norms, their norm ratio and cosine, clipping observations, and effective update norms. They are worker-produced telemetry and have not been independently replayed. Per-run compute-unit usage is not independently measured; the account balance decreased by the amount recorded below.

The operator recorded matching local and remote SHA-256 readbacks for the signed input bundle and runner in `UPLOAD_VERIFICATION.json`. This is an operator-observed CLI record, not a Google-signed attestation. The bundle itself is excluded from this branch because it contains one-use transport credentials; only its digest, size, signed omission record, and rotation record are retained.

## Integrity and limits

The signed ledger contains 15 verified events. All five artifact ZIPs passed CRC, member-set, internal inventory, admission, receipt, and within-worker replay checks. The source archive matched the signed source closure (163 files). The one-use transport file is omitted from this public evidence package because it held bearer credentials; its signed size and digest are retained, but its credential values are not published.

The Colab balance changed from 168.43 to 166.87 CU (1.56 CU observed account delta). The supervisor's conservative rate-based upper bound was 2.249 CU. No active assignment remained after cleanup.

Limitations: one seed; short screening run; public synthetic recipes; one operator controlled admission and witness keys; no independent evaluator or hardware attestation; no independent checkpoint replay; no protected final evaluation; no retention or continual-learning study. `wheel_hash_origin` was reported as `NOT_INDEPENDENTLY_VERIFIED`. Qualification remains `NOT_EVALUATED`, and external deployment remains disabled.

## Next decision

Do not turn rank scheduling on yet. The static λ=0 result is a candidate for a fresh, separately admitted multi-seed diagnostic only if it is reproduced with matched initializations and improves authorization behavior without creating a false-refusal regression. Keep all current attempts and budgets spent; this report does not authorize another campaign.
