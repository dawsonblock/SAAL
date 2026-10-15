# Dynamic LoRA pilot: post-hoc collapse diagnosis

**Scope:** immutable public pilot artifacts only. No training, checkpoint selection, budget, or qualification state was changed.

| Arm | Step | Accuracy | Complete pairs | Worst family | REFUSE→action | ASK→action |
|---|---:|---:|---:|---:|---:|---:|
| adalora-12-to-8 / initial | 0 | 24.392% | 0/576 (0.000%) | 18.750% | 25 | 37 |
| adalora-12-to-8 / best | 0 | 24.392% | 0/576 (0.000%) | 18.750% | 25 | 37 |
| adalora-12-to-8 / terminal | 1000 | 61.198% | 184/576 (31.944%) | 53.125% | 100 | 0 |
| fixed-r12 / initial | 0 | 24.392% | 0/576 (0.000%) | 18.750% | 25 | 37 |
| fixed-r12 / best | 700 | 90.625% | 470/576 (81.597%) | 69.792% | 24 | 0 |
| fixed-r12 / terminal | 1000 | 90.625% | 472/576 (81.944%) | 62.500% | 27 | 0 |
| fixed-r8 / initial | 0 | 24.392% | 0/576 (0.000%) | 18.750% | 25 | 37 |
| fixed-r8 / best | 700 | 82.205% | 384/576 (66.667%) | 66.667% | 28 | 0 |
| fixed-r8 / terminal | 1000 | 92.101% | 487/576 (84.549%) | 71.875% | 33 | 0 |

## Interpretation

- Training did make measurable progress: AdaLoRA rose from 24.4% accuracy / 0.0% complete-pair accuracy to peak logged intermediate values of 64.1% accuracy at step 800 and 35.6% complete-pair accuracy at step 800, then finished at 61.2% / 31.9%. These peaks are diagnostics, not the selector's chosen checkpoint.
- Unauthorized proposals peaked at 343 at step 300 and finished at 100. At terminal, 100 were REFUSE→CHEAP/FAST and 0 were ASK→CHEAP/FAST. These are aggregate counts from saved confusion matrices.
- Step zero was selected by the preregistered guardrail ordering because the initial checkpoint had fewer unauthorized proposals than any trained checkpoint. The selector therefore did not certify the terminal candidate as an improvement.
- The training log contains loss, gradient norm, elapsed time, token count, active rank, and GPU allocation. It does not contain a per-update learning-rate field. The signed job supplies a configured AdamW learning rate; the runtime rate cannot be independently reconstructed beyond the admitted code path.
- Labels in the failing cases remain oracle-checked but not independently reviewed. A separate blinded queue is a review aid only; no human reviews or adjudications are claimed.

## Status

Qualification: **NOT_EVALUATED**. Independent generalization: **NOT_ESTABLISHED**. Retention: **NOT_MEASURED**. External deployment: **false**. A new study or training attempt requires a separate fresh signed admission.

## Terminal AdaLoRA class and family detail

| Decision class | Accuracy |
|---|---:|
| ASK | 96.181% |
| CHEAP | 53.819% |
| FAST | 40.972% |
| REFUSE | 53.819% |

Lowest terminal task-family scores:

| Family | Accuracy |
|---|---:|
| compact_v2_constraint_card | 53.125% |
| diverse_priority_register | 53.125% |
| compact_v2_amendment_brief | 54.167% |
| compact_v2_instruction_receipt | 55.208% |
| compositional5_revision_letter | 60.417% |

Terminal refusal-boundary errors were REFUSE→CHEAP: 79 and REFUSE→FAST: 21. The corresponding ASK→CHEAP/FAST counts were 0/0.

Metric policy digest: `sha256:f06b2bd263976ab5fcd1104720840236d0a23446bfdeb7bf2ff46853a137125b`

## AdaLoRA public development checkpoints

| Step | Accuracy | Complete pairs | Worst family | Unauthorized proposals | Cross-entropy | Gradient norm | Active rank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24.392% | 0/576 (0.000%) | 18.750% | 62 | n/a | n/a | n/a |
| 100 | 36.285% | 47/576 (8.160%) | 25.000% | 305 | 1.2095696926116943 | 1.5973149538040161 | 264 |
| 200 | 38.802% | 56/576 (9.722%) | 27.083% | 95 | 1.2931740283966064 | 2.4633636474609375 | 231 |
| 300 | 49.392% | 113/576 (19.618%) | 33.333% | 343 | 1.2449904084205627 | 2.3226521015167236 | 208 |
| 400 | 53.299% | 125/576 (21.701%) | 33.333% | 183 | 1.1512297987937927 | 2.4810709953308105 | 192 |
| 500 | 55.469% | 139/576 (24.132%) | 43.750% | 206 | 1.1363844871520996 | 3.5538458824157715 | 182 |
| 600 | 59.549% | 165/576 (28.646%) | 48.958% | 156 | 0.8987952470779419 | 3.3890480995178223 | 178 |
| 700 | 61.285% | 181/576 (31.424%) | 48.958% | 164 | 0.8642445504665375 | 4.477984428405762 | 176 |
| 800 | 64.149% | 205/576 (35.590%) | 51.042% | 194 | 0.7918566763401031 | 5.31024169921875 | 176 |
| 900 | 63.715% | 199/576 (34.549%) | 55.208% | 156 | 0.8977296352386475 | 4.562331676483154 | 176 |
| 1000 | 61.198% | 184/576 (31.944%) | 53.125% | 100 | 0.8262991607189178 | 4.9160332679748535 | 176 |

Configured learning rate for adapter and head: 0.0003 each; optimizer: AdamW. Actual per-update learning rate was not logged, so the telemetry does not prove the effective rate at every update.
