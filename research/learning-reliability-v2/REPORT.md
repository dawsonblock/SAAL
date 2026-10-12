# SAAL-RSI Learning Reliability v2 — corrective experiment

Additive implementation and public ModernBERT learning/retention experiment. Candidate qualification: **NOT_EVALUATED**. Independent generalization: **NOT_ESTABLISHED**. External deployment remains prohibited. Human reviews: **0**.

## Implemented repairs

Fresh compact constraint examples plus replay address measured v1 failures; the original evaluation population is preserved. A trusted-state simulation preflight checks permission, scope, expiry, missing information, limits and the current objective before requiring the existing one-use capability. A separate public learning audit keeps adaptation visible alongside the unchanged safety-first checkpoint selector. Blind review export/resume, signed quorum/adjudication validation and semantic-screen binding support future reviewers. Public confidence/deferral curves keep raw errors visible. Pinned Linux CI runs the regression suite.

No historical model worker, qualification threshold, promotion policy or previous result is rewritten. Source preregistration was remotely verified before this campaign at [5b322d6](https://github.com/dawsonblock/SAAL/commit/5b322d6c9291a6b383d100c2ecf8ab7ddaa506e1). This controller-owned Git anchor is not independent custody.

## Corpus and admitted experiment

5,464 records: 3,840 balanced training records (960 per action), 768 development and 856 exposed regression. Training comprises 768 historical-v3, 1,536 prior-diverse and 1,536 fresh compact cases. Compact population adds 2,304 cases/1,152 one-condition pairs with 16/4/4 train/development/regression renderer groups. Exact former evaluation cases never move into training; fresh equivalents are disclosed as failure-informed. All 5,464 prompts passed native tokenization without truncation (maximum 166 tokens). Shared author, oracle and state-generation mechanisms prevent independent-population claims.

Three jobs: official ModernBERT-base revision 11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88, T4, FP32, batch 8, class-balanced loss, 1,000 updates, checkpoint interval 100, 2,400-second admitted per-job limit. Full tuning uses seed 47/lr 0.00003; rank-4 LoRA uses seeds 47 and 83/lr 0.0003. Same-seed sampling is matched; methods/dropout differ. Prior v1 used 600 updates and different training data: cross-release improvements cannot be attributed solely to replay or diversity. Two adapter seeds and one full seed are not a reliability campaign.

## Raw terminal results

| Method / seed | Dev accuracy | Dev complete pairs | Worst dev recipe | Prior-diverse regression | Legacy regression | Compact regression | Dev / reg critical proposals |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed_full_47 | 52.86% | 16.93% | 52.08% | 52.86% | 43.18% | 52.86% | 148 / 159 |
| mixed_lora_47 | 85.81% | 72.14% | 78.12% | 89.58% | 55.68% | 86.20% | 26 / 36 |
| mixed_lora_83 | 82.42% | 69.01% | 68.75% | 76.56% | 40.91% | 92.97% | 7 / 7 |

Critical means an action proposal (CHEAP/FAST) when the oracle requires ASK/REFUSE. No real action is executed. A deterministic guard blocking a wrong proposal does not improve raw neural accuracy or demonstrate learned safety.

## Preserve initial, selected and terminal evidence

| Run | Snapshot | Step | Dev accuracy | Dev pairs | Dev critical | Dev public screen |
|---|---|---:|---:|---:|---:|---|
| mixed_full_47 | initial | 0 | 23.57% | 0.00% | 370 | FAIL |
| mixed_full_47 | best | 500 | 46.35% | 10.94% | 0 | FAIL |
| mixed_full_47 | terminal | 1000 | 52.86% | 16.93% | 148 | FAIL |
| mixed_lora_47 | initial | 0 | 23.57% | 0.00% | 370 | FAIL |
| mixed_lora_47 | best | 1000 | 85.81% | 72.14% | 26 | FAIL |
| mixed_lora_47 | terminal | 1000 | 85.81% | 72.14% | 26 | FAIL |
| mixed_lora_83 | initial | 0 | 25.00% | 0.00% | 0 | FAIL |
| mixed_lora_83 | best | 0 | 25.00% | 0.00% | 0 | FAIL |
| mixed_lora_83 | terminal | 1000 | 82.42% | 69.01% | 7 | FAIL |

“Best” is the unchanged safety-first worker selection, not necessarily greatest learning gain. The new learning audit compares only physically retained initial/best/terminal snapshots. Intermediate metric rows without saved tensors cannot be restored. Regression and retention never select weights. Public screens retain 90% complete pairs, 80% worst recipe, correct coverage of all four classes and zero critical/invalid proposals; passing them still grants no independent qualification.

## Exact prior-case retention

Reference: frozen v1 full-tuning terminal (seed 47). All 384 prior development and 472 prior regression cases are required byte-identically. The 5-point per-recipe drop tolerance and no-increase-in-critical requirement were fixed before this experiment.

| Run | Prior dev accuracy before → after | Prior reg accuracy before → after | Retention screen | Failures |
|---|---|---|---|---|
| mixed_full_47 | 99.48% → 53.12% | 89.19% → 51.06% | FAIL | development:FAMILY_REGRESSION, development:CRITICAL_PROPOSAL_INCREASE, regression:FAMILY_REGRESSION, regression:CRITICAL_PROPOSAL_INCREASE |
| mixed_lora_47 | 99.48% → 86.46% | 89.19% → 83.26% | FAIL | development:FAMILY_REGRESSION, development:CRITICAL_PROPOSAL_INCREASE, regression:FAMILY_REGRESSION, regression:CRITICAL_PROPOSAL_INCREASE |
| mixed_lora_83 | 99.48% → 76.30% | 89.19% → 69.92% | FAIL | development:FAMILY_REGRESSION, development:CRITICAL_PROPOSAL_INCREASE, regression:FAMILY_REGRESSION |

This is descriptive performance on exposed correlated cases, not an independent estimate. Aggregate improvement can coexist with lost families; per-recipe effects and failures remain in RESULTS.json. Full sequential transfer, matched fresh-learner controls and forgetting matrices remain future experiments.

## Training-only controls and resources

| Control | Dev accuracy | Dev complete pairs | Regression accuracy |
|---|---:|---:|---:|
| training_only_unigram | 61.98% | 33.85% | 59.58% |
| training_majority | 25.00% | 0.00% | 25.70% |

Unigram and majority controls fit training labels only; fixed retrospective diagnostics, not final evaluators.

| Run | Updates | Tokens | Worker training seconds | Trainable parameters | Peak allocated GPU GB |
|---|---:|---:|---:|---:|---:|
| mixed_full_47 | 1000 | 1008107 | 720.189423191 | 149607940 | 3.425 |
| mixed_lora_47 | 1000 | 1008107 | 559.3195615139998 | 864004 | 1.928 |
| mixed_lora_83 | 1000 | 1008853 | 581.3361398389998 | 864004 | 1.932 |

Hash-locked Linux Python 3.13 installation, bounded stage/job deadlines, durable development reservations, step/loss/token/memory monitoring, chunk recovery and exact-session watchdog cleanup are retained. A controller time ledger is neither a billing cap nor rollback-resistant independent authority. Worker GPU measurements are self-reports; local artifact validation and reproduction corroborate checkpoint behavior, not hostile-worker attestation.

## Verification and remaining work

Local macOS Python 3.12: 566 core tests passed and 38 existing corpus tests passed. Fresh GitHub Linux Python 3.13: 565 core passed, one macOS-specific sandbox test skipped, 38 corpus passed. The source and preregistered evidence remain frozen. One earlier audit test fixture omitted a category and was repaired; its initial failing log is retained. The blind review UI passed Node VM export/resume/tamper tests; a real browser smoke test was unavailable and is not claimed.

Native validation checks each ten-file candidate bundle, strict tensor names/shapes/dtypes/finite values, model/dataset/source/dependency identity and checkpoint selection. Fresh GPU replay restores the selected checkpoint. Separate controller CPU replay restores all initial/selected/terminal tensors and compares every public prediction with tolerance 1e-4. Full-tuning intentionally changes the backbone; its empty frozen tensor set is not an immutability guarantee. Controller reproduction is not independent qualification.

Human semantic review, resolution of similarity flags, independently sourced protected sampling, separately administered custody/evaluation and empirical qualification remain unmet. Confidence is uncalibrated; coverage curves are diagnostics. Qwen, dynamic-rank comparisons, recursive self-modification and external tool execution were not expanded in this repair. Prior implementations and failed results remain preserved.

Completed evidence: 3 T4 jobs, 3000 updates, 3025067 training tokens; 9 CPU checkpoint restorations and 14616 prediction comparisons. All decision labels match: True. All registered probability-tolerance screens pass: False. Maximum probability difference 0.00010313093662261963 (unchanged tolerance 1e-4). The failed seed-47 LoRA development replay and one-thread numerical probe remain preserved; the probe reproduced the same discrepancy with no decision changes. Session saal-9536da587f7f4492ad83 termination was verified and separately queried after stop. No protected evaluation receipt or alpha was consumed.
