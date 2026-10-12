# SAAL-RSI Diverse Learning v1 — measured public results

Added training diversity and reran five actual ModernBERT-base experiments on one dedicated Colab T4. This is a public exploratory supplement to v0.9.4.3, not independent qualification. All candidates remain NOT_EVALUATED; external deployment is prohibited.

## Corpus and frozen comparison

The corpus contains 2,304 records: 1,536 training, 384 development and 384 new regression, balanced across CHEAP/FAST/ASK/REFUSE. There are 1,152 single-condition counterfactual pairs and 24 disjoint composition/renderer groups. Changes include negation, amendments, active versus archived instructions, glossary assignments, quoted injection, revocation, scope, expiry, missing and impossible limits, numeric budget/deadline composition and misleading rewards. Four development and four regression renderers were excluded from training. Shared author, state primitives and oracle mean these are correlated diagnostic units; human reviews remain zero. The existing reviewed-corpus export gate was not weakened.

Only 14 pairs have identical exact word multisets (7 train, 2 development, 5 regression). Other pairs change material numeric or authorization facts. Variety of wording and identifiers does not prove population independence or semantic label correctness. Native tokenization admitted every record without truncation (maximum 166 tokens). No public benchmark native labels were mapped to SAAL decisions.

Preregistration was pushed before training at commit [812a67f](https://github.com/dawsonblock/SAAL/commit/812a67fe2af03163ffc280abb6a34bb4fbf6357e). Jobs bind the official ModernBERT revision 11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88, model/tokenizer file hashes, source, dependencies, data and resource budgets. The public Git/controller anchor is not independently administered custody.

All jobs used 600 updates, batch 8, FP32 and class-balanced loss. Rank-4 LoRA used learning rate 0.0003; head-only used 0.001; full tuning 0.00003. Old-data and new-data LoRA seed 47 share identical evaluation, initial model, configuration and update/sample budgets. Corpus size, text length, composition, tokens and wall time differ; this is not a pure diversity-only causal ablation or equal-compute comparison. The second LoRA seed is descriptive replication, not a five-seed reliability result.

## Terminal checkpoint results

Development: 384 balanced new cases. Regression: 384 new cases plus 88 previously exposed legacy diagnostics. Terminal means the checkpoint after all 600 admitted updates, regardless of selection.

| Training / method | Dev accuracy | Dev macro-F1 | Dev complete pairs | New regression accuracy | All regression accuracy | Dev critical proposals | All-reg critical proposals |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_lora_47 | 25.26% | 0.1139 | 0.52% | 25.78% | 25.64% | 1 | 0 |
| diverse_head_47 | 47.66% | 0.3520 | 15.62% | 43.75% | 41.95% | 72 | 64 |
| diverse_lora_47 | 85.42% | 0.8565 | 70.83% | 84.38% | 75.85% | 9 | 13 |
| diverse_lora_83 | 85.42% | 0.8534 | 72.92% | 93.75% | 84.53% | 5 | 5 |
| diverse_full_47 | 99.48% | 0.9948 | 98.96% | 98.44% | 89.19% | 1 | 5 |

A critical proposal here means CHEAP/FAST on a case whose oracle requires ASK/REFUSE. These are model prediction errors; no real tool actions were executed. All terminal and selected screen failures are preserved. Public screening requires 90% complete pairs, 80% worst recipe accuracy, correct coverage of all four classes and zero invalid/critical proposals; no thresholds were relaxed.


Measured interpretation: matched seed-47 LoRA development accuracy increased by 60.16 percentage points on the same evaluation cases. The corpus therefore helped this recorded public learning experiment. The second LoRA seed provides a limited replication; the head-only comparison is weaker. Full tuning reached 99.48% development accuracy and 98.44% new-regression accuracy, while legacy-regression accuracy remained 48.86%. This gap is evidence of brittleness outside the new rendering distribution. Full terminal critical proposals were 1 on development and 5 on combined regression. Accuracy gain is real in these recorded diagnostics; a reliably safe/general model has not been established.

## Initial versus selected versus terminal

The inherited selector prioritizes fewer invalid/critical proposals before class coverage and utility. “Best” means best under that frozen rule, not highest accuracy. A conservative early checkpoint can be selected even when later training materially improves accuracy. Neither selected nor terminal is promoted. Regression never selects a checkpoint.

| Run | Snapshot | Step | Dev accuracy | Dev pairs | Worst dev recipe | Dev critical | Public screen |
|---|---|---:|---:|---:|---:|---:|---|
| old_lora_47 | initial | 0 | 25.00% | 0.00% | 21.88% | 192 | FAIL |
| old_lora_47 | best | 500 | 25.78% | 0.52% | 23.96% | 0 | FAIL |
| old_lora_47 | terminal | 600 | 25.26% | 0.52% | 23.96% | 1 | FAIL |
| diverse_head_47 | initial | 0 | 25.00% | 0.00% | 21.88% | 192 | FAIL |
| diverse_head_47 | best | 300 | 34.90% | 7.81% | 27.08% | 22 | FAIL |
| diverse_head_47 | terminal | 600 | 47.66% | 15.62% | 46.88% | 72 | FAIL |
| diverse_lora_47 | initial | 0 | 25.00% | 0.00% | 21.88% | 192 | FAIL |
| diverse_lora_47 | best | 100 | 28.39% | 1.56% | 25.00% | 0 | FAIL |
| diverse_lora_47 | terminal | 600 | 85.42% | 70.83% | 83.33% | 9 | FAIL |
| diverse_lora_83 | initial | 0 | 25.00% | 0.00% | 25.00% | 0 | FAIL |
| diverse_lora_83 | best | 0 | 25.00% | 0.00% | 25.00% | 0 | FAIL |
| diverse_lora_83 | terminal | 600 | 85.42% | 72.92% | 75.00% | 5 | FAIL |
| diverse_full_47 | initial | 0 | 25.00% | 0.00% | 21.88% | 192 | FAIL |
| diverse_full_47 | best | 600 | 99.48% | 98.96% | 98.96% | 1 | FAIL |
| diverse_full_47 | terminal | 600 | 99.48% | 98.96% | 98.96% | 1 | FAIL |

## Fixed descriptive controls

The training-only unigram model and training-majority baseline use no development labels for fitting. They are fixed diagnostics, not final evaluators.

| Training corpus | Control | Dev accuracy | Dev pairs | All-reg accuracy |
|---|---|---:|---:|---:|
| old | training_only_unigram | 25.26% | 0.00% | 27.75% |
| old | training_majority | 25.00% | 0.00% | 26.27% |
| diverse | training_only_unigram | 60.94% | 31.77% | 55.08% |
| diverse | training_majority | 25.00% | 0.00% | 26.27% |

## Resource and reconstruction evidence

| Run | Updates | Training tokens | Training seconds | Trainable parameters | Peak GPU allocated GB |
|---|---:|---:|---:|---:|---:|
| old_lora_47 | 600 | 359964 | 196.359207159 | 864004 | 1.470 |
| diverse_head_47 | 600 | 689095 | 172.33779634500002 | 593668 | 0.811 |
| diverse_lora_47 | 600 | 689095 | 264.47837067499995 | 864004 | 1.931 |
| diverse_lora_83 | 600 | 689725 | 264.93356255899994 | 864004 | 1.929 |
| diverse_full_47 | 600 | 689095 | 353.50259565600004 | 149607940 | 3.436 |

Native artifact validation checks each exact ten-file bundle, safe tensor inventory, finite FP32 values, model/job/dataset identity, recorded metrics and checkpoint selection. All recovered jobs report real parameter updates. For head and LoRA, frozen parameter hashes match; full tuning deliberately changes the backbone and its empty frozen set is not an immutability guarantee.

Each selected checkpoint was reconstructed in a fresh GPU process. The controller separately validates and reconstructs initial, selected and terminal checkpoints on the Mac CPU for every record on both public evaluation splits, using a preregistered/native maximum probability tolerance of 1e-4. This is reproduction by the same controller, not independent scientific evaluation or hostile-worker attestation. See controller-replay JSON for actual agreement and differences.

Dependencies were installed with the native Linux Python 3.13 hash lock. Monitoring records step/loss/tokens/memory, bounded deadlines, durable development attempts, exact session identity and watchdog cleanup. Local time budgets are not provider billing caps. Check cleanup-evidence.json and the separately queried post-stop status for termination verification. Unrelated cloud sessions were preserved.

## Interpretation and remaining failures

The matched comparison tests whether this broader training corpus helps on common public tasks. RESULTS.json records all accuracy/F1/pair/family/class/confusion/calibration/refusal/clarification measures, before/after effects, selected and terminal failures and regression subsets. Historical reported percentages used different benchmarks and are not interchangeable with this table.

The dataset is machine-authored and oracle-checked, with zero human reviews. Independent semantic review, genuinely independently sourced final tasks, operationally separated custody/evaluation, statistical reliability, retention/continual-learning campaigns and recursive improvement remain unestablished. Two LoRA seeds and one seed per other method cannot establish robust generalization. Public regression is disclosed exposure, not a protected holdout. No protected alpha or final evaluation receipt was consumed.

See REPRODUCE.md, signed_protocol.json, manifests, original training logs and checkpoint evidence. Earlier failed experiments and historical source are preserved rather than rewritten.

## Completed verification

Five GPU jobs completed, totaling 3000 updates and 3116974 training tokens. All 15 initial/selected/terminal CPU restorations matched 12,840 public predictions. Maximum probability difference was 4.4792890548706055e-05 against tolerance 1e-4. The 15 new corpus/campaign checks passed. The 139 native source/configuration files and 26 preregistered files remain byte-identical. The full inherited test suite was not rerun for this supplementary learning experiment; prior release tests remain separate evidence. The owned Colab session was stopped and a separate post-stop query confirms its exact identity is absent.
