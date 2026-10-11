# SAAL-RSI v0.9.4 release report

**Status: engineering research release. Research qualification: NOT_EVALUATED. Independent generalization: NOT_ESTABLISHED. External deployment: prohibited.**

## What was built and actually tested

The release adds native ModernBERT head/full/LoRA training and native Qwen3.5 post-trained/Base completion-only LoRA; immutable upstream snapshots, native tokenizer/processor/chat identities; version-5 signed admission; strict trainable safetensors inventories; initial/selected/terminal evidence; public task oracles; matched seed distributions; shared/isolated sequence APIs; replay and fresh controls; monitored Colab execution and explicit source/evidence integrity.

Final source tests: **397 passed, 0 failed**, Python 3.12.0, macOS ARM64, 70.01 seconds. All 300 inherited tests remain included. Added tests exercise native adapter updates/frozen weights/reload, malformed jobs and metadata, unknown revisions/targets, dataset leakage and missing partitions, unsafe model claims, invalid JSON, forged metrics, replaced/nonfinite tensors, atomic validation before restoration, concurrent/consumed attempts, interrupted transfers, CLI false-success handling, old-source nonexecution and release-signature authority limits.

Final worker identity: `sha256:e3cea596f0a1ab795c105b7ba00b1a9a6a8188da1dfd59bcfd0d2601b0176f14`. After source hardening, separate small CPU and actual T4 ModernBERT/Qwen jobs completed with real updates and fresh-process replay. Their compatibility scores are excluded from learning distributions. Exact jobs/results and `release-evidence/final-source-compatibility.json` bind that scope.

The ARM64/Python 3.12 hash lock was installed locally. The Linux x86_64/Python 3.13 hash lock was installed with verified hashes in fresh actual Colab T4 environments. Official Colab CLI **0.6.0** was exercised; optional optimized attention/DeltaNet kernels were not installed. ModernBERT used admitted FP16 autocast; Qwen used separately signed FP32 after its preserved FP16 failure.

## Actual public research

24 completed ordinary learning runs, plus two retention/replay studies and two final-source T4 compatibility runs. Remote registry states: **28 completed, 3 interrupted, 4 reserved but unrun**. Reserved unused jobs are not counted as completed or as empirical evidence. Small controller CPU compatibility runs are recorded separately.

Five-seed ModernBERT terminal head accuracy averages 45.16%; terminal LoRA 66.41%, from a matched initial mean 22.97%. Selected LoRA mean is 39.53% because some runs retain step zero to avoid worse unsafe proposals. The matched lexical control is 42.19%. Full tuning is a one-seed comparison: selected 72.66%, terminal 61.72%, with nonzero unsafe proposals. Utility, pair and safety targets remain unmet.

Five-seed Qwen post-trained terminal accuracy averages 61.67% from 58.33%; Qwen Base 40.00% from 8.33%, on only 12 correlated public development examples. Malformed outputs and regressions remain; a before/after gain does not establish independent task generalization.

The one-seed no-replay sequence loses all refusal/clarification accuracy after the last task. The matched 25% replay sequence retains it, with other residual forgetting. Every stage adapter restores on the controller and reproduces recorded accuracies. This is exploratory public retention evidence, not repeated independently qualified transfer.

See `docs/PRETRAINED_RESEARCH_REPORT.md`, its factual JSON, figures, complete per-seed records and task-specific diagnostics. No failed result was rewritten as success and no qualification threshold was weakened.

## Runtime, integrity and preservation

**Nine exact named Colab sessions are confirmed absent**, including interrupted campaigns. The streaming monitor, deadline supervisor and independent local watchdog remain preserved. Loss, updates, tokens, memory, selected step, failures and cleanup are recorded. A time ledger is not a financial billing cap, global quota or hostile rollback defense.

All completed ordinary candidates replay in a fresh same-T4 process. Controller CPU replay covers all 128 development cases for the seed-17 ModernBERT LoRA candidate and all 12 cases for each seed-17 Qwen checkpoint; probabilities/strings meet declared tolerances. Both public sequences restore all six stage checkpoints and match recorded group accuracies; original sequence per-record predictions were not recorded.

The original v0.9.3 ZIP and inherited checkpoint/report bytes are unchanged. Only README, RELEASE_REPORT, package version/entry metadata and the release ZIP timestamp builder are accounted metadata changes. Final admission repairs have separate source identities; original worker ZIPs remain authenticated verification/replay-only history. Legacy authority, custody, alpha-spending, qualification and promotion implementations remain intact and reject pretrained v5 substitution.

The controller-signed release capsule binds source and evidence hashes. It establishes attribution/integrity, not independent evaluator truth or deployment authorization. Full-tuning tensors are a separate licensed evidence wrapper containing the exact authenticated inner bundle; its hashes are included in the main source package. Official base-model caches are distributed separately through pinned verified snapshots.

## Packaging validation

The main ZIP is built only after the final source/evidence freeze, with a complete SHA-256 file manifest. `tools/verify_archive.py` then performs CRC inspection, a safe new extraction, exact manifest verification, the **full real test suite in that extraction**, and a second manifest check. The actual final ZIP hash and completed results are recorded in its adjacent `.verification.json` receipt and signed archive statement; this in-package report is not a substitute for that post-build evidence.

## Remaining limits

Protected pretrained custody/evaluation integration, independent OS/machine authorities, a large reviewed independent task population, repeated transfer/retention studies, calibrated abstention qualification, learned routing, combined-model evaluation, QLoRA, dynamic rank and recursive improvement remain incomplete or untested. The local same-UID worker is not a hostile-code sandbox. No model output, score, JSON claim, source signature or public result authorizes external actions. See `docs/PRETRAINED_KNOWN_LIMITATIONS.md`.
