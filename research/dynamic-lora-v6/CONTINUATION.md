# Dynamic LoRA v6 — authorized continuation

Work in `/Users/dawsonblock/Downloads/SAAL-diverse-publish`. This is the user's
authorized Dynamic LoRA v6 study and release, including the 15 matched runs
and four exploratory parent-data controls. New protected evaluations and
external deployment are not authorized by any model score.

Use `/Users/dawsonblock/Downloads/SAAL-RSI-validation-envs/adaptive-v6-py312/bin/python`.
Set PATH to include `/Users/dawsonblock/.local/bin` and `/opt/homebrew/bin`.
The model cache is `/Users/dawsonblock/Downloads/SAAL-RSI-model-cache`.
Controller signing material stays in the existing private-key directory
outside this repository. Never upload it or any evaluator/custodian keys.

The source branch is `research/saal-adalora-v6`, commit
`849760f49b1fc5d0cf8d941a3757c0e33fcf0a35`. Main must remain unchanged.
The original v1 worker identity is
`sha256:c5bd510d80b383ebde3e197c069995f9f0a3c1c222ba8686e5f20558ebfd4054`.
Do not modify any of its 20 source modules or reinterpret old jobs.
CUDA-safe AdaLoRA uses a separate v2 contract and four additive worker modules.
All original v5 files and its five RESERVED jobs must remain byte-identical.

## Monitor the admitted work

Read `study-v2/colab/monitor-state.json`, `events.jsonl`, `outcomes.json`,
`registry-export.json`, the training logs and watchdog/cleanup evidence.
The outer registry export is the initial reservation, not live state.
Read the exact session identity from its plan; never stop unrelated sessions.
The controller has its own deadline and watchdog. Check that its process is
alive while progress is nonterminal. No duplicate launch, automatic retry,
budget reset, changed batch/precision/rank/data, or silently resumed training.
An absent controller with a live runtime is an incident: recover available
evidence, stop only the owned session and verify actual absence. Uncertain
network cleanup must stay unresolved rather than be called successful.

The two old campaigns `pilot-experiment` and `pilot-repair-02` remain FAILED.
The latter contains two individually completed fixed controls with all three
GPU and CPU checkpoint replays. `prepare_v2.py` authenticates those controls,
the completed four-update CUDA probe, and the separately admitted full AdaLoRA
pilot. Only `prepare_v2.py gate` can accept that composite engineering pilot.
It is not behavioral qualification and never changes the old failed budgets.

## Finish the registered comparisons

1. After all 15 study runs complete and cleanup is confirmed, from
   `research/dynamic-lora-v6` run
   `analysis/report_study.py study-v2 validation/study-v2-report.json` using
   project-relative paths. It authenticates every job, tensor bundle, initial/
   best/terminal replay and matched base/head/sampling identity before producing
   the preregistered terminal comparison. Include all five seeds and failures.
2. Only after that succeeds, use `prepare_v2.py prepare --stage DATA_CONTROL`
   with the same local controller key, then `prepare_v2.py run --stage
   DATA_CONTROL --cache ...`. This is the already authorized four-run,
   two-seed exploratory parent-data control. Preserve its separate budget and
   cleanup. Never generalize a two-seed interaction to a population claim.
3. Evaluate public regression only after each candidate is frozen, outside the
   training runtime, using `adaptive_evaluator.evaluate_frozen` and the proper
   v1/v2 portable replay backend. Validate bundles first; restore exact initial
   heads; preserve all three snapshots. Use bounded processes, report measured
   work, and never select checkpoints or restart training using these results.
   The separate bounded CPU collector is now
   `analysis/frozen_regression.py`. Invocations are recorded in
   `validation/frozen-regression-process-*.json`; inspect the latest record
   and actual process command before launching another collector. The first
   invocation (session 41905) completed the two fixed seed-47 candidates.
   The second invocation (session 68132) collects the newly completed AdaLoRA
   candidate, retaining the same plan and skipping both authenticated completed
   tasks. Output is `regression-study-v2`; logs are
   `validation/frozen-regression-study*.log`.
   It freezes the 1,240-case public regression population and each completed
   candidate before inference. Each task has a 600-second wall limit and no
   automatic retry. Do not duplicate a live collector. Once that invocation
   exits successfully, the same command can collect newly completed candidates;
   authenticated completed tasks are skipped, and failed tasks block retries.
   Command from repository root: `python
   research/dynamic-lora-v6/analysis/frozen_regression.py
   research/dynamic-lora-v6/study-v2
   research/dynamic-lora-v6/regression-study-v2 MODEL_CACHE`.
   Use a separate output plan for the four parent-data controls. The collector
   is additional post-freeze analysis, not part of the earlier 932-test QA
   count. Its actual admission-negative checks and native inference outcomes
   must be reported separately.
4. If any campaign fails, retain its failed budget and remaining reservations,
   report the scientific comparison as incomplete, and do not create another
   training admission automatically. No default-adapter promotion. Mechanism
   ablations and sequential retention require separate protocols; do not infer
   retention from independent training runs or public regression scores.

## Complete the release

The latest full QA is `colab-tests-05/verified-results.json`: source 849760f,
931 passed, one macOS-only platform skip, zero failures/errors, 932 tests total,
hash-locked Python 3.13.16 and confirmed runtime termination. Its clean local
preflight passed all 932 tests. Six semantic corruptions of a real v2 CUDA
bundle were rejected in `validation/cuda-v2-negative-artifacts.json`.

Create an accurate `RESULTS.json` and release report with terminal and selected
metrics, seeds, rank histories, actual resources, every failure and replay
outcome. Public families remain correlated and have zero human reviews.
Qualification is NOT_EVALUATED, independent generalization NOT_ESTABLISHED,
external deployment false. Preserve historical numerical failures at 1e-4.

Publish results on a separate `results/dynamic-lora-v6` branch derived from the
source branch. Inspect the remote before committing/pushing; never force push.
Stage only this task's evidence and analysis, not unrelated preexisting files,
private keys, bytecode, redundant QA input archives/chunks or model caches.
Preserve signed admission-time source snapshots and meaningful failure logs.

`tools/adaptive_capsule.py` builds the full physical release from the authentic
v5 ZIP and tracked files, checks actual LFS payloads, and binds source/evidence
commits. It requires the new RESULTS.json and controller key. Verify free disk
first: the full tracked historical payload is about 12 GB, larger than v5.
Do not strip genuine old checkpoints just to reduce size. Pretrained upstream
weights remain a separately authenticated pinned cache download.

After final report edits, pack the final ZIP, authenticate its actual digest,
clean-extract it, verify every physical member/manifest/signature and historical
frozen file, and run the canonical eight suites from that extracted tree.
Keep test logs outside the tree; disable bytecode and pytest's cache provider.
Reverify final files and ZIP bytes. Preserve detached verification evidence
without rewriting the tested archive. A prior payload test or hash sidecar
alone is not final archive verification.

Return the physical ZIP/report and exact remote branch commits. No ChatGPT
Library-saving API is available in this session; do not claim a save or a
storage-limit error without an actual attempt. Pause the run monitor only
after all owned sessions have confirmed cleanup and the final handoff is
complete, or after reporting a blocking failure that requires the user.
