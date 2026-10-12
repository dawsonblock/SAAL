# Reproduce the corrective experiment

This is a public, unqualified learning supplement to v0.9.4.3. Start with REPORT.md, RESULTS.json and PACKAGE_LAYOUT.json in the complete capsule. Model-cache weights, full historical fixtures and the v1 full-tuning retention reference are separate dependencies. GitHub source ZIPs may contain LFS pointers: they are not the complete candidate capsule.

The source was preregistered at commit `5b322d6c9291a6b383d100c2ecf8ab7ddaa506e1`. The 166-entry PREREGISTERED_FILES.json must continue to match physical bytes. The native v5 worker identity is unchanged. New review signatures must come from actual enrolled reviewers; current human review count is zero.

## Environment and software checks

Use a fresh virtual environment for a supported platform. `configs/dependency-locks/pretrained-macos-arm64-py312.txt` is the local Python 3.12 profile; `pretrained-linux-x86_64-py313.txt` is the Linux Python 3.13 profile. Select the actual platform lock; install with `python -m pip install --require-hashes -r <matching-lock>`. Python version, architecture and dependency identity must match admission. The blind UI logic tests require Node 22; actual runs used 22.22.3.

From the **complete hydrated repository**, run `PYTHONPATH=src python -m pytest tests`. The existing reviewed-corpus suite requires `PYTHONPATH=src:reviewed-decision-corpus/src python -m pytest reviewed-decision-corpus/tests`. Those full core checks passed locally and on fresh Linux CI; the source commit's CI evidence is bundled.

The complete **candidate capsule** includes source/configuration, all new checkpoint payloads, the retained historical LoRA fixture and review component. It excludes the large historical `examples/` tree. Its clean test command is:

```bash
PYTHONPATH=src:reviewed-decision-corpus/src python -m pytest \
  tests/integration/test_decision_guard.py \
  tests/integration/test_public_review_v2.py \
  tests/integration/test_reliability_corpus.py \
  tests/integration/test_learning_audit_v2.py \
  tests/integration/test_reliability_cli.py \
  research/learning-reliability-v2/tools/test_postprocessing.py \
  reviewed-decision-corpus/tests
```

Do not interpret tests failing due to deliberately excluded legacy fixture payloads as a clean capsule full-suite run. Use the complete repository/LFS prerequisites for the full core suite.

## Physical archive and tensor admission

From the clean capsule root:

```bash
PYTHONPATH=src python research/learning-reliability-v2/tools/verify_capsule.py
```

The default bundled public pin verifies controller self-consistency. Authenticate it independently before relying on that controller identity. Verification checks physical hashes, rejects LFS pointers/symlinks, preserves all preregistered bytes, reruns native candidate/tensor/metric validation, reconstructs learning audits and verifies recorded CPU/cleanup evidence. Faithfully recorded negative results verify as evidence; they remain failed screens. Seed-47 LoRA CPU replay exceeded the registered probability tolerance slightly despite exact decision agreement; a controlled one-thread probe reproduced it. No tolerance was relaxed. This verifier does not rerun GPU training or independent qualification.

## Fresh checkpoint inference

Fetch the exact official upstream snapshot through the authenticated registry:

```bash
PYTHONPATH=src python -m saal_rsi.pretrained_cli fetch \
  --repository answerdotai/ModernBERT-base \
  --revision 11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88 \
  --cache /absolute/model-cache
```

On a CPU-only host (or with `CUDA_VISIBLE_DEVICES=''`), replay each of indices 0, 1, 2 into a **fresh output directory**. Example:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python \
  research/learning-reliability-v2/tools/replay_results.py \
  0 /absolute/model-cache /absolute/fresh-replay
```

All three retained snapshots are restored, and every development/regression prediction is compared to the GPU record. Recorded reports are preserved: attempting to write over one fails. Declared tolerance is 1e-4 with exact decision agreement. This reproduces behavior under the pinned runtime; it is not an independent final evaluation.

## Review, learning and retention

Open `research/learning-reliability-v2/corpus/blind-review.html` locally. It starts with no labels, supports partial export/resume and makes no network requests. The UI has unit logic tests; a live browser test was unavailable. `python -m saal_rsi.reliability_cli --help` exposes corpus reconstruction, review packet/signature/quorum checks, snapshot audit and exact-case retention comparison. Incomplete/rejected/mismatched reviews cannot count as public approval. Key signatures do not establish human identity or administrative independence.

Re-running `tools/report_results.py` needs the preserved v1 full terminal bundle under `research/diverse-learning-v1/experiment/artifacts/run-04-4f133755fa72`, including all its tensor payloads. Obtain the complete v1 capsule or hydrate `results/diverse-learning-v1` commit `f05876e43408f4300fc30b923ca0e2852feb44b3` with Git LFS. The public parent pin, signed job and inventory digest are fixed in signed_protocol.json. Missing prior cases or substituted records fail retention validation. Check confidence/coverage curves separately; deferral is not correct ASK behavior.

## Repeating cloud training

Do not rerun the consumed campaign directory or reset its durable ledger. Create a new versioned workspace, controller key, signed protocol/jobs and externally recorded public preregistration. Declare new compute budgets and all retry attempts. `run_campaign.py` binds the original source/job/sampler identities; the native Colab backend enforces dedicated session identity, Linux hash-lock installation, deadlines, exact artifact recovery and watchdog cleanup. Existing session IDs are historical evidence, not permission to stop another runtime.

No evaluator/custodian private keys, protected labels or final receipts are sent to workers. Colab is compute, not an independent security authority. Separate role administration, human review and genuinely protected final sampling remain necessary. Public screening or training success never grants external execution authority.
