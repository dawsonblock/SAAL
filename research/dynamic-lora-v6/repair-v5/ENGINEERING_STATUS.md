# Dynamic LoRA v5 repair status

Status: **development implementation; no A100 campaign admitted or run**.

This slice implements a static AdaLoRA diagnostic path and hard authorization
checks. It does not change v4 job semantics, historical results, spent budgets,
qualification status, or deployment authority.

## Implemented

- Added a closed scientific config for the four 200-update static AdaLoRA
  regularization arms and a matched fixed-r8 reference. The config binds model,
  tokenizer, dataset, sample order, dependency lock, numerical metric policy,
  optimizer, seed, head policy, and checkpoint-selection policy.
- Expanded the scientific source digest to bind every Python file under
  `src/saal_rsi` plus the pretrained model registry. The dependency lock and
  model snapshot remain separately bound.
- Added a v5 ModernBERT backend. It passes the registered training seed,
  supports fixed-r8 and static AdaLoRA-r12, applies the configured orthogonal
  coefficient exactly once, and never calls the dynamic rank allocator.
- Added a bounded 200-update loop with completed-update checks, FP16/BF16
  handling, CE/orthogonal gradient diagnostics, clipping telemetry, per-tensor
  parameter changes and Adam moments, effective low-rank update norms,
  class/family/pair metrics, resource observations, and initial/best/terminal
  checkpoint replay checks. The dropout RNG uses its separately registered
  seed. Development checkpoint selection puts complete-pair accuracy first,
  then worst-family and family-macro accuracy, with safety and nontrivial
  learning gates applied before ranking. It cannot select step zero as a
  trained result; an empty feasible set is `NO_QUALIFYING_CHECKPOINT`.
- Added a signed-admission controller path. It requires a controller-owned
  public trust store, an unexpired signature binding the exact config/data/
  source/model/tokenizer/dependency/metric identities, an A100/high-RAM
  hash-lock preflight, a signed one-use attempt receipt, and a signed terminal
  outcome receipt. Receipts bind the entire signed admission envelope, the
  exact attempt and configuration, and a strictly later witness sequence.
  Failed hardware checks record a sanitized stage and consume no attempt.
  Successful completion is committed only after the result and a pre-commit
  artifact inventory have been written; uncertain authority calls are never
  retried.
- Added CPU tests for SVD reference parity, coefficient and metric-contract
  identity, admission/signature rejection, attempt-receipt replay rejection,
  preflight failure handling, pair-primary checkpoint selection, backend seed
  binding, and a 200-step training-loop lifecycle using a tiny synthetic
  fixture. The tiny fixture is only a software test, not a model result.

## Validation performed

The Dynamic LoRA research suite passed **224 tests** and the repository-wide
core suite passed **636 tests** on the current source state. The focused v5
suite passed **21 tests** after the latest receipt and preflight changes. Ruff
and whitespace checks pass for the v5 implementation and tests.

These are software and synthetic-fixture checks. They do not establish
ModernBERT/A100 training, SVD performance recovery, behavioral improvement,
checkpoint authenticity, or generalization.

## Authorization required before Colab execution

No fresh signed admission or v5 attempt-authority service configuration is
present in this checkout. Therefore `execute_admitted_static_pilot` was not
called and no Colab runtime or model training was started.

Before the ModernBERT pilot can run, the controller needs all of the following:

1. A newly signed v5 admission for the exact config, dataset, source digest,
   metric policy, dependency lock, A100/high-RAM preflight, update/exposure
   limits, wall-clock limit, compute-unit allocation, and output-size limit.
2. A controller-owned `SAAL_STATIC_PILOT_TRUST_STORE` containing the authorized
   admission public key and a distinct attempt-authority public key. Do not
   place private keys in the repository, Colab notebook, or worker.
3. A separately administered attempt-authority client that implements the
   one-use `consume_attempt` and terminal `finalize_attempt` operations and
   returns receipts signed by the pinned attempt-authority key.
4. A fresh Linux x86_64 Python 3.13 Colab runtime that passes the actual A100,
   high-RAM, dependency-version, PEFT-source, and hash-lock preflight.

The hash-lock flag is an installer assertion; runtime package versions and the
PEFT source bytes are checked directly, but wheel-origin hashes are not
independently attested by this controller. The preflight result now records
that distinction instead of claiming a verified hash-lock installation. The
signed compute-unit allocation is recorded, but Colab compute units are not
independently measured by this code. The result must say
`NOT_INDEPENDENTLY_MEASURED`; the time, GPU memory, and sample/update bounds are
the directly enforced limits. The trust-store file and preflight run in the
controller's Colab environment, so they do not establish independent custody
or hardware attestation. The controller is not a hostile-code sandbox.

## Not established

- No ModernBERT training or GPU/A100 compatibility run has occurred.
- No new admission, budget receipt, or attempt was created or consumed.
- No actual Colab cleanup was observed.
- No independent fresh-process checkpoint replay has occurred; the engineering
  fixture checks restoration within one process.
- Static AdaLoRA performance recovery, dynamic-rank benefit, retention,
  behavioral qualification, independent evaluation, promotion, and recursive
  self-improvement remain unestablished.
- The v5 attempt-authority protocol has a strict client contract, but no
  separately deployed authority service is configured in this workspace.

The older v4 campaigns and their registries, checkpoints, receipts, and failed
attempts remain historical inputs and have not been rewritten or reused.
