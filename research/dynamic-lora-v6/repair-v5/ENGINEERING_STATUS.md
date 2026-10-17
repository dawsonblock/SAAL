# Dynamic LoRA v5 repair: engineering status

Status: **development diagnostics only**. This change does not create a signed
campaign, launch training, qualify a checkpoint, or authorize deployment.

## Implemented

- Added a closed scientific-configuration schema for the 200-update static
  AdaLoRA orthogonal-regularization screen. It binds the immutable ModernBERT
  model and tokenizer identities, registered target modules, platform lock,
  dataset/source/sample-order digests, seeds, optimizer settings, head policy,
  evaluation cadence, exposure count, and checkpoint policy.
- Added four static AdaLoRA coefficients (`0`, `0.01`, `0.1`, `0.5`) and a
  matched fixed-r8 reference. The only planned ablation difference is the
  orthogonal coefficient; AdaLoRA rank allocation remains disabled.
- Kept these records distinct from `saal-adaptive-job-v4`. Every generated
  record explicitly requires a fresh signed admission and carries
  `NOT_EVALUATED`, no holdout, no resume, no automatic retry, and no external
  deployment authority.
- Added an independent PyTorch reference for PEFT 0.21.2's SVD linear update,
  including the stored-rank normalization epsilon, optional rank mask, and
  adapter-disabled path.
- Confirmed with a differential test that PEFT's `AdaLoraModel` adds its
  orthogonal term to a loss-bearing output, while the generic `PeftModel`
  wrapper in the tested `task_type=None` configuration returns the task loss
  without that augmentation. SAAL's manual regularizer path is therefore
  necessary for this configuration and must remain single-counted.
- Added diagnostics for CE versus weighted orthogonal gradients and their
  cosine/relative norm, low-rank effective-update norms and changes, rank-mask
  budget checks, per-tensor gradients/updates, Adam moments, and predicted
  clipping scale.
- Reproduced a rank-mask edge case in pinned PEFT 0.21.2: a zeroed singular
  value can receive a nonzero AdamW gradient/update and regrow unless its mask
  is reapplied. SAAL's current rank controller reapplies the saved mask after
  each successful update, which restores the forward mask, but the optimizer
  moment remains nonzero. This gives the later reference-versus-hardened
  optimizer-state comparison a concrete test case; neither policy is changed
  in this static-SVD repair slice.
- Added CPU tests for PEFT forward parity, zero initial SVD output, ten-step
  factor updates, checkpoint reload parity, regularizer aggregation, numerical
  diagnostics, masks, and static pilot configuration invariants.

## Validation performed

The new targeted CPU suite passed **18 tests** in the pinned Python 3.12
environment using PEFT 0.21.2. The existing dynamic LoRA test suite passed
**202 tests** before the final numerical additions; both the full dynamic suite
and complete repository suite are being rerun on the final source state.

The tests verify the SVD formulas on tiny synthetic layers. They do not
establish ModernBERT training recovery, A100 behavior, allocation quality,
checkpoint authenticity, or model generalization.

Final validation on the current branch completed successfully: the pinned
repository suite collected 636 tests and exited successfully, and the complete
Dynamic LoRA research suite collected 210 tests and exited successfully. Ruff
and whitespace checks also pass for the newly added files. These software
checks do not substitute for an admitted GPU pilot.

## Still blocked or untested

- No new training job has been admitted or run. The four ablation records are
  candidate descriptions, not runnable worker jobs.
- The local tensor artifacts from the earlier A100 diagnostic have not been
  independently authenticated or moved into trusted content-addressed storage.
- The A100 hardware preflight has not been run. A new wrapper requires an
  operator-configured controller trust store, validates the protocol and
  every legacy v4 job against its role-bound key, and stages only the verified
  bytes before delegating to the existing runner. This local trust store is
  not independent key custody, does not make the legacy ledger rollback
  resistant, and is not a v5 scientific-pilot coordinator. The new research
  config is not yet wired to an executable v5 trainer.
- Static AdaLoRA performance parity has not been tested. Dynamic allocation,
  behavioral qualification, retention, and independent evaluation remain
  unclaimed.
- The exact full source closure and publisher trust review remain outside this
  first repair slice.

## Next authorized engineering gate

Implement and test an isolated v5 diagnostic trainer/coordinator that consumes
only a freshly signed v5 admission, uses an externally configured trust
anchor, and logs initial/step/best/terminal CE, regularizer, gradient,
clipping, effective-update, per-class, and per-family metrics. The current
trust wrapper only gates the legacy v4 runner, so this integration is still
required. Then obtain separate authorization for the 200-update static pilot.
A successful pilot would still be a development result, not qualification.
