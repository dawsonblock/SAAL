# v4 collapse-remediation follow-up

This additive repair preserves the failed study and v3 signed artifacts. It
does not reserve compute, start Colab, or qualify a model.

## Diagnosis retained from the immutable run

The AdaLoRA run learned public patterns but degraded the refusal boundary.
The terminal checkpoint had 100 `REFUSE` to `CHEAP`/`FAST` proposals and no
`ASK` to action proposals. The best-development checkpoint remained step zero
because the preregistered guardrail prioritized invalid/unauthorized proposals
ahead of ordinary accuracy. This was a rejection of the trained candidate,
not evidence that step zero was a successful adaptation. The historical run
did not record effective learning rates at each optimizer step.

## Added in this follow-up

* A separate `saal-adaptive-job-v4` contract binds numerical diagnostics v2
  and a distinct worker identity. v1-v3 jobs still reject v4 and the v3
  unsigned proposal remains frozen rather than being silently upgraded.
* The v4 training loop records the optimizer step, AdamW group learning rates,
  scheduler identity, losses, gradients, rank telemetry and resource data.
* The v4 artifact validator requires exactly one contiguous learning-rate
  record per completed optimizer update and rejects missing, altered or
  interrupted-step logs.
* A v4 protocol validator binds ordered job descriptors, worker/policy
  identities, source digests, Colab CLI version, a one-provision/no-retry
  budget, and explicit `NOT_EVALUATED` status.
* An additive v4 local ledger enforces one-time reservations and irreversible
  terminal state within its local SQLite file. It explicitly reports
  `rollback_resistant: false`; it is not independent custody.
* Tests cover v4-to-v3 downgrade rejection, metric/policy substitution,
  stale-source rejection, reservation completeness, one-use budget behavior,
  skipped optimizer updates, and per-update learning-rate telemetry.

## Validation

Pinned local canonical validation passed **1,073 tests** across nine suites
(636 core, 38 corpus, 10 reporting, 42 project, 15 diverse, 4 reliability,
45 repair, 159 adaptive, 124 numerical repair): zero failures, errors or skips.
The environment was the project Python 3.12 macOS validation environment.
The installed Colab CLI reports 0.7.4. This verifies only local software and
the client version; **no Colab GPU, worker installation, training, artifact
replay, or runtime cleanup was run in this follow-up**.

## Still blocking a Colab campaign

1. The blinded error-case packet is still pending real independent reviewer
   attestations and adjudication. The organizer copy contains the labels and
   must stay separate from reviewers.
2. No fresh v4 signed protocol or durable budget was supplied. The failed v2
   study, its spent budget, and all reserved attempts cannot authorize a retry.
3. The v4 module currently stages an allowlisted worker and validates contracts
   and a local ledger; a complete v4 Colab campaign coordinator and portable
   v4 replay backend still need to be finished and tested.
4. The planned static-AdaLoRA/no-reallocation mechanism control is not
   implemented. The current v4 adapter allowlist is fixed-r8, fixed-r12 and
   adaptive 12-to-8 only. A study that claims rank-scheduling causality must
   add and test that control or preregister a narrower question.

The next safe step is to finish the static-control and end-to-end coordinator,
then have the independent reviewers and authorized controller return their
attestations and fresh signed admission. Do not launch training before those
conditions are satisfied. Retention, novel independent evaluation, operational
custody independence, hostile-code containment, qualification, deployment and
recursive self-improvement remain unestablished.
