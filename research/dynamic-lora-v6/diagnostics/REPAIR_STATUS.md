# Dynamic LoRA v6 failure diagnosis and repair status

This is an offline engineering repair on a separate branch. It does not start
training, create a signed admission, reserve or refund compute, modify the
closed study, or claim independent review or qualification.

The subsequent v4 metric-bound worker and local admission/accounting checks are
tracked in `metric-repair/FOLLOWUP_V4_STATUS.md`. That follow-up still has no
Colab campaign run; its reviewer, signed-admission, static-control and full
campaign-executor gates remain explicit.

## Findings from the immutable pilot

The pilot did not show zero learning. AdaLoRA began at 24.4% public development
accuracy and 0/576 complete counterfactual pairs. Its best observed development
checkpoint was step 800 at 64.1% accuracy and 205/576 complete pairs (35.6%);
the terminal checkpoint fell to 61.2% and 184/576 (31.9%). The step-zero best
selection was a deliberate guardrail result: the selector ranks invalid outputs
and unauthorized proposals before pair accuracy and ordinary accuracy. The
initial checkpoint had 62 such proposals, versus 100 at terminal and as many
as 343 at an intermediate checkpoint. The training run therefore learned
public patterns but did not produce an acceptable selected candidate.

The terminal AdaLoRA confusion matrix attributes all 100 unauthorized
proposals to expected REFUSE predicted CHEAP/FAST; it has zero ASK-to-action
proposals. This localizes the terminal failure to refusal/feasibility boundary
cases, but does not explain the cause. The run contains class/family aggregates,
not intermediate row-level predictions, so no per-example attribution is
claimed for intermediate spikes.

Loss components, gradient norms, elapsed time, token counts, active rank, and
GPU allocation are logged. The job binds both AdamW learning rates at 0.0003,
but the per-update learning rates are absent from telemetry. No scheduler is
declared in the admitted configuration. This is a logging gap for future
workers, not evidence of an LR-schedule failure in the old run.

## Implemented here

* `src/saal_rsi/numerical_metrics_v2.py` adds a closed, digest-bound numerical
  policy with class/family/pair supports and explicit critical-error counts and
  denominators. It delegates original arithmetic to v1 and leaves all old
  metric and artifact meanings unchanged.
* `diagnose_pilot.py` reconstructs endpoint predictions under v2 and reports
  every saved development checkpoint, step-level losses, gradients, ranks,
  selector choice, confusion-derived critical errors, and source-file hashes.
* `prepare_blind_review.py` creates two shuffled annotation forms and a
  permission-restricted organizer key. It labels the package pending; it does
  not create reviewer identities, signatures, or human decisions.
* `NEXT_PILOT_DRAFT.json` defines a 12-run matched public screening design and
  its preconditions. It is intentionally unsigned, unreserved, and non-executable.

## Still required before another campaign

1. Two real, independent reviewers must complete the blind forms and an
   adjudicator must resolve disagreements. Current source labels are
   oracle-checked but not independently reviewed.
2. The v2 policy must be integrated into a new, allowlisted worker and exact
   artifact/replay validator. Current signed v3 worker jobs bind metric v1;
   changing that interpretation in place is prohibited.
3. Local tests must validate the new worker's per-update learning-rate logs,
   metric replay, selected checkpoint, adapter inventory, and static AdaLoRA
   control. This branch contains the v2 diagnostic contract and analysis tools,
   not a v4 training worker.
4. An authorized controller must issue a fresh signed protocol and durable
   budget. Nothing here authorizes execution. No Colab campaign was started.

Even after these items, the 3-seed pilot is only a public screening experiment.
Retention, mechanism attribution beyond this pilot's controls, new independent
task families, operationally independent custody, hostile-code containment,
protected qualification, external deployment, and recursive self-improvement
remain separate and unestablished.
