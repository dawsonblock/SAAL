# Evidence Closure v5

This is a corrective source release and a registered public learning pilot.
Training, public screening, independent qualification and execution authority
remain separate. Model qualification is NOT_EVALUATED and external deployment
is prohibited. No historical model or negative result is overwritten.

The diagnosed software defects are repaired through new versioned entry points:

- `model_monitor_v3`: validates plans before launching a watchdog; stops work on
  monitoring evidence failure; attempts teardown and status even with a full
  disk or failed stop transport. The independent watchdog uses its original
  session identity when state JSON is missing, corrupt or substituted.
- `epoch_backend_v6`: uses that lifecycle, admits local recovery capacity before
  training and transfer, and preserves the original interruption in SQLite when
  JSON evidence export also fails. Workers and optimization semantics stay frozen.
- `epoch_artifacts_v3`: requires the registered checkpoint schedule, matches all
  named snapshots with their history, checks same-step tensor identities, runtime
  attention/platform identity, resource accounting and reported adaptation gain.
  It adds checks to existing safe tensor and signature validation.
- `family_gain_v2`: orders family IDs explicitly. The old bootstrap depended on
  dictionary insertion order; canonical JSON sorted those keys. Historical gain
  validation reconstructs the old worker's order from the signed record list.
- `behavior_screen_v2`: a pair with invalid probabilities fails even when its
  textual decision matches. Previously the error list could omit that pair.
- `structured_choice`: an exact planning control for trusted structured facts,
  including consent, expiry, scope, hard limits, missing information and current
  objective. It cannot authenticate text or authorize execution. Its performance
  must not be reported as neural accuracy or independent generalization.

New supervision teaches both correction directions within each renderer and
widens cost/time values while preserving material single-condition pairs. It
adds 768 training and 384 exposed diagnostic records in 576 pairs. The 6,616
parent records remain unchanged. All inputs fit 256 native tokenizer tokens.
There are zero human reviews; state generators, operators and oracles are shared.

The registered pilot compares parent training and expanded training for fixed
rank-4 LoRA at seeds 47 and 83, with identical model initialization, update counts,
optimizer settings and 16,000 sample exposures. Both use identical old/new public
diagnostics. One additional full-tuning seed 47 is exploratory. Report initial,
selected and terminal snapshots, all failures, and old/new populations separately.
Five runs have distinct signed jobs and durable reservations; no automatic
training retries or protected evaluation are allowed. Smaller per-example exposure
in the expanded corpus remains a disclosed confound. This is a two-seed pilot,
not a population-level demonstration of generalization.

The two historical numerical replay failures remain failures at 1e-4. Targeted
CPU probes reproduced them exactly and found no label disagreement; changing
thread count did not resolve them. PyTorch documents that mathematically identical
floating-point computations can differ across execution paths and devices.
[Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).
The probe is diagnostic, not grounds for relaxing a registered acceptance rule.

Human semantic review, independently constructed final tasks, independently
administered custody/evaluation, and hostile-worker OS containment remain external
requirements. Neither more synthetic records nor a signing key establishes those
requirements. New Qwen, dynamic-rank and recursive-improvement claims are excluded
until meaningful behavior and retention evidence support them.
