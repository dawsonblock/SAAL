# Reliable Learning v3

Public development experiment, not independently qualified. Human reviews: zero.
External deployment remains prohibited. Recursive improvement is not demonstrated.

The new `saal-model-job-v6` worker adds complete balanced epochs, training
diagnostics, per-class negative log likelihood, gradient norms, verified sample
streams and full-training-population initial/terminal measurements. It preserves
the historical v5 source and admission semantics. V6 uses a distinct signed worker
identity, result schema and twelve-file inventory. Overlong inputs are rejected;
the tokenizer never silently truncates corrections or authorization constraints.

Five matched ModernBERT LoRA seed pairs compare uniform sampling with replacement
against complete balanced epochs. Each branch receives the same 3,840 available
training records, initial seed, rank, learning rate, 1,000 updates and 8,000 sample
exposures. Measured token exposure can differ. One full-tuning diagnostic warms
the classification head for 100 updates before admitting encoder updates at a
lower learning rate. It is not a repeated-seed full-tuning result.

Initial, best-development and terminal checkpoints are always retained. The
historical safety-first selector uses public development only; training and
exposed regression remain diagnostic. A best checkpoint at step zero is reported
alongside the actual terminal learning outcome. Strict behavioral thresholds and
the registered numerical replay tolerance are preserved, even if runs fail.

The 600-case review pilot preserves complete counterfactual pairs, lineage and
the original data splits. Its blind review interface is ready for actual human
annotations. It is not human reviewed, does not replace the main corpus for this
controlled sampler comparison and is not a protected holdout. Generating labels,
signing a protocol or passing mechanical checks cannot supply independent review.

The sequential study is conditional on a non-initial checkpoint passing the
registered basic development screen. A failed screen skips that next phase with
an explicit reason. Comparing freshly trained replacements on old tasks is not
an actual sequential forgetting experiment. Dynamic LoRA is deferred until this
fixed-rank baseline demonstrates competent behavior.

Colab is a compute backend. Named sessions, hash-locked dependencies, actual T4
admission, finite-loss and gradient checks, cumulative deadlines, source-bound
artifacts, fresh-process selected-checkpoint replay, durable negative outcomes,
an independent local watchdog and confirmed exact-session cleanup are retained.
The time ledger is not a provider billing cap. Subprocesses and worker reports do
not provide hostile-worker attestation or independent authority. Same-admin
SQLite history is not resistant to privileged rollback.

Run from the source root in the documented hash-locked environment:

```sh
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src:reviewed-decision-corpus/src python -m pytest reviewed-decision-corpus/tests -q
python research/reliable-learning-v3/run_campaign.py run
```

The last command requires a frozen, signed `experiment/` and authenticated
official Colab CLI. It consumes real GPU resources, creates one dedicated T4
session and has no automatic retry or resume. Do not recreate the attempt ledger
to conceal an interrupted experiment. Upstream pretrained weights are recovered
using the exact official revision and file hashes, not redistributed in the ZIP.
