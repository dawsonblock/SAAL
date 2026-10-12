# SAAL-RSI Reliable Learning v3 — Colab progress and fixes

**The full campaign is incomplete.** Status: **NOT_EVALUATED**. Independent
generalization: **NOT_ESTABLISHED**. Human reviews: **0**. External deployment
remains prohibited. Recursive improvement is not demonstrated.

Four T4 jobs completed training, fresh-process GPU replay, artifact recovery and
native validation. The fifth completed its 1,000 updates and reported GPU replay,
but its transfer-manifest download returned not found. Its tensors were lost when
the monitored session stopped, so it is recorded as INTERRUPTED and excluded from
model aggregates. Remaining jobs were not executed. Three subsequent T4
provisioning attempts returned server-side Service Unavailable. Each failed
runtime's absence was confirmed. The separately signed head-only control was
not run. No CPU or different GPU was substituted for the admitted T4 experiments.

## Verified terminal results

| Run | Development | Complete pairs | Worst development family | Critical development proposals | Regression | Critical regression proposals |
|---|---:|---:|---:|---:|---:|---:|
| uniform-v1_47 | 85.81% | 72.14% | 78.12% | 26 | 84.58% | 36 |
| complete-balanced-epochs-v1_47 | 89.58% | 79.69% | 76.04% | 13 | 87.15% | 12 |
| uniform-v1_83 | 82.42% | 69.01% | 68.75% | 7 | 80.26% | 7 |
| complete-balanced-epochs-v1_83 | 79.17% | 63.54% | 63.54% | 5 | 78.74% | 2 |

Balanced epochs helped seed 47 but hurt seed 83. The two-seed mean development
accuracy was 84.38% for balanced epochs and 84.11% for uniform replacement.
This is not a consistent advantage or an independently sampled population
estimate. Every recovered candidate failed the fixed behavioral screen: 90%
complete pairs, 80% worst-family accuracy, correct coverage of all four classes,
and zero critical or invalid proposals. Some exposed regression families still
have zero accuracy. The registered thresholds were not weakened.

The entire-training-population terminal accuracies are measured in RESULTS.json.
For seed 47, uniform sampling reached 85.96% training and 85.81% development;
balanced reached 89.22% training and 89.58% development. These close aggregate
scores do not support calling the first pair strong training-set memorization.
They also do not establish broad transfer. The training-only unigram control
scores 61.98% on this development population; majority scores 25%.

## Implemented corrections

The additive v6 worker provides complete class-balanced epochs, a fixed 512-case
training diagnostic, full-population initial/terminal training evaluation,
per-class negative log likelihood, gradient norms, optimizer-group rates,
reconstructed exposure streams and distinct initial/best/terminal checkpoints.
Uniform replacement covers roughly 87% of the 3,840 available training cases;
complete epochs cover all of them. The matched sampler study fixes seeds,
updates, data, precision, loss weighting and learning rate. Training and exposed
regression never select weights. A step-zero safety-selected checkpoint does not
hide terminal learning or its failures.

The full-tuning warm-up diagnostic is implemented and tested mechanically but
was not run on GPU. Its two changed factors (warm-up and encoder learning rate)
would remain a single-seed exploratory diagnostic. LoRA trains adapters AND the
classification head; without the unrun head-only control this release cannot
isolate their separate contributions. No new dynamic AdaLoRA or Qwen campaign
was executed.

The new transport backend preserves the original source and adds explicit
export-complete acknowledgement, five bounded retries of immutable artifact
reads, authentication before acceptance, retained negative logs and guarded
export errors. Read retries cannot execute training commands. Verified redundant
remote payloads are removed after recovery. Nine positive and negative transport
tests passed. Its real GPU recovery path remains untested because subsequent T4
provisioning was unavailable.

The original interrupted attempt and six unstarted original reservations remain
intact. The seven continuation jobs have fresh, explicitly signed budgets and
change only budget identifiers; seeds, data and all learning parameters remain
identical. The failed provisioning attempts reuse the same unstarted continuation
reservations without resetting a ledger. This is controller-owned research
accounting, not privileged rollback resistance or a provider billing cap.

A 600-case balanced counterfactual pilot and blind review interface are included.
They preserve pairs, lineage and original partitions. Its labels remain
oracle-checked and provisional; actual human review is still required. It did not
replace the main corpus during the sampler ablation. Public labels do not become
an independent final holdout by moving them to a private folder.

Trusted-state guard diagnostics separately blocked wrong action proposals in the
compact synthetic cases. Those blocked neural mistakes remain raw errors; no
capability was issued or action executed. Confidence curves also retain errors:
for uniform seed 47, a 0.95 confidence threshold retained about half the cases and
still retained two critical proposals. Confidence does not grant authorization.

Sequential learning was not started. No recovered candidate passed its basic
learning gate, and the five-seed campaign is incomplete. Replacement scores on
exactly preserved old cases are stability diagnostics, not an actual sequential
forgetting experiment. V6 is currently development-only; existing qualification
commands have not been upgraded to admit this contract. A separately reviewed,
operationally independent protocol is required before qualification.

## Verification and completeness

Updated local suite: 605 core tests and 38 corpus tests passed. Ten additional
reporting/forgery tests passed. Updated fresh Colab CPU suite: 604 core passed,
one macOS-only platform skip, and 38 corpus passed, with zero failures/errors.
Fresh Linux CI on the transport-fix commit also passed. The earlier Colab test
payloads omitted historical archives (13 then 1 failure); those failures and
shutdown records are preserved. Recursive archive closure and a complete local
minimal-payload preflight preceded the passing cloud runs.

All 12 recovered initial/best/terminal checkpoints were restored on the local
CPU: all 19,488 decision labels agree. The unchanged 1e-4 probability tolerance
failed for uniform seed 47's best/terminal development predictions, with maximum
difference 0.000103130936623. This faithfully recorded numerical failure remains
a failure. All three other recovered runs passed that numerical screen.

All 253 preregistered files, including the 166-file prior release freeze, remain
byte-identical. Private controller keys, evaluator secrets and protected data
were not transferred to training workers. Signatures establish controller
attribution and integrity; they are not independently trusted key custody,
hardware attestation or proof that reported learning generalizes. Subprocesses
and Colab are not qualified hostile-code sandboxes.

The source branch contains the primary freeze, head-only protocol, and transport
fix/amendment. Main remains unchanged. The progress ZIP contains source, all
recovered current tensors, exact jobs and protocols, corpus/review pilot, native
results, raw errors, monitored lifecycle records, failed attempts, CPU replays,
CI evidence and physical fixtures needed by the complete core suite. It cannot
contain the lost fifth checkpoint or unrun models. No LFS pointer is accepted
as a payload. The official Hugging Face snapshot/cache is not separately bundled;
full-tuning checkpoints, when generated, necessarily include encoder weights.
Older full experiment tensors remain in their separately preserved releases.

## Reproduction and continuation

```sh
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src:reviewed-decision-corpus/src python -m pytest reviewed-decision-corpus/tests -q
PYTHONPATH=src python -m pytest research/reliable-learning-v3/tests -q
PYTHONPATH=src python research/reliable-learning-v3/tools/verify_progress_capsule.py CONTROLLER_PUBLIC_PIN
```

Fresh CPU inference uses tools/reproduce_cpu.py INDEX MODEL_CACHE --output
NEW_DIRECTORY and the exact pinned official ModernBERT snapshot. Existing
records are never overwritten. The signed amendment and reserved-session helper
are ready for a newly available T4; continuation requires the unchanged source,
existing unconsumed reservations and confirmed absence of the prior session.
There is no automatic background retraining or scheduled task.

Clean-extraction tests and archive verification are emitted as separate sidecars
after ZIP construction. Their actual records establish that check; this report
does not anticipate a passing result. The next scientific requirement remains
independently reviewed, diverse scenarios and credible behavior, rather than a
larger architecture or more public benchmark tuning.
