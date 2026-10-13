# SAAL-RSI Evidence Closure v5 — corrective release

The identified software defects are repaired and tested. Model reliability is
still unestablished. **No new model was trained:** Google Colab rejected three
bounded T4 provisioning requests with `Service Unavailable`. Every failed session
was confirmed absent. All five original signed training jobs remain RESERVED;
no job started, no training attempt was retried, and no budget was reset.
Independent qualification is **NOT_EVALUATED**. External deployment is prohibited.

## Actual fixes

New versioned controller entry points prevent evidence-write failure from
skipping cloud teardown, validate plans before launching a watchdog, and use the
watchdog's original session identity when its state file is missing or corrupted.
Cleanup queries status even after a failed stop transport. A disk-full fallback
can execute cleanup without requiring a writable log, while retaining an evidence
failure rather than reporting complete success. Recovery checks available storage
and preserves the original interruption when JSON export also fails.

Artifact validation now requires all registered checkpoint steps, matches initial,
selected and terminal metrics with history, rejects same-step tensor substitution,
and checks attention/platform identity and resource/gain accounting. Negative
tests reproduce metadata omissions accepted by the preceding validator and show
that the new validator rejects them.

A versioned public bootstrap orders family IDs explicitly; historical validation
reconstructs the old worker's insertion order from signed records. A corrected
pair-error audit counts invalid probability outputs as failed pairs even when
their textual labels happen to match. The old sources and old reports are preserved.

The exact structured-state planning control correctly applies consent, scope,
expiry, known feasibility, missing information and the current objective. An
exhaustive test checks 2,592 combinations against the existing oracle. Mechanical
state/label agreement also holds for 5,760 existing corpus records; 856 historical
records lack that state audit. The control receives trusted structured facts,
does not parse language, and does not authorize execution. Its correctness is
not neural accuracy or independent behavioral qualification.

## Stronger supervision and fresh challenge evidence

The new corpus adds **1,152 records / 576 material counterfactual pairs**:
768 training, 192 public development and 192 exposed regression. It balances
archived preferences within each renderer and widens numerical values while
preserving comparison semantics. Every one of the 6,616 parent records remains
unchanged. The complete dataset has 7,768 records: 5,376 training, 1,152 development
and 1,240 regression. All fit the admitted 256 native tokens; maximum is 251.
There are **zero human reviews**. Shared authors, generators and oracles remain;
the added public renderer groups are not independently sampled final families.

Fresh CPU inference uses the unchanged full-tuning seed-47 and LoRA seed-211
terminal checkpoints. No adaptation is performed. Old and new populations are
reported separately in `FROZEN_CHALLENGE.json`, with prompts and complete errors
in the corresponding audits.

| Frozen model and population | Accuracy | Complete pairs | Critical proposals |
|---|---:|---:|---:|
| Full tuning, old development | 97.19% | 94.38% | 3 |
| Full tuning, new development | 37.50% | 9.38% | 6 |
| Full tuning, old regression | 95.42% | 91.98% | 7 |
| Full tuning, new regression | 41.67% | 10.42% | 7 |
| LoRA, old development | 94.48% | 88.96% | 12 |
| LoRA, new development | 52.08% | 22.92% | 45 |
| LoRA, old regression | 91.51% | 84.92% | 19 |
| LoRA, new regression | 45.31% | 19.79% | 60 |

Every population fails the complete behavior screen. The two chosen frozen models
are diagnostic references, not a matched inference about LoRA versus full tuning.
The performance drop exposes dependence on narrow instruction/numeric patterns.
It does not establish a single internal cause or prove that added training fixes
it. Mapped full-tuning development errors include 22 wrong-current-objective
choices, two known-infeasible proposals, and three consent-related errors.
Some wrong answers have softmax confidence above 95%; confidence alone cannot
repair these failures.

The signed, unstarted Colab pilot compares parent and expanded training at seeds
47 and 83 under the same rank-4 LoRA, update and sample-exposure budgets. One
full-tuning seed is exploratory. The experiment remains pending because T4
provisioning was unavailable. No learning-gain claim is made for the new corpus.

## Verified engineering evidence

Fresh Colab CPU QA ran seven separate suites in hash-locked **Python 3.13.16**:
**789 passed, one macOS-only skip, zero failures or errors**. Core: 635 passed
and one skip; corpus: 38; historical reporting: 10; previous project: 42;
previous diversity: 15; previous reliability: four; new repairs: 45.
The exact payload SHA-256 is
`e43b6f3c5cf56fbaec176438f4b8ae7d170c04e1f97d9e12a837c31b47c0cee3`,
recorded against source commit `5dcdcc03f5c54ff37589a20043a78ce509e1ceb0`.
The same clean payload passed **790 tests locally** on pinned macOS Python 3.12.
Later additions are reporting, delivery tools and bounded provisioning drivers;
their scope is disclosed separately. CPU tests are not a fresh T4 training probe.

The original 253 frozen v3 source/protocol files remain byte-identical. All 23
historical current model roles and their 69 native checkpoints are retained in
the full physical ZIP. The prior replay outcome remains 21 passes and two
numerical failures at 1e-4. Targeted fresh CPU probes reproduced both failures
exactly; changing thread count did not fix them and labels still agree. Their
failures were not reclassified or pardoned.

The physical archive carries a hash manifest, published source identity and
controller signature. **Fresh clean-extraction verification and tests of the
final v5 archive have not been completed:** work stopped at the user's request.
The completed clean-payload tests above apply to the Colab QA input, not to this
final full archive. Pinned upstream ModernBERT weights remain a
separately verified cache prerequisite. GitHub automatic source ZIPs may contain
LFS pointers; the supplied physical ZIP contains checkpoint payload bytes.

## What still needs independent work

Reliable raw decision behavior and transfer have not been achieved. The next
learning experiment needs an available admitted T4 runtime, real training/reload,
and old/new failure reporting. The meaningful design direction is verified
semantic fact extraction followed by exact constraint planning; mathematical and
permission checks should not depend solely on a four-label neural prediction.
Extracted facts must be independently authenticated or validated. A model claim
of permission can never supply authority.

Two authenticated human reviewers with adjudication, separately sourced final
tasks, independently administered custody/evaluation/witness services, and a
frozen passing candidate remain unmet. Same-UID subprocesses are not a hostile
worker security sandbox; worker signatures do not attest GPU execution or honesty.
Time quotas are not provider billing caps. No production, dynamic-rank advantage,
continual-learning or recursive-improvement claim is established by this release.
