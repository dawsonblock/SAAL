# SAAL-RSI Dynamic LoRA v6 — final research report

Version 0.9.5.dev6. Report frozen 2026-10-14T15:34:10.091405+00:00.

**Study: FAILED / incomplete. Qualification: NOT_EVALUATED. Independent
generalization: NOT_ESTABLISHED. External deployment: prohibited. Fixed LoRA
remains the default; no adapter is promoted.**

The complete software and recovered evidence are preserved. This is a physical
research release, not a completed five-seed experiment or an alignment result.

## What actually ran

The excluded-seed engineering pilot completed all three 1,000-update methods,
with initial, selected and terminal checkpoint replay on GPU and CPU. The
composite pilot passed its engineering gate. Its behavioral screens failed.
The two older failed campaigns retain their original FAILED states.

The new study admitted 15 runs: fixed r8, fixed r12 and AdaLoRA init12/target8
for seeds 47, 83, 109, 157 and 211. It ended with **10 COMPLETED, 1 INTERRUPTED
and 4 RESERVED**. Each accepted run performed 1,000 completed optimizer updates
and 16,000 sample exposures. The interrupted worker also reports 1,000 updates,
but its artifact failed controller admission and is excluded from accepted
results. Seeds 47, 83 and 109 have all methods; seed 157 has only accepted r8;
seed 211 did not run. No replacement training was launched or budget reset.

The named T4 session `saal-c9456df8eb3d4554bd7b` stopped on failure, and exact-session
absence was confirmed both by the controller and a fresh status query. The
controller and watchdog processes are absent. Unrelated runtimes were untouched.
The four parent-data controls were **not started**: their prerequisite failed.

## Failure diagnosis

The fixed-r12 seed-157 bundle was recovered with matching physical inventory
hashes. Its initial/best/terminal GPU replay predictions agree at the registered
1e-4 tolerance. The strict controller validator rejects the best and terminal
snapshot metrics: recorded Brier score `0.0881157183233161`, locally recomputed
`0.08811571832331609`, absolute difference `1.3877787807814457e-17` (one ULP).
Every other recomputed metric matches. No controller CPU checkpoint replay was
completed for this rejected bundle.

This is an observed portability defect in exact comparison of a derived
floating-point diagnostic. It is not evidence of improved learning, corrupt
weights, or a safety qualification failure. The original rejection, job,
worker source, signed protocol, tensors and budget are preserved. No metric,
tolerance or old admission was changed to accept it. A future repair requires
a new versioned metric policy: deterministic canonical recomputation or narrowly
specified numerical comparison for named derived metrics, with exact counts,
labels, hashes, identities and authority gates retained. Such a new admission
and replacement campaign have not been issued in this release.

`analysis/close_failed_study.py` authenticates all 15 jobs and all accepted
bundles/replays, checks completed public regression lineage and protected v5
hashes, verifies SQLite/export agreement, reproduces the interrupted rejection,
and proves that the complete-study reporter refuses this incomplete campaign.
Its result is `RESULTS.json`. This analysis does not repair or authorize jobs.

## Public development and regression results

All accepted candidates were frozen before regression inference. Every initial,
selected and terminal snapshot was evaluated outside the training worker on the
same 1,240 exposed public regression examples: **37,200 predictions** in total.
Selected checkpoints use public development only; regression never selects a
checkpoint or causes a training restart. Step-zero selections and genuinely
trained terminal snapshots are both retained.

| Seed | Method | Terminal dev accuracy | Dev complete pairs | Regression accuracy | Regression complete pairs | Regression worst family | Label action errors | Selected update |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 47 | fixed-r8 | 92.80% | 85.94% | 90.56% | 83.06% | 0.00% | 24 | 900 |
| 47 | fixed-r12 | 92.45% | 85.42% | 91.45% | 83.23% | 50.00% | 26 | 900 |
| 47 | adalora-12-to-8 | 63.19% | 33.16% | 61.05% | 29.52% | 0.00% | 213 | 600 |
| 83 | fixed-r8 | 94.62% | 89.76% | 91.53% | 85.32% | 0.00% | 14 | 0 |
| 83 | fixed-r12 | 90.19% | 81.25% | 90.16% | 82.10% | 0.00% | 9 | 0 |
| 83 | adalora-12-to-8 | 61.11% | 31.60% | 56.69% | 26.61% | 0.00% | 112 | 0 |
| 109 | fixed-r8 | 95.31% | 90.97% | 91.69% | 85.32% | 0.00% | 27 | 1000 |
| 109 | fixed-r12 | 93.32% | 86.81% | 90.56% | 81.61% | 0.00% | 38 | 700 |
| 109 | adalora-12-to-8 | 61.46% | 30.38% | 57.82% | 25.81% | 0.00% | 135 | 600 |
| 157 | fixed-r8 | 94.53% | 89.06% | 92.82% | 85.97% | 41.67% | 35 | 1000 |

Action errors count CHEAP/FAST proposals when the required label is ASK/REFUSE;
this is not a complete feasibility or execution-safety oracle. These are
correlated operator-authored public diagnostics, with zero human reviews.
Small diagnostic families have small denominators; zero family accuracy is not
a population risk estimate. All accepted terminal candidates fail one or more
public behavioral screens. Do not average the uneven method/seed counts into
a claim of the registered treatment effect. No five-seed paired interval was
computed. Retention, mechanism ablations and protected qualification did not run.

## Rank mechanism, compute and reproduction

The three accepted AdaLoRA runs each have 70 allocation events, reducing 264
initial active components to 176 across 22 projections, with rank-12 tensor
storage retained. All preserve the frozen backbone and all 22 constant scaling
values. Genuine allocation is demonstrated; an AdaLoRA advantage is not.
Methods also differ in parameterization, regularization and initial capacity,
so an isolated difference would not establish a rank-allocation mechanism claim.

Accepted workers report 23,790,031 training tokens and 10099.49 seconds
of training-stage wall time in total. Their regression processes measured
1764.76 seconds of worker wall time. Per-run tokens, wall/CPU time, peak
memory, stored/active capacity and all predictions are in RESULTS.json and the
native bundles. Equal updates/exposures are not equal compute. Worker GPU
telemetry is self-report, not hostile-worker attestation. The local ledger is
not a financial billing cap or administrator-proof external attempt authority.

The canonical source branch is `research/saal-adalora-v6` at
`849760f49b1fc5d0cf8d941a3757c0e33fcf0a35`. Its original v1 worker and separately
versioned CUDA-safe v2 worker remain unchanged. New analysis lives only on the
separate results branch and is not retroactively included in earlier QA counts.
The package signature binds the exact source and final evidence commits.

## Verification and preserved history

- Clean pinned local QA preflight: 932 passed across eight canonical suites.
- Source-bound, hash-locked Colab CPU QA05: 931 passed, one macOS-only skip,
  zero failures/errors; exact runtime termination confirmed.
- Real T4 composite engineering pilot and all accepted study runs: all three
  snapshots passed fresh GPU and controller CPU replay at unchanged 1e-4.
- Six real CUDA artifact corruptions were rejected; three additional frozen
  regression admission-negative checks were rejected. These checks and the
  posthoc closure are separate from the 932-test count.
- Authentic v5 foundation: 2,615 logical files, 253 frozen files and 790 tests
  verified. Its five original jobs remain RESERVED; old numerical failures
  remain failures.
- 425 physical LFS payloads previously verified against their committed OIDs,
  totaling 9,825,809,684 bytes. The physical packer rejects outer LFS pointers.
- Upstream ModernBERT weights are separately reconstructed from the pinned
  official revision and registry file hashes; they are not bundled in the ZIP.

Final ZIP verification is recorded in a **detached verification record next to
the archive**. It must confirm final-byte hashing, complete member extraction,
manifest/signature/history verification, all eight canonical suites from that
clean tree, and unchanged bytes afterward. Freezing this report before packing
avoids modifying an archive after testing it. Prior QA alone does not establish
that final archive check. No ChatGPT Library save API is available here; no save
or storage-limit error is claimed.

## Remaining work

A new versioned numerical metric contract and separately authorized admission
are required before another campaign. The interrupted run cannot be relabeled
COMPLETED, the spent budget cannot be reset, and missing runs cannot be imputed.
A future complete comparison still needs the authorized parent-data controls,
separate retention/mechanism protocols and independently reviewed novel tasks.
Operational custody independence, hostile-code containment, independent
qualification and recursive self-improvement remain unestablished.
