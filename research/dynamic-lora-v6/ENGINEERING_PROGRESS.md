# SAAL-RSI Dynamic LoRA v6 — engineering pilot complete

Final status: the subsequent study stopped with 10 accepted runs, one
interrupted run and four unrun reservations. See the
[final research report](RELEASE_REPORT.md) for the numerical validation defect,
all recovered results and confirmed cleanup. The snapshot below is retained
as the earlier engineering progress record.

Snapshot: 2026-10-14T11:06:07.322202+00:00. This is a progress report; the final study and ZIP are pending.

The CUDA AdaLoRA initialization defect is fixed through a new signed v2
contract. The original v1 worker, its successful fixed controls and both failed
parent campaigns remain unchanged. AdaLoRA is experimental and has not
replaced fixed LoRA. Qualification is **NOT_EVALUATED**; external deployment
is prohibited.

The matched engineering pilot used seed 60606, excluded from scientific
analysis. Each method completed 1,000 optimizer updates and 16,000 training
exposures from the same base/head and sampling stream. All initial, selected
and terminal snapshots passed fresh GPU and CPU replay. Exact-session cleanup
is confirmed. The composite gate explicitly binds the two completed v1 fixed
controls to the separately admitted v2 AdaLoRA run; the old campaigns remain
FAILED.

| Terminal public diagnostic | Fixed r8 | Fixed r12 | AdaLoRA 12 to 8 |
| --- | ---: | ---: | ---: |
| Accuracy | 92.10% | 90.62% | 61.20% |
| Complete counterfactual pairs | 84.55% | 81.94% | 31.94% |
| Worst-family accuracy | 71.88% | 62.50% | 53.12% |
| Label-based action proposal errors | 33 | 27 | 100 |
| Selected checkpoint update | 700 | 700 | 0 |

These errors mean CHEAP or FAST was proposed when the required label was ASK
or REFUSE. They are not a complete feasibility or execution-safety oracle.
All three candidates fail the public behavioral screens. AdaLoRA is worse
than both fixed controls in this single excluded-seed pilot; this is not a
five-seed conclusion or a claim that all adaptive-rank methods are inferior.

AdaLoRA made 70 allocation events and ended with 176 active components across
22 projections. Its tensors still store rank 12 per projection. All 22 scaling
values remained fixed at 12. The full frozen-backbone integrity check passed.
The precommitted safety-first selector chose AdaLoRA's step-zero checkpoint;
the genuinely trained terminal model is retained and reported separately.

[Learning curves](analysis/composite-pilot-figures/pilot-learning-curves.png)
and [layer-rank allocation](analysis/composite-pilot-figures/pilot-adalora-ranks.png)
show the actual validated artifacts. Stars mark selected checkpoints.

Validation completed:

- Clean local preflight: 932 passed.
- Fresh hash-locked Colab CPU run: 931 passed, one macOS-only platform skip,
  zero failures/errors, all eight suites; runtime termination confirmed.
- Six semantic corruption checks against a real v2 CUDA bundle: all rejected.
- V5 foundation: 2,615 logical files, 253 frozen files and 790 tests verified.
- Git LFS: 425 actual payload files, 9,825,809,684 bytes fully hashed against
  committed object identities; no unresolved outer pointer payloads.
- The five original v5 jobs remain RESERVED. Historical numerical failures
  at the existing 1e-4 tolerance remain failures.

The five-seed study now has 15 separately signed jobs, protocol
`sha256:ed38b130f68fd0f186d8cedd5a8089f1a0819a3587951c7136a994f9d990382d`.
Its named T4 session is `saal-c9456df8eb3d4554bd7b`; observed phase at this snapshot
is `CAMPAIGN`. No scientific run is claimed complete by this report.
The controller enforces deadlines, recovery, per-run replay and exact-session
cleanup. The existing thread monitor checks every five minutes and continues
with only the four authorized parent-data controls after the main study passes.
No automatic training retry or budget reset is allowed.

Source: `research/saal-adalora-v6`, commit
`849760f49b1fc5d0cf8d941a3757c0e33fcf0a35`. Main is unchanged.
The final results branch and full physical ZIP will be completed after the
admitted runs and final clean-extraction verification; this snapshot is not
that delivery.

Public families remain correlated and operator-authored, with zero human
reviews. Independent generalization, sequential retention, an adaptive-rank
advantage and recursive self-improvement have not been established. Tests
and controller signatures cannot supply independent custody or qualification.

## First post-freeze public regression check

The additional bounded CPU collector has evaluated all initial, selected and
terminal snapshots of the completed seed-47 fixed-r8 study candidate. It made
3,720 predictions over the frozen 1,240-case public regression population,
finished in 196.34 seconds against a 600-second worker wall limit, and verified
the frozen backbone unchanged. Its native inference and three admission-negative
checks are additional evidence, separate from the earlier 932-test QA count.

The terminal checkpoint reached 90.56% overall accuracy and 83.06% complete-pair
correctness, with 24 label-based action proposal errors. Two small diagnostic
families scored zero: negated preference, 0/4; latest valid correction, 0/12.
These exposed cases are not independent population estimates. The result shows
why overall accuracy is insufficient for admission and does not qualify the
candidate. Checkpoint selection and all admitted training jobs remain unchanged.

The collector is separate from the T4 worker and uses no protected final data.
Its plan, predictions, metrics, work accounting and frozen candidate identity
are retained under `regression-study-v2`. Remaining candidates are evaluated
only after their training, artifact verification and replay complete.

## Seed-47 public regression block complete

All three seed-47 methods now have completed bounded CPU regression of their
initial, selected and terminal snapshots: 11,160 predictions in total. The
signed jobs, native artifacts, GPU/CPU replay bindings, regression result
hashes and unchanged frozen backbones were checked again. Both collector
invocations exited successfully. The evidence is recorded in
`validation/frozen-regression-seed47-verified.json`.

| Terminal public regression | Fixed r8 | Fixed r12 | AdaLoRA 12 to 8 |
| --- | ---: | ---: | ---: |
| Accuracy | 90.56% | 91.45% | 61.05% |
| Complete counterfactual pairs | 83.06% | 83.23% | 29.52% |
| Worst-family accuracy | 0.00% | 50.00% | 0.00% |
| Label-based action proposal errors | 24 | 26 | 213 |

These are descriptive results from one seed on exposed, correlated public
examples. All three terminal candidates fail the existing behavioral screens.
This is not the registered five-seed comparison, independent generalization
or sequential retention evidence. All admitted training configurations and
checkpoint choices remain unchanged; no candidate is promoted.
