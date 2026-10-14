# SAAL-RSI Dynamic LoRA v6 — engineering pilot complete

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
