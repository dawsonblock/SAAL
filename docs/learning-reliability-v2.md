# Learning Reliability v2 — additive repairs, public exploration

The v1 diverse-data campaign learned its new presentation well, but full tuning scored only 48.86% on legacy cases and still made critical proposals. This batch addresses that measured failure, preserves historical source/results and leaves all independent qualification and promotion gates intact.

| Gap | Implemented repair | Limit |
|---|---|---|
| Compact constraints poorly learned | Fresh short instructions with scope, missing limits, amendments, expiry, numeric composition and quotation contrasts | Shared author/oracle; human review pending |
| Previously seen formats lost |3,840 balanced training records:768 old-v3,1,536 prior-diverse and1,536 fresh compact; old evaluation stays evaluation | Uniform seeded sampling gives expected 20/40/40 shares; data/compute changes confound v1 comparison |
| Step-zero checkpoint hides adaptation |Separate learning audit over retained snapshots, alongside unchanged safety selection |Unretained intermediate weights cannot be reconstructed from scores |
| Critical action proposals |Trusted-state guard checks permission source, scope, fresh clock, mandatory information, feasibility and objective, then requires existing one-use capability |Synthetic two-option policy; context/clock must be supplied by a trusted host; no production sandbox guarantee |
| Confidence can be misleading |Public selective-coverage/error curves; deferral is not silently scored as ASK |Not independently calibrated confidence |
| Human review has not happened |Offline blind UI, save/restore, signed two-reviewer quorum, adjudication and bound semantic review |Signatures prove key attribution, not that independent humans acted; zero real reviews yet |
| Regressions can be hidden by averaging |Exact common-case and family retention comparison against the frozen v1 terminal model |Exposed, correlated public diagnostics, not final holdout inference |
| Linux compatibility can drift |Pinned-action Linux Python 3.13 CI with hash-locked Python dependencies and source test evidence |CI results must be read before claiming success; CPU CI is not a GPU training qualification |

Use `python -m saal_rsi.reliability_cli --help` with `PYTHONPATH=src`. The `corpus`, `audit`, `retention`, `review-packet`, `sign-review`, and `assess-review` commands are additive. Existing v5 jobs, historical checkpoints, reviewer export gate, worker source identity and promotion remain unchanged. Actual reviews must be supplied by enrolled reviewers; the agent does not create them. Private review keys must be owned, regular, unlinked 32-byte files with private permissions.

The guarded executor supports only `sim.purchase`. It constructs the action from trusted state, binds context into the existing action capability, checks constraints again at execution, and delegates token consumption to the existing Authority. It neither loads/promotes models nor provides an external adapter. Guarded outcomes must be reported separately from raw neural predictions. Blocking an error with deterministic policy is not evidence that the model learned safety.

The mixed corpus is for ModernBERT four-class classification. Its native legacy decision-output metadata is not a new generative reasoning supervision contract. Qwen adaptation, dynamic-rank optimization and recursive improvement are outside this corrective experiment. The immediate bottleneck is reliable task learning and retention.

The frozen campaign compares fresh official ModernBERT full tuning (seed 47) with fixed-rank 4 LoRA (seeds 47/83). Each receives 1,000 updates, batch 8, FP32, class-balanced loss, 100-step checkpoint intervals and 2,400 seconds. Data sampling is matched for seed 47; adapter/dropout implementations differ. Previous results used 600 updates and another corpus, so cross-release gains cannot be attributed solely to replay. Expected sampler schedules are controller reconstructions, not hostile-worker attestation. All initial/best/terminal tensors and failed screens stay visible. Neither regression nor retention measurements select checkpoints. Public retention screening allows at most 5 percentage points of loss per prior recipe and no increase in critical proposals; zero-critical and independent gates remain separate.

Native Colab provisioning, hash installation, progress streaming, time limits, durable development attempt accounting, artifact recovery and exact-session watchdog cleanup are retained. No private evaluator/custodian keys, protected tasks or holdout labels are staged. Controller-owned development ledgers remain vulnerable to a privileged administrator; local time budgets are not billing caps. Independent service administration, protected sampling and human reviews remain unmet operational/scientific work, even if public diagnostics improve.

This supplement is NOT_EVALUATED. External deployment is prohibited. It does not demonstrate AGI, guaranteed alignment or recursive self-improvement.
