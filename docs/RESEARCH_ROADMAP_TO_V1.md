# Research development roadmap: v0.9.1 to v1.0

Learn, qualify and authorize remain separate operations. The current release is v0.9.2, an evaluation-integrity release. Later version labels below are targets, not delivered features or claims of alignment.

| Target | Work | Required experiment/evidence | Status |
|---|---|---|---|
| v0.9.2 Evaluation integrity | Provenance, task-family partitions, duplicate/lineage review, paired family metrics, retention and custody-reviewed gates | Frozen artifacts, negative controls, reproducible rejected evidence, independently sampled evaluation when externally available | Engineering implemented; independent empirical qualification NOT achieved |
| v0.9.3 Robust neural learning | Add tiny-transformer-lora-v2 with positions, masks and strict tensor inventory; preserve tiny-attention-lora-v1 | Multiple preregistered seeds; majority/rules/incumbent/position-aware frozen/LoRA/full tuning/replay/dynamic-rank ablations; unfamiliar families | Planned |
| v0.10.0 Real LLM adaptation | LearningBackend interface and pinned TransformerPEFTBackend; exact model/tokenizer/template/config revisions, no unreviewed remote code | Small open-weight model, T4 admission probe, LoRA save/reload/frozen-weight checks; optional verified 4-bit QLoRA; memory/tokens/time/cleanup accounting | Planned; inherited training helpers are not model admission |
| v0.11.0 Independent assurance | Separate identities/machines/keys, authenticated IPC, restricted custody and artifact predecessor chain | Cross-job substitution, stale/replayed signatures, concurrent reservations, rollback and unauthorized activation attacks; independently repeated measurements | Planned; local signing identities alone are insufficient |
| v1.0.0 Governed improvement | Bounded configuration proposer and verified evidence memory; authority/evaluator changes excluded | Matched fixed-control/improvement branches, equal search/compute budget, multiple seeds, fresh future families and repeated cycles; account for rejected proposals | Planned |

## Data and experimental design

The six categories remain ordinary objectives, injection resistance, true-goal versus proxy conflict, safe refusal, necessary clarification and current-user correction. Include CHEAP and FAST, unfamiliar language, negation, material authorization changes, swapped speaker roles and counterfactual order. Augmentations and related templates stay within one split. The reusable development benchmark never qualifies a candidate. Regression suites track prior useful decisions. An evaluator/custodian separately obtains and protects final task families after protocol/artifact freeze. Statistical thresholds and sampling/sign-null assumptions must be registered before protected evaluation, with existing attempt/alpha accounting preserved.

For v0.9.3 continual learning, measure tasks before adaptation, after each task and after subsequent tasks. Publish the complete retention matrix, backward/forward transfer, maximum past-versus-final forgetting, adaptation sample efficiency, trainable parameter/adapter growth and unsuccessful trials. Compare replay and dynamic-rank variants against matched frozen algorithms. Neither larger rank nor lower training loss is evidence of transferable improvement.

For v0.10.0, LearningBackend must support baseline load, bounded adapter creation/training, task evaluation, checkpoint export/restore and predictions. Pin model weights/revision, tokenizer/revision, chat template, target modules, precision and quantization. Begin approximately 0.5B–1.5B, but admit configurations only after actual available-memory and throughput probes. A model that merely loads does not pass: it must train, restore, evaluate and clean up within budget. Do not assume a 3B model or QLoRA is equivalent to standard LoRA on T4.

## Authority and compute

The compute sequence stays: signed job validation, quota reservation, a fresh dedicated session, dependency/GPU admission, measured work, bounded checkpoint export, confirmed stop, then independent checkpoint replay. Final labels/keys and promotion authority never enter training workers. Admission quotas are not billing caps. Unconfirmed cleanup must remain an unresolved incident rather than a completed job.

The future chain is CandidateManifest → BuildManifest → EvaluationBundle → QualificationRecord → PromotionDecision → RuntimeManifest. Bind exact artifacts, protocol/dataset/worker/dependency identity, custody/evaluator identity, thresholds, predecessor signatures, scope and expiry. The sole promotion authority may authorize only explicitly approved simulation deployments. It must reject stale, substituted, consumed, expired or incorrectly signed predecessors. Neural qualification alone grants no execution capability.

## Governed improvement experiment

Initially permit only approved bounded rank, learning rate, replay ratio, curriculum, routing, checkpoint selection and architecture variants. Executable code changes require source review and sandbox qualification. Signing services, authority policies, promotion gates and evaluator changes remain outside learner authority. Record all proposals, budgets, negative results and known failures in verified experimental memory without exposing final labels or allowing rewriting independent evidence.

A fixed algorithm control and a governed improvement branch receive the same starting model, tasks and resources. Repeated cycles must improve performance or sample efficiency on subsequent unfamiliar families, across preregistered seeds and accounting for unsuccessful search. One discovered configuration or one better task score is ordinary adaptation, not recursive improvement.

## Acceptance

Integrity (tests and invalid authority-transition rejection), generalization (new independent families and justified inference), safety/utility (specified errors and conservative bounds without needless refusal), and reproducibility (checkpoint and evidence reconstruction) remain mandatory. Recursive improvement requires a fifth gate only for v1.0: repeated gains against fixed controls within declared budgets. A failed gate preserves the rejection and blocks the claimed qualification. Development can continue with fresh protected evaluations and intact attempt accounting.
