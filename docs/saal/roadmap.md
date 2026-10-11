# Research engineering execution: 0–30–60–90 days

## Phase 0 — here / reference version

Shipped in `v0.1.0`: toy environment; uncertainty and clarification policy; offline softmax training; deterministic hidden synthetic preferences; measured true-vs-proxy reward conflict; six fixed study arms; authorization data model; single-use tickets; separate Unix-socket authorization example; SQLite hash chain; attested train records; signed eval reports; qualification gate; automated tests. No real-world systems can be operated through this code as supplied.

Exit evidence: tests green and the study records its own failure cases without automatically promoting the trained arm.

## Days 1–30: security foundation and experiment validity

**Week 1:** freeze threat model and action contracts, implement signed delegation issuer, principals and authorization revocation. Split actor/authority/executor/evaluator under different OS users. Replace in-process demo trust assumptions with integration tests that attempt illicit credential access.

**Week 2:** add immutable external audit head anchoring, signed outcome receipts, error-state instrumentation, privacy redaction and training data provenance. Make all tool results typed with no arbitrarily executable command field.

**Week 3:** create 3 more independently developed synthetic task families: trip planning with changing mobility constraints, document editing with privacy side effects, and software debugging with test-oracle reward exploits. Keep task oracles sealed on separate evaluator host. Prioritize realistic clarification cost and delayed outcomes.

**Week 4:** run 5 training seeds × 5 environment seeds on a preregistered factorial; evaluate control bypass, false confidence and grader incentives. Declare a pass/fail result with confidence intervals; publish failures and implementation limitations.

Exit criteria: independent evaluator cannot be probed from actor; all unauthorized operations are blocked under realistic test fixtures; at least 3 task families with documented valid outcomes.

## Days 31–60: genuine behavioral learning tests

Build a typed adapter for a **small open-weight language model**. Freeze base weights in deployment. Use independently reviewed safe trajectories for SFT and LoRA, compare with memory-only and uncertainty-only policies, evaluate preference updates and calibration. Expand beneficial-behavior data beyond scalar rewards to distinguish outcome accuracy, truthfulness, user agency, uncertainty and corrective compliance. Include adversarial user-model manipulation tests and strict train/validation/holdout partitioning.

Add review of errors from an *unexposed development set*; promote no weights unless a separate private evaluation is passed. Require that any learned policy outperform simpler non-trained interventions in a meaningful net-value metric rather than just increasing perceived alignment.

Exit criteria: independently reproducible OOD benefit or negative finding; no statistically supportable rise in rare-failure estimate; no new permission bypasses; release candidate can be rolled back by trusted operator.

## Days 61–90: controlled human pilot only after approvals

If previous gates pass, begin read-only/draft-only, consent-based pilot with real users and approved data handling. Observe whether clarifying questions meaningfully improve user-corrected outcomes and burden. Use randomized ethically approved A/B comparison where possible, with explicit limits on causal interpretation. Keep irreversible actions disabled. Add separate delayed-outcome and third-party impact review. Create an independent safety case and red-team sign-off; have a human authorize any expansion.

Exit criteria: evidence supports a limited, low-impact deployment envelope; privacy and incident response are ready; external reviewers sign off on permissions and harm bounds. If evidence is negative or inconclusive, stop progression and retain frozen baseline.

## Suggested resource budget (estimates, not quotes)

Reference simulator: normal laptop/CPU, no GPU, minutes for thousands of cases. Small LLM LoRA experiments: one appropriate GPU or a rented equivalent depending on checkpoint size and quantization, weeks of engineering/data and evaluation work. Meaningful independent human-welfare measurement and assurance require much more effort than just inference compute; prioritize an independent evaluator/security engineer and high-quality task design over early multi-GPU spending.

## P0/P1 cut list

P0: binding authorization + replay + revocation; realistic independent outcome evidence; sealed evaluations; preregistration integrity; dataset admission; clear rollback. P1: learned monitor, interpreter probes, multi-principal social welfare negotiations, large RL, complex multi-agent orchestration. Do not implement unrestricted online weight updating in this research phase.


## v0.2.0 remaining engineering roadmap

1. **Isolation proof:** split actor/gate/executor/evaluator into separately deployed principals; test authorization at the OS and network boundaries with independent red-team harnesses.
2. **Evidence:** trusted receipts, out-of-band append-only/WORM anchor, evidence revocation and privacy-preserving deletion, and signer rotation.
3. **Statistical validity:** externally frozen and timestamped preregistration; 20+ independent seed clusters, task-family stratification, multiplicity control, sequential testing rules and truly private holdouts.
4. **Real model integration:** externally admitted dataset, immutable LoRA train artifacts with full runtime closure, adversarial beneficial-trait objective tests and blinded human evaluations.
5. **Controlled promotion:** independent deployment controller, dual-signature model-release authorization, version pinning, controlled canaries and deterministic rollback; **not built** in v0.2.

Exit gate for anything outside synthetic tests: an audited executor, actual least-privilege credentials, out-of-band independent eval and actual human approval. None is implied by passing the unit tests.


## After v0.5 — highest-priority work

1. Deploy evaluator, witness, authority, signer and actor under separate UIDs/hosts with a trusted executor for any side effects.
2. Anchor preregistration witness receipts to a genuinely independent time-ordered registry; independently pin witness keys.
3. Measure binary dependency/runtime image integrity, not only Python source hashes.
4. Add audited real low-risk task outcome receipts and consent-aware memory.
5. Before LLM weight adaptation, run independent blinded comparisons and test reward–outcome conflicts under hidden-goal domain shift.
