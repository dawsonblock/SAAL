# SAAL reference design and scientific commitments

## Thesis

An acting model may predict likely intentions and propose reversible assistance, but must not unilaterally (a) define legitimate objectives, (b) acquire permissions, (c) judge its own outcomes, (d) train on its own self-scored trajectories, or (e) promote its own successor. Separation of concerns is a **control strategy**, not proof that the model has internalized normative values.

```
[Human with legitimate scope] ────> [Authenticated Delegation Registry]
            │                                │
            ▼                                ▼
[Explicit correction/consent]           [Authority service / key custody]
            │                                │  DENY/REQUIRE APPROVAL/issue ticket
            ▼                                ▼
[Belief & memory] -> [Untrusted proposal] [Trusted executor] -> [External tools]
                            │                    │
                            │                    ▼
                      [Low-trust monitor]   [Outcome verifier]
                                                │
                                                ▼
                                       [Audited evidence chain]
                                                │
                              [Independent train-data admission]
                                                │
                                           [Offline learner]
                                                │
                                    [Independent sealed evaluation]
                                                │
                                         [Promotion authority]
```

**Dependency direction**: Learner may read approved feedback but cannot write policy, verifier, evaluation set, keys or promotion state. Verification evidence is multi-source and uncertainty-preserving; no direct in-process `actor.promote()` or `actor.change_policy()` API exists.

## Formal task model

A delegation is `D=(principal, task_id, expressed_goal, tools, expiry, spending_limit, external_effect_scope)`. An action proposal is `a=(tool, args, principal, task_id)`. The server computes action fingerprint `H(serialize(a))`, never trusts actor-assigned impact labels, and applies hard constraints first.

Let `theta` be the latent user preference, `h_t` the current interaction evidence, `b_t(theta)=P(theta|h_t)` the belief. The reference policy predicts utility under the belief and optionally asks when:

```
EVPI = sum_theta b(theta) * max_a U(a,theta)
       - max_a sum_theta b(theta) * U(a,theta)
ASK if EVPI > ask_cost and uncertainty is material
```

This **perfect information upper bound** is deliberately simplified; a genuine query has noisy answers and should use query-specific *expected value of sample information*, include user interruption cost and calibration. The included four-goal synthetic simulator uses known deterministic objective functions, enabling independent oracle scoring. It is not a general Bayesian model of human welfare.

For impactful real actions, the authorized feasible set is `F={a : explicit_scope(a) AND allowed_schema(a) AND policy(a) AND required_approvals(a)}`. No amount of inferred helpfulness expands `F`. The actor's model parameters are frozen while servicing any live delegation.

## Control-plane threat boundaries

**Role 1 — principal and delegation issuer**: identity-authenticated session creates a narrow, expiring delegation; user can correct preferences but cannot override other parties' legitimate rights.

**Role 2 — acting policy**: processes untrusted web/task text and returns structured proposals. It has no signing key, no privileged filesystem and no direct external service credentials.

**Role 3 — authorization**: distinct process/host, installed (not client-submitted) delegated scopes, policy digest, schema validation, human approval credential out of actor control, token expiry and nonce replay ledger.

**Role 4 — trusted executor**: owns privileged tool credentials; checks ticket against the *exact* action, scope, policy hash, expiry and unused nonce. Our executor is simulation-only. A real connector must be implemented with all of these checks at its final privileged boundary.

**Role 5 — outcome verifier**: reads external receipts/sensors and records provenance, uncertainty, delayed correction; never accepts a model's statement that the job is done as independent evidence.

**Role 6 — evaluation/data/promotion**: frozen test manifest, held-out scenarios inaccessible to trainer, independent attested results, separate data-review signing key, final model-release approval by a trusted principal. A passing recommendation is not executable deployment permission.

## Legitimate authority under contested values

The reference handles *one synthetic principal*. A real multi-party assistant needs externally governed adjudication rules; no global scalar welfare optimum is defined. Escalate conflicts. Human preferences and permissions are not interchangeable. Changes of mind must be acknowledged promptly; authenticated explicit consent may change task scope but not override statutory/third-party constraints.

## Learning hierarchy

1. Fast: explicit corrections, belief changes, scoped and consented memory; memory must be erasable and subject to TTL.
2. Medium: detect prediction mismatch, delayed failures and conflicting evidence. Review dataset eligibility independently.
3. Slow: create a candidate offline (SFT/LoRA/other empirical method), compare to frozen baseline, run reward-conflict/goal-change attacks, and reject if tail risks worsen.
4. Release: human authority assesses evidence and selects a constrained exposure/canary. Rollback triggered by an external controller, not actor wishes.

## What is intentionally omitted

No unconstrained action loop; no live medical/financial/physical tool access; no autonomous production credential issuance; no world-welfare reward; no causal human-welfare inference; no transparent access to hidden held-out cases in a trained model service; no automatic runtime weight replacement; no claim to mechanistic proof of values.
