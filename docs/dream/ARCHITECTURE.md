# Governed Dream-RSI architecture

This reference build intentionally separates five roles:

1. **Online discovery** produces immutable observations and repeated stochastic outcomes.
2. **Policy proposal** may invent or tune exploration strategies, but has no activation authority.
3. **Replay evaluation** scores candidates on development and selection data without exposing sealed holdouts.
4. **Qualification** compares a candidate with the currently approved baseline on sealed holdout worlds and rejects unsupported/high-uncertainty candidates.
5. **Fresh live canary** supplies out-of-replay evidence that the candidate does not regress on a newly executed task/environment.
6. **Promotion authority** signs only passing qualification + canary records; runtime activation should accept only a valid signed manifest whose artifact and qualification digests match.

The critical invariant is:

`proposal != qualification != promotion != execution`

## Replay semantics

The original research framing treats history as a replay simulator. This build makes its limitations explicit:

- A replay world can contain multiple outcomes for one action, approximating a stochastic transition distribution rather than a single frozen child.
- Replay never fabricates unobserved children. Unsupported actions are counted and reduce support metrics.
- The evaluator carries uncertainty and support terms into the objective.
- Policies receive only prefix observations through `PolicyContext`; future outcomes are never passed to policy code.

## Dataset isolation

World IDs are deterministically assigned to development, selection, and holdout splits using a salted SHA-256 partition. In a real deployment the holdout index and contents should live behind a separate service/account so the proposal agent cannot read them.

## Authority closure

A candidate cannot promote itself. `Qualifier` produces an evidence digest; `PromotionAuthority` signs a `PromotionManifest` only when all gates pass. The manifest binds:

- candidate identity and source digest;
- qualification evidence digest;
- artifact-root digest;
- promotion generation;
- signer key identity.

## What remains external

This package does not pretend to be the unreleased Google/DeepMind implementation. It supplies the missing control-plane skeleton and demonstrable invariants, but production deployments still need a real coding-agent adapter, evaluator adapter, OS/container sandbox, secret isolation, remote signer/HSM, fresh-task authority, and replicated live canary experiments.
