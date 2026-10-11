# v0.3 independent synthetic study qualification

## What independent replay checks

The evaluator signs a deterministic study report. On a trusted qualification
host, `qualify(..., model=exact_candidate)` now recomputes **every arm and every
episode** using the exact candidate model and preregistered procedural seeds.
The full canonical report digest must match. It catches changes to synthetic
utility, regret, injection counts, aggregate scores, or arm/episode membership,
even if a report was re-signed and internally self-consistent. The model digest
and preregistration digest are also bound.

The qualifier **must compute replay itself**. It never trusts an unverified
caller-supplied `replay_ok` flag or imported replay JSON.

## Sealed experimental procedure

1. An independent research principal controls the evaluator and generates
   `saal prereg-create --out /secure/private-prereg.json` in a sealed directory.
2. Record the SHA-256 hash of that full file in an external timestamped audit
   system **before candidate search begins**. The local hash alone does not
   establish freeze time. Freeze thresholds, simulator revision and study design.
3. Keep private seeds, evaluator signing key, reward/oracle code and tests out
   of the learner/trainer execution environment.
4. The trainer submits an immutable candidate file. The evaluator computes
   `saal evaluate --model ... --prereg ... --eval-key-file ...` from scratch.
5. On a qualification host with vetted model and evaluator artifacts run
   `saal qualify --model ... --report ... --prereg ... --eval-key-file ...`.
   `require_replay=true` is mandatory for this additional gate.
6. A separate qualifier Ed25519 signature and separate human staging approval
   remain required to stage a candidate. No automatic deployment exists.
7. A candidate that fails on a sealed evaluation is rejected. Do not tune to
   that same holdout and continue calling it previously unseen.

## Scientific limits

Exact replay checks **integrity/reproducibility**, not whether the simulator
accurately represents welfare, user agency, downstream harms, or real-world
causal effects. Nor does replay defeat a compromised evaluator AND compromised
trusted reference implementation. HMAC signing key is currently shared by
qualified checking code; production should use detached Ed25519 evaluator
signatures and verified immutable evaluator environment measurements.

The number of *independent clusters* is the effective uncertainty unit. Twelve
seed clusters and 100 tasks/seed are an illustrative starting point, not a
statistically sufficient sample for rare catastrophic events; a one-sided 95%
zero-failure upper bound is still approximately 3/n. Independent adversarial
evaluation is required on top of these toy checks.
