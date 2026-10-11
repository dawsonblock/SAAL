# Validation protocol

A defensible experiment should report:

1. repeated online runs with independent seeds;
2. replay-to-live rank correlation for candidate policies;
3. confidence intervals or bootstrap intervals;
4. total token, evaluator, replay, wall-clock and monetary cost;
5. ablations against deterministic schedulers (UCB, successive halving/Hyperband-style allocation, random, fixed breadth/depth);
6. sealed task holdouts never exposed to the proposal agent;
7. support coverage and uncertainty on every promoted policy;
8. signed immutable artifacts and reproducible evaluator versions;
9. canary rollback criteria before full deployment.

A single historical trajectory is not sufficient evidence for policy promotion.
