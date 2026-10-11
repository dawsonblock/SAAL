# Reference study findings — October 8, 2026

The public synthetic study reruns the original v0.2 configuration on the new
v0.3 runtime: 5 procedural seeds × 120 matched episodes each per arm. The
trained intent model used 55 epochs and seed 1201. Summary and report hash are
recorded under `reports/reference-study-v0.3.0-public-demo.json`.

| Arm | Mean regret | Clarification rate |
| --- | ---: | ---: |
| A frozen no inference | 2.132354 | 0.0000 |
| B frozen + uncertainty | 0.002882 | 0.7567 |
| C offline model no ask | 0.745632 | 0.0000 |
| D offline model + ask | 0.015174 | 0.6650 |
| E B + simulated authority | 0.002882 | 0.7567 |
| F D + simulated authority | 0.015174 | 0.6650 |

**Result:** The offline-trained candidate F did **not** outperform frozen
uncertainty-assisted baseline E on average synthetic regret. This is a failure
to establish improvement, not a reason to lower the promotion threshold.

**Interpretation:** The learned text classifier makes fewer clarifying requests
than the rule-based baseline, but that reduction does not compensate for
mistakes in intent inference. The authority simulation prevents prohibited
synthetic tool requests; it does not improve understanding of user goals.

This is a public-seed, unsigned toy evaluation. No claim of sealed holdout,
causal human benefit, frontier-model corrigibility or production readiness is
justified. To test learning fairly, run a privately frozen study with multiple
training seeds, confidence bounds clustered by independent procedural seeds,
novel simulated families, and multiple adversarial conditions. The original
small classifier training here is not LLM beneficial-trait RL or DPO.
