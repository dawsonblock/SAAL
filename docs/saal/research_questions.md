# Open questions, honesty constraints, and prior research

## Unresolved questions, ordered by expected leverage

1. **Goodhart-resistant measurement**: When outcomes are delayed, multidimensional and contested, what independent observations can establish whether an assistant helped? No scalar "human welfare" reward solves this universally.
2. **Transfer under goal change**: Do improvements persist when a new user, new task family and conflicting constraints are simultaneously unseen? Clarification can outperform weight updates; null or negative results matter.
3. **Scalable independent oversight**: Can less capable/independent monitors detect deliberate subversion from a stronger actor? How do we handle adaptive monitor bypass, compromised tools or confidence laundering?
4. **Corrigibility without instrumental pressure**: How to make correction, replacement and shutdown robust under longer horizons and new capabilities, rather than just plausible under tested prompts?
5. **Fidelity of causal evidence**: Which outcome effects can be attributed to an AI action via randomization, natural experiments, or credible identification? Which must be left unknown?
6. **Value plurality and consent**: Who resolves conflicts across principals, third parties and shared norms? Policy/governance cannot be silently learned by the actor.
7. **Interpretability confidence**: Can activation probes generalize under adversarial optimization? How do we estimate uncertainty and adversarial sensitivity rather than confuse a successful probe with a proof?
8. **Longitudinal learning**: How do we avoid regression, forgetting, reward-tampering gradients and training-data poisoning across many releases?

## Literature starting points (review independently before quoting numerical claims)

- Hadfield-Menell et al., *Cooperative Inverse Reinforcement Learning* (2016): https://arxiv.org/abs/1606.03137
- Everitt et al., *Reward Tampering Problems and Solutions in Reinforcement Learning* (2019): https://arxiv.org/abs/1908.04734
- UK AISI Inspect AI evaluation framework: https://inspect.aisi.org.uk/
- The user's October 2026 research synthesis also discusses beneficial-trait RL, AssistanceZero, automated alignment research, AI control, reward-seeking experiments, alignment faking, sleep-agent experiments, CoVal and model specification training. Those papers informed this design **but this repository does not independently reverify every result or attach reproduced numerical claims**. Read and verify primary sources, versions, evaluation design, and exact generalization caveats before citing specific numbers externally.

## Epistemic contract

A benchmark improvement is evidence of performance on the defined benchmark. It does not prove a model "wants to help," has decoded the true objective, or will retain values under unlimited capability increases. Distinguish:

- **Behavioral alignment in task distribution**: measurable, narrow, empirical.
- **System-level control**: enforceable under explicitly stated threat and trust assumptions.
- **Internal alignment / motivation**: not established by outcomes or a handful of activations.
- **General aligned superintelligence**: unsolved; no claim is made here.

The correct disposition after a failed preregistered study is to reject, publish evidence, revise hypotheses, and generate fresh hidden tests. Do not re-label failed held-out examples as new independent evidence.
