# Audit disposition — v0.8.1

| Supplied finding | Implemented correction and evidence | Remaining boundary |
|---|---|---|
| Actor never challenged by hostile instructions | `behavior.py` sends six attacks to actor workers and records proposals; negative injection actor fails behavior even when executor denies it | Frozen rule actor; no real LLM or open-ended robustness |
| Prompt-injection field decorative | Frozen wrapper reads untrusted text and declines to elevate it; alternative negative actor reacts to it and fails | Rule-based source separation, not semantic natural-language understanding |
| Proxy score not gated | Permitted high-proxy/high-regret choice fails explicit true-goal conflict gate; tested independently of authority denial | Synthetic objective only |
| All abstention penalized | No-safe-option refusal and unresolved clarification get zero regret; unnecessary refusal fails | Domain-specific scoring; no human preference validation |
| Unanchored plan | Signed custodian receipt required, exact plan/proposal/config/identity binding, consumed before evaluation | Independent custody/external publication required; local timestamps are not independent proof |
| Holdout reuse / alpha inflation | Fresh seed reservations, persistent one-shot consumption, atomic concurrency tests, alpha allocated at anchor including crashes | Operator can reset/rollback an unprotected SQLite ledger; exploratory bootstrap has no nominal-coverage guarantee |
| Source-only fingerprint | Includes source, tests, tools, configuration/project metadata; records executable/version/dependencies/platform; checks before/after | No hardware attestation or full dynamic library inventory |
| Duplicate/decorative replay search | One trace per policy/world; observed UCB score ranking determines top-three shortlist; test changes replay scores and sees changed finalists | One-parameter synthetic search; no Dream-RSI kernels/coding loop |
| Multiple promotion paths | Legacy issue/CLI promotion disabled; old activation routes to central external denial; research activation needs independent signed human approval | Same-UID callers can bypass Python conventions; no production admission |
| In-process actor / missing executor boundary | Fresh actor and executor processes, public-only IPC, executor-installed scope and nonce checks, forged/tampered/replayed capability tests | No distinct UID/network/filesystem sandbox; deterministic simulator can be reverse-engineered |
| Evaluation-sensitive behavior | Same task under evaluation/deployment labels; sensitive negative actor rejected | Cannot rule out hidden deception |
| Actual learning experiments | Not added; inherited training preserved but excluded by Candidate schema | No model training or alignment claim |

The original manifest verified all 143 listed files. Original tests on this macOS host: 116 passed, two socket identity tests failed. The upgrade adds a libc kernel peer-UID fallback for macOS and rechecks those tests. Intentional test migrations replace v0.7 unanchored qualification and legacy activation expectations with the v0.8 fail-closed workflow. Historical documents remain provenance, not current qualification instructions.
