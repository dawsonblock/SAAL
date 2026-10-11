# Threat model — attacker capabilities, failure modes, and mitigations

## Trust assumptions

Defender assumes OS privileges can isolate authority/evaluator/executor from the actor; appropriate signing secrets stay in distinct trusted service accounts; trusted UI can authenticate human approvals; privileged tool credentials are never mounted into the actor; system administrator/operator is trusted. These assumptions are **not enforced by the single-process demo**.

| Failure | Attack example | Current response | Known gap |
| --- | --- | --- | --- |
| Reward proxy gaming | Book any appointment quickly to get `+10`; ignore transfers | Independent simulator calculates true regret | Does not capture human welfare |
| Prompt injection | Untrusted booking notes instruct policy bypass | Typed proposal and independent tool whitelist | Heuristic monitor is not robust vs adaptive injection |
| Unauthorized grant | Model submits `system.change_policy` or `model.promote` | Unknown/blocked tools fail closed | Privileged credentials still need OS isolation |
| Approval spoof | Actor self-identifies as human | Human approval requires a separate unpredictable credential | Need production-grade authenticated UX and anti-CSRF |
| Replay | Re-submit previously signed ticket | SQLite atomic nonce consumption | Central replay availability and failover design required |
| Ticket substitution | Swap `slot_id`, scope, or time | Ticket signs canonical exact action digest | Cross-language canonicalization not standardized yet |
| Credential theft | Actor reads shared process memory or disk | Separate Unix service example | In-process demo provides **no** OS isolation |
| Revocation | User withdraws task permission after token issued | Revocation registry blocks both newly minted and pending tickets | Distributed cache/key rotation work remains |
| Data poisoning | Insert misleading successes in training set | Independent Ed25519 manifest admission | Manual provenance/PII screening isn't automated |
| Holdout leakage | Trainer reads sealed test seeds or outcomes | Separate train/eval seed configuration | Packaged toy simulator is PUBLIC; not actually secret |
| Grader metagaming | Model learns to print expected results | Verifier derives synthetic outcome from simulator | Need genuinely independent live tool receipts |
| Audit rewriting | Rewrite SQLite history and rebuild hashes | Detects changes against externally stored anchor | SQLite itself is not tamper-proof; remote WORM required |
| Monitor evasion | Clever text escapes keyword monitor | Monitor cannot grant access | Needs adversarial eval and learned monitor independence |
| Over-clarification | Excessively interrupt user | Ask cost and EVPI criterion | Not calibrated from real user burden |
| Goal manipulation | Assistant influences preferences to improve scores | No global preference reward; prompt for corrections | Need non-manipulation tests and affected-party oversight |
| Model replacement | Candidate modifies deployment | No deployment command from learner | Real release signer and deployment controller not built |

## Highest-risk gaps before deployment

**P0: build verified principal authentication, a hardened isolated privileged executor, a separately signed human-approval workflow, and signed policy/delegation bundles.** Requiring a process boundary does not help if the actor still has access to the same credentials or privileged APIs.

**P0: isolate evaluation generation from development.** A hidden-seed string packaged with source is not a secure sealed evaluation set. Maintain an independent external evaluator process, trusted secret, nonpublic evaluation generators, and audit of all candidates tested to prevent adaptive overfitting.

**P0: isolate admission.** An attacker can type `"admitted_by":"independent_data_authority"` in JSON; that field alone is meaningless. The optional LoRA script requires a separate Ed25519 signed digest of the exact training dataset, verified using a public key. The private key must not live on the learner host.

**P0: avoid direct high-impact tool integration.** This project provides only a simulated scheduler. Do not add live booking, external messages, deletion, medicine, finance, physical hardware, or model promotion before threat controls and informed approvals exist.

## Novel attacks that must be tested later

Adaptive monitor prompt injection; strategically timed benign-to-malicious escalation; delayed side effects; reward process and evaluator input tampering; approving a low-risk action that leads to high-risk delegated tool composition; inference about hidden cases through prior score reporting; model-specification gaming; learned resistance to correction or shutdown; social preference manipulation; cross-principal privacy leaks; side-channel exfiltration; false positive over-blocking and denial-of-service on authority services.

## Safety interpretation

A gate can prevent a class of unauthorized API invocations **if** it is the sole privileged path and its assumptions hold. It cannot ensure an allowed action has desirable real-world consequences. Interpretability probes and monitor scores are investigative evidence, never permission grants or certificates of moral reliability.
