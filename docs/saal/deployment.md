# Hardening path — no real tool integration in this release

The reference is not ready for untrusted high-impact deployment. The threat model assumes attackers may strategically manipulate input, instructions, feedback and learner code. Minimal production separation requires:

1. **Different OS principals or hosts** for untrusted model runner, authorization, execution, verification, data admission, evaluation and promotion. Do not put their signing/admin keys on the actor filesystem, in its environment, or its container.
2. **Authenticated principal/delegation creation**. The Unix server reads trusted pre-installed scopes but does not supply SSO/authenticated end-user request identity. Add signed delegation records with issuer, permitted resources, expiry, purpose, policy epoch and revocation authority.
3. **Privileged executor** as *sole* holder of calendar/financial/comms credentials; validates exact action fingerprint, signed policy epoch and replay registry; uses idempotency keys and rate limits; never accepts arbitrary shell commands or URLs from the model. Record receipt from the actual downstream service; retries must be carefully designed.
4. **Approval control** with session-authenticated user input, exact action preview and short-lived single-action authorizations. Keep approval key in a trusted subsystem. Consider hardware-backed keys and dual authorization for exceptional actions.
5. **Outcome verifier** reading trusted tool receipts and independently reporting actual effects, including delayed consequences. Keep provenance and multiple perspectives; default to UNKNOWN when not identifiable.
6. **Off-host audited receipts**. SQLite hash chaining is not immutable; periodically anchor a chain head to an independently controlled WORM bucket or external signed transparency log and verify after reboots/recovery. Design retention, redaction and legal deletion.
7. **Sealed evaluation** in an environment the learner cannot access or probe; secret key and task generator isolated, evaluation budget and disclosures logged, holdout rotated after leaks. Preserve training run artifacts for all attempts.
8. **Separate promotion release authority** with risk-specific canary rollout, emergency stop, permission downgrades, safety telemetry, reproducible signed model manifests and complete rollback.
9. **Assurance and observability**: adversarial prompts, policy bypass, credential extraction, replay, context poisoning, monitor bypass, denial-of-service, supply-chain scanning, secrets scanning, private data leakage checks. Do not claim security based on a single model scoring highly.
10. **Operational agreements**: incident response and external party impact review; consent, privacy, audit transparency, non-manipulation design and clear escalation rights.

## Trust-by-design tests required before P0 sign-off

- Untrusted actor has no route to actual external-service credentials; test from inside actor sandbox.
- Revoked delegation immediately invalidates prior tickets; test concurrent servers/restarts and partial failure.
- Unauthorized side effects fail independently even when model monitor is misled.
- Evaluation private key and hidden tasks are physically/infrastructurally inaccessible to training agents.
- Human approval is authenticated and exact-action bound; every requested permission increase must be tied to a legitimate principal.
- Tool schema canonicalization and signature validation are consistent across language stacks and versions. Current JSON subset is **not RFC 8785**.
- Key rotation, disaster recovery, crash consistency, compromise recovery, timestamp accuracy, denial-of-service and race conditions tested under load.
- Agents cannot compose apparently benign tools into policy-forbidden effects.

## Security posture

Current Unix-socket server validates pre-installed delegations and returns grants but does not itself perform credential-protected real execution. Client reads/writes are bounded. Socket filesystem permissions alone are not user authentication. A hostile actor with the same OS credentials as authority could inspect its memory/file or bypass this service; use distinct OS identities and filesystem ACLs.

A separate simulated executor demonstrates token consumption, not deployment-level process isolation. Production requires a narrow signed-authority protocol and hardened containers/VMs as appropriate, independent human approval, external key custody and crash-safe replay storage. 'Independent' in this repository describes *logical* separation plus an example authority process, not fully independent infrastructure for all roles.


## v0.2 trust and rollout status

No deployment functionality is provided. `candidate-stage` only verifies qualifier and human signatures and writes an immutable staged JSON candidate. It never sets a production pointer or executes model weights. For real integrations, authority, executor, evaluator, qualifier and human credentials must live in different OS accounts or hosts. All effectful tool costs must be independently verified by the trusted executor; the research sample has no such external executor. Do not run either demo service as root.

The gate now refuses to unlink **any** existing Unix socket. Inspect and remove stale sockets manually under operator authority. The v2 replay database refuses automatic migration of v1 token records. See `docs/upgrade_v2.md`.
