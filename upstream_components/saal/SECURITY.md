# Security policy and status

SAAL v0.5.0 is a **research-only reference**. It must not be deployed to run financial transfers, sensitive medical workflows, permanent file deletion, equipment/robotics or other high-impact tools. The in-memory demonstration is not a security boundary. No live external connectors are included.

Review `docs/threat_model.md` and `docs/deployment.md` before integrating even low-impact tools. In a production installation, a separate privileged executor and authority must be the sole owners of credentials, and independent evaluator/admission signing keys must stay outside trainer/actor access. The example JSON policy is explanatory; `authority.py` is the normative current policy implementation.

Secrets (`*.key`, `.env`) and local evaluation artifacts are excluded from distribution. Never send them in issue reports. Treat all agent-generated content as untrusted. Scope every task; deny unknown actions. Test every critical boundary with privilege-escalation attempts. A signed evidence manifest does not authenticate facts unless its signer independently verified those facts.

There is no claim that the monitor, toy scenario, HMAC gate or LoRA adapter proves or guarantees alignment. Submit security reports privately to the software owner rather than publicly disclosing sensitive keys or user data.


## v0.2 remaining threats

A single-UID installation does not provide meaningful authority isolation. Cumulative spending controls declared costs only; a future real executor must independently confirm final amount before any external effect. The evaluator HMAC key is shared with the qualifier; use public-key/remote-attestation and documented key custody for production. Signature verification cannot establish outcome truth without trusted evidence collection. Staging makes no runtime model changes and assumes a trusted staging filesystem.

## v0.3 threats and residual risks

- The authority daemon's Unix peer UID plus administrator-installed principal/task
  mapping stops caller-supplied identity spoofing across UIDs. It **does not**
  distinguish mutually untrusted actors sharing one UID or authenticate an
  actual human user. OS process isolation, dedicated identities, and a separate
  identity provider remain required.
- Socket permissions are `0600` by default; operator can explicitly enable
  `0660` for a dedicated group using `--socket-group-gid`. Never allow the
  actor identity to read signing keys, authorization DB, policy files, or
  evaluator secrets. No unrestricted ``chmod 777`` workaround is acceptable.
- A v3 single-use ticket is issued by an isolated policy service. `max_actions`
  caps zero-cost requests and persists consumed counts. This does **not**
  prevent resource exhaustion from unauthorized connections at the OS layer,
  nor substitute for per-process CPU/memory/network budgets.
- `require_replay=true` detects fabricated synthetic study records given a
  trusted, unchanged oracle, simulator and exact candidate file. The evaluator
  HMAC is still shared with the qualifier, so trust is not cryptographically
  separated. The replay cannot validate measurements of human welfare.
- Public seed/config/example files in this ZIP are NOT sealed benchmarks.
  Generate real seeds privately, preregister thresholds before training, and
  externally timestamp/freeze the commitment. Do not train on a heldout leak.
- This prototype does not supply process sandboxing, network restrictions,
  SELinux profiles, live external tool execution, an authenticated human-approval
  UI, secure key rotation, signed evaluator code provenance, or a reliable
  bound on all adversarial failure rates.

## v0.4 evaluator-signature and source-integrity boundary

v0.4 closes **the evaluator/qualifier shared-HMAC-secret design for new sealed synthetic studies**: the evaluator signs with an Ed25519 private key, and the qualifier needs only its public key. The public key must be externally pinned in the preregistered study; the preregistration digest must be independently published before evaluation. Without external witnessing of *when* the preregistration was fixed, the command-line hash check cannot establish preregistration chronology. The code digest validates exact source agreement between separately controlled evaluator and qualifier installations, not the provenance of any binary dependencies or the integrity of execution hardware.

Legacy HMAC evaluation remains for running old v3 experiments, **not for v4 qualification**. `signing_status` and `signature_scheme` must match the v4 protocol. HMAC-only evaluators must not be used in security-sensitive promotion decisions.

Socket connections have a bounded read timeout, but the single-process Unix socket demo lacks all production-grade denial-of-service and sandbox controls. It does not provide authenticated human approvals, remote identities, cross-machine key management, evaluator key revocation, or trustworthy observation of real user welfare. The evaluator and qualifier must not share private credentials or an actor-writable filesystem. There is still no real external effects tool or automated deployment command.

## v0.5 independent witness and data admission

New v5 signed synthetic study reports require a separate Ed25519 witness receipt
for the preregistration and independent witness key pinning. The evaluator signs
and binds the witness digest, and the qualifier verifies and replays the report.
A signed witness timestamp is **not** independently verified chronological evidence;
store the receipt in an external independently controlled append-only log before
evaluation. Sharing a UID or filesystem among actor, evaluator and witness breaks
the independence assumption. v4 verification persists for compatibility only.

Admitted JSONL is parsed with duplicate-key rejection, exact field whitelist,
strict per-record size and partition checks, and no-follow one-snapshot hashing.
Dataset admission signatures authenticate an authorized reviewer, not the data's
truth, privacy appropriateness, provenance or absence of poisoning. Do not use
private human data without express lawful authority, consent and retention rules.

See `docs/safety_case_v5.md` for bounded claims and open attack paths.
