# v0.9.4.1 corrective engineering and research scope

This release continues the actual v0.9.4 pretrained implementation. It does not
restart from v0.9.2. The original v0.9.4 ZIP is preserved exactly in `provenance/`,
alongside the previously retained v0.9.3 history. Every incoming file is frozen
in `v0.9.4.1-historical-input.json`; only release metadata may change. Historical
workers, checkpoints, dataset manifests, negative reports, receipts and source
archives are unchanged. The old source capsule applies to the old ZIP; the new
`corrective-evidence` capsule applies to this release. Neither grants authority.

## Attempt accounting

`attempt_authority.py` implements a custodian mirror and a separately managed
witness. The witness owns one fixed enrolled research question and global
attempt history. It assigns attempt n and alpha = total_alpha / 2**n, appends a
signed event with the previous envelope hash, and commits atomically. Attempts
are spent on reservation, including abandoned and failed attempts. An evaluator
must consume the exact reservation before inference. Candidate, baseline,
protocol, protected-data, sampling-review and research-question digests are bound.

The controller cannot ask the witness to enroll a new question or reset state.
Creating another mirror after any witness history exists is rejected. Restoring
the mirror to an earlier snapshot produces a head mismatch and blocks further
reservations. An evaluator cannot reserve, a custodian cannot consume, and worker
certificates cannot access either service. Concurrent requests use a remote
compare-and-swap plus SQLite transactions; only one request with the same head
can advance it. Witness absence, divergence and partial cross-service commits
block progress. They do not refund alpha or silently initialize a database.

`attempt_service.py` supplies actual HTTPS servers and clients with mutual TLS,
CA/hostname validation, exact certificate pins, bounded unique-key JSON and role
ACLs. Controller-to-custodian and evaluator-to-witness paths are tested through
real child processes. The witness's Ed25519 key stays on its server. Clients
receive public signed events, not private keys or protected answers.

**Operational separation is not deployed.** Test services run under this Mac's
operator. To resist that administrator, the witness enrollment, journal, key and
backups must be on a separately administered host. The independent custodian
must preserve its receipts outside the controller's control. A witness operator
who rolls back the witness itself can defeat its guarantee; a same-owner hash
chain, container or second key directory is insufficient. Recovery from a
commit/mirror mismatch requires a separately audited replay of witness history;
there is no automatic recovery API in this release. Inference exceptions can
leave custody awaiting the committed consumption receipt; preserve and audit the
witness event instead of recreating the mirror.

The inherited `ModelRegistry` and legacy local qualification APIs remain intact
and retain their documented trusted-operator assumptions. They are not migrated
to this witness. Pretrained v5 candidates remain ineligible for those legacy
qualification/activation routes. This release cannot claim all P0 governance
defects closed merely because new witness tests pass.

## Authority chain and execution

`evidence_chain.py` verifies six typed records, exact predecessor-envelope
digests, pinned distinct controller/custodian/evaluator/promotion/executor keys,
exact artifact/runtime identity, witnessed consumption, explicit gates and
simulation-only scope. Missing predecessors, replaced tensors or identities,
signature impersonation, stale consumption, expired approvals and failed gates
are rejected. The caller must derive the expected identity from independently
verified artifacts; the verifier does not derive truth from a signed assertion.

Its output is **verified evidence only**. It does not create a qualification
record, run a protected model evaluator or issue a runtime capability. Even a
valid synthetic test chain cannot activate a pretrained model. The existing
`promotion.py` remains the activation entry point; learned backend activation and
external tool execution are unavailable. There is no additional promotion path.

`restricted_execution.py` is an optional macOS process restriction: declared
authority paths cannot be read, writes outside the output and network access are
denied, and credentials are not inherited. A real child attempts to read holdout
and key sentinels and connect to the network in the test. Unsupported hosts fail
closed. This is not automatically applied to the historical training worker,
does not provide separate administrators, and is not a complete hostile-code
sandbox. Linux/rootless-container settings are deployment requirements only,
not exercised deployment evidence. Training workers remain trusted implementation
code; their self-reports are not GPU-host attestation.

## Dataset mechanisms and review

Six additional public mechanisms cover a numeric budget condition, permission
provenance, the latest valid correction, a novel glossary, a missing deadline
and changed permission scope. Labels depend on the material constraint and
speaker, not a domain noun. Each has opposite-label pairs. All were authored by
the same operator. Related variations and mechanisms are explicitly correlated;
there are six mechanisms, not 72 independent families.

`lineage_review.py` screens normalized text, parent/source/template/generation
lineage and lexical near-duplicates. A custodian-side protected fingerprint index
denies derived prompts, parents, sources, templates and mechanisms without
exporting protected answers. Optional bounded cosine screening accepts identified
embeddings and emits unresolved semantic flags. It does not compute trusted
embeddings or auto-certify independence. Actual pretrained embedding production
and human/independent review were **not performed** in this release.

Signed public reviews must resolve all flags and cannot clear prohibited lineage
or truncated screening. Their scope never becomes independently qualified
sampling. No independent authors, protected population or human review has been
fabricated. Source IDs alone cannot establish independence. Existing protected
family statistics and alpha thresholds are unchanged.

## Checkpoint decisions and calibration

`research_selection.py` makes public learning diagnostics a distinct decision
from qualification admission. It preserves initial, best public learning and
terminal snapshots and their unsafe/invalid results. It cannot alter a signed
historical safety selector. The retrospective analysis explicitly distinguishes
the highest observed learning-curve step from a retained checkpoint: missing
weights are not reconstructed or declared candidates. Its policy is post hoc,
not retroactively preregistered.

Public temperature scaling is fit on development only and evaluated on public
regression. Brier score, calibration error and coverage/conditional-error curves
are reported. Low-confidence escalation is not counted as correct clarification;
ASK correctness still requires an ASK label. Confidence and model-generated
authorization claims never grant permission. These diagnostic calibration gains
do not establish independently calibrated abstention.

## Replay and stochastic controls

The historical sequence-v1 source remains exact. New sequence-v2 separately signs
its source identity and adds declared reservoir or family-stratified replay,
bounded storage, separate current/replay RNG streams, per-update RNG resets,
actual example IDs, current-example matching against fresh controls, full stage
predictions, tokens, timing, trainable hashes and bounded safetensors. Replay
never accepts development or protected records. Experimental replay changes
prior exposure and work; equal steps do not imply equal compute, and batch shape
can still affect dropout draws despite matched initial RNG states.

The new local CLI consumes a durable one-use development budget before training
and preserves interruptions. This local budget is controller-owned, is not
rollback-resistant and is not a financial cap. It is distinct from global
qualification attempt accounting. Sequence-v2 is currently admitted for local
FP32 only. No cloud job silently falls back to CPU; no new sequence-v2 T4 campaign
has been run. The historical Colab executor, hash-enforced locks, monitoring,
watchdog, failure records and nine exact cleanup confirmations are preserved.

Two bounded native local smoke runs each completed eight updates. The first
preceded final CLI budget/parameter-change hardening and is retained with its
exact worker source archive. The final run has durable budget evidence and fresh
per-record checkpoint replay. These demonstrate working machinery, not a new
retention advantage or repeated learning-to-learn evidence.

## Remaining release gates

The externally administered witness, independent custody and protected pretrained
evaluator are not deployed. No independently reviewed final population exists.
Safety/pair targets are unmet, repeated novel-task retention and controlled
compute comparisons remain absent, and recursive improvement is deferred.
Candidates remain NOT_EVALUATED, independent generalization NOT_ESTABLISHED,
and external deployment prohibited. A research prototype is not guaranteed AI
alignment.

Transport and deployment references: [Python SSL](https://docs.python.org/3.12/library/ssl.html)
and [Docker rootless mode](https://docs.docker.com/engine/security/rootless/).
These documents describe mechanisms, not certification of this deployment.
