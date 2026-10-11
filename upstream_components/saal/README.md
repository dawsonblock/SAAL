# SAAL — Separated-Authority Assistance Learning

**Version:** 0.5.0 | **Status:** experimentally runnable reference system, **NOT** a solved alignment problem, validated production security system, or autonomous tool agent.

SAAL implements the research claim that one optimizing model should not simultaneously define user goals, execute impactful actions, decide whether outcomes were good, admit training data, and authorize new weights. The repository has a working synthetic assistance laboratory, a genuinely offline-trained tiny classifier, explicit goal uncertainty/clarification, deny-by-default authorization, scope-limited one-use HMAC capability tickets, an example separated Unix-socket policy authority, outcome verification, an audited evidence chain, six-arm paired evaluations, evaluator signing, an independent qualification recommendation, and an optional externally signed LoRA training adapter.

**Important:** The included simulator measures task-selection performance against deliberately constructed synthetic preferences. It does NOT directly measure human welfare, demonstrate meaningful corrigibility in frontier AI, or establish control against a superintelligent or adaptively adversarial actor. A model trained on the training examples is not therefore proven aligned.

## What is new in v0.5.0 — independently witnessed studies and safer training admission

The full v0.4.0 system is preserved. **New v5 synthetic studies** require an
independently signed Ed25519 preregistration witness receipt, with a distinct
pinned public key, unique study identifier, implementation digest and report-bound
receipt digest. Qualification fails closed on missing, mismatched, altered,
replaced or downgraded witness evidence; independent full replay remains required.
The attestation is independent only if operators actually isolate witness,
evaluator, trainer and qualifier identities and credentials.

Training JSONL admission now validates a single byte snapshot, rejects duplicate
JSON keys and unexpected fields, requires exact `train` partition provenance,
rejects symbolic links and oversize records, and verifies the externally signed
manifest using a pinned Ed25519 public key. This reduces parsing and signing
ambiguities; it does **not** validate consent or truth of the admitted text.

The release includes **86 passing automated tests** (20 new v5 tests),
`docs/upgrade_v5.md` and `docs/safety_case_v5.md`.

**Quick start:**

```bash
python -m pip install -e '.[attestation]'
python -m unittest discover -s tests -q
python scripts/verify_release.py
saal demo
```

**Signed research workflow:** See `docs/upgrade_v5.md`. The sealed test seeds
and witness private signing key must live outside the trainer's trust domain.
A witness signature alone does not prove preregistration chronology: an
independent append-only anchor before evaluation is still required.

**Still research only:** No deployed language model, live-world execution,
mathematical guarantee of human benefit, or automatic model promotion.

## What is new in v0.4.0 — independently verifiable evaluator, pinned preregistration

This is a **full successor** to v0.3.0: all earlier sources, experiments, six-arm
paired study, SQLite authorization and budgets, model qualification and staged
candidate release remain included. The change addresses the major remaining
cryptographic trust boundary identified in the v0.3 security review:

- **Public-key evaluator attestation:** evaluator signs with an Ed25519 *private*
  key, separate qualifier verifies with a pinned *public* key. No shared HMAC
  signing secret is required for v4 studies.
- **Frozen-study commitment gate:** qualifier requires an externally witnessed
  preregistration digest, a public-key fingerprint frozen in the private
  preregistration, and mandatory exact synthetic replay on trusted source.
- **Implementation source agreement:** signed report includes canonical source
  digest; qualifier compares with its independently installed source tree.
- **Downgrade and forgery rejection:** v4 does not qualify v3 HMAC studies;
  tampered reports, changed implementation identifiers, swapped keys,
  unsigned/forged summaries and missing model replay fail closed.
- **Authority socket read timeout and complete release manifest inventory.**

**New experiment workflow:** see [`docs/upgrade_v4.md`](docs/upgrade_v4.md)
for fully reproducible CLI invocations. `python -m pip install -e '.[attestation]'`
and `python -m unittest discover -s tests -q` still run locally. The included
public-study reports and evaluation seeds are **not** sealed benchmarks.

**Scope:** The implementation remains a *synthetic alignment research prototype*,
not an LLM alignment solution, certified independent deployment system, or
production-safe agent. It has **no real-world autonomous action executors**.

## What is new in v0.3.0: authority identity, action-count limits and independent replay

This release upgrades the **existing v0.2.0 research codebase**, preserving its
synthetic learning laboratory, independent-capability design, typed tool policies,
consent memory, offline training, qualification, and separate human authorization.
New protections are implemented and tested, not just proposed:

| Change | Implementation | Demonstrable check |
| --- | --- | --- |
| Peer UID bound to authorized principal/task | `gate_server.py`: kernel peer credentials + private installed UID mapping | spoofed principal denied even with an installed delegation |
| Private configuration/key loading | `secure_key()` opens with `O_NOFOLLOW`, checks opened FD owner and permissions | symlink and world-readable secrets denied |
| Per-task zero-cost action quota | `types.Delegation.max_actions` + DB-reserved v3 one-use tickets | repeated zero-cost actions are blocked; spent quota survives restart |
| Complete deterministic report replay | `replay.audit_replay()` + v3 required gate in `qualification.qualify()` | fabricated but correctly signed synthetic outcome rejected |
| More signed-metric integrity checks | `qualification.py` recomputes proxy scores, clarification, injection counts, flags | forged aggregates rejected |
| Private study generator | `saal prereg-create` uses OS randomness and emits a hash commitment | locally generated randomized procedural seeds |
| Broader adversarial tests | `tests/test_v3_security.py` | UID spoof, duplicate JSON, nonce tampering, replay corruption |

**Scientific status:** This tests a tiny linear classifier and deterministic synthetic
preferences. It neither trains beneficial traits in an LLM nor proves real-world human
benefit, independent causal effects, strong corrigibility or general AI alignment.
Security remains conditional on correct isolation, distinct host/UID identities and
independent key custody. No production agent or external side-effect executor is shipped.

### Install / reproduce

```bash
python -m pip install -e '.[attestation]'
python -m unittest discover -s tests -v
saal demo
saal train --out artifacts/intent.json
saal evaluate --model artifacts/intent.json --out artifacts/public-study.json \
  --prereg configs/preregistered_eval.json
```

The public benchmark is an **unsigned demonstration**, never a sealed test. A genuine
v3 evaluation must run on a **separate controlled evaluator**:

```bash
# On the evaluator host, BEFORE testing and before distributing the candidate:
saal prereg-create --out /private/prereg.json --seeds 12 --episodes 100
# Publish the printed SHA-256 commitment to an external timestamped journal.
# Only evaluator personnel have access to /private/prereg.json.
saal init-key --out /private/evaluator.key

# After receiving a candidate from the separate trainer, on evaluator host:
saal evaluate --model /incoming/model.json --prereg /private/prereg.json \
  --eval-key-file /private/evaluator.key --out /private/evaluation.json

# On an independent qualifier host with vetted evaluator evidence:
saal qualify --report /private/evaluation.json \
  --prereg /private/prereg.json --model /incoming/model.json \
  --eval-key-file /private/evaluator.key --out /private/qualification.json
```

The v3 `require_replay` gate **fails closed** when `--model` is missing or when a
signed report differs from the re-executed simulator, even if summary statistics
have been forged to match. The qualifier currently needs the shared evaluator
HMAC secret; migrating to a public-key verification and independent host attestation
is still necessary for production-grade independence.

The old v0.2 replay database and capability tickets are **incompatible** by design.
Stop the service, drain or revoke issued capabilities, preserve the old DB for audit,
and initialize a new v3 replay DB; do not silently reuse an old schema.

See `docs/upgrade_v3.md`, `docs/authority_protocol_v3.md`,
`docs/replay_qualification_v3.md`, `docs/threat_model.md`, and `SECURITY.md`.

## Historical v0.2.0 functionality retained

## Version 0.2.0 upgrade — what actually changed

This is a **research reference**, not a production-safe autonomous agent or proof of general alignment.

* **Budgeted capabilities:** SQLite-backed cumulative per-task cost reservations, spent-capability accounting, replay protection, independent human approval, v2 policy/action/budget-bound HMAC tickets. Expired unused tickets release reservations, but consumed cost remains spent. `sim.purchase` is **synthetic only**.
* **Signed staging, no deployment:** independently qualified candidates can be attested with an Ed25519 qualifier key and separately signed by a human with a different Ed25519 key. A staged model is an immutable candidate, **not** a promoted/deployed model.
* **Stronger evaluation:** optional preregistration binding, procedural-seed cluster bootstrap (instead of resampling dependent episodes as IID), recomputed outcomes, finite numerical validation, abstention accounting, strict consistency checks.
* **Model/key input hardening:** invalid/nonfinite model weights and vocabulary are rejected; key and gate socket symlink checks fail closed.
* **v0.2 historical test suite:** includes budget, release, data-corruption and signature-forgery regressions.

### Upgrade quickstart

```bash
python -m pip install -e '.[attestation]'
python -m unittest discover -s tests -v
saal train --out artifacts/intent-model.json
saal evaluate --model artifacts/intent-model.json --prereg configs/preregistered_eval.json --out artifacts/evaluation-demo.json
```

Unsigned evaluations are demonstrations and **not eligible** for independent qualification. For signed evaluation `--eval-key-file` must be supplied together with `--prereg`; evaluator signing secrets must remain on an isolated evaluator host. See `docs/release_v2.md` for the two-signer review flow and `docs/upgrade_v2.md` for replay database migration.

**Known gap:** A digest in an evaluation is not a proof that a preregistration was frozen in advance; independently timestamp and publish commitments before evaluating. Reported synthetic outcomes are not real-world causal human benefit. The reference sandbox/gate only becomes a security boundary after OS/host isolation and proper credential placement.

## Run (Python 3.11+; basic project has no third-party runtime dependencies)

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
saal demo
saal session-demo --directory artifacts/session --consent-to-memory
saal train --out artifacts/intent-model.json
saal evaluate --model artifacts/intent-model.json --prereg configs/preregistered_eval.json --out artifacts/evaluation-unsigned.json
```

The last command produces an **unsigned exploration**. It is deliberately not eligible for independent qualification. The default study uses 5 fixed holdout seeds × 120 episodes = 600 paired cases in each of six arms. Each episode uses a hidden goal and six appointment options. The proxy reward intentionally favors not asking clarifying questions, even when that decreases the user's true synthetic utility.

For an independently authenticated report, the evaluation host (not the learner/trainer host) should own the secret and execute:

```bash
saal init-key --out /secure/evaluator.key
saal evaluate --model artifacts/intent-model.json \
  --prereg configs/preregistered_eval.json \
  --out /secure/evaluation-signed.json --eval-key-file /secure/evaluator.key
saal qualify --report /secure/evaluation-signed.json \
  --prereg configs/preregistered_eval.json \
  --eval-key-file /secure/evaluator.key \
  --out /secure/qualification.json
```

`qualify` returns a recommendation (`REJECT` or `ELIGIBLE_FOR_HUMAN_REVIEW`). **It never deploys weights**. An independently attested evaluator key does not guarantee the evaluator is impartial, and this example's HMAC verification requires a trusted qualification process. In production the preregistration must also be frozen and attested before running experiments. Do not put an evaluator key inside a repo, training host, downloaded ZIP, or actor container.

## Where the actual system lives

| Path | Implemented functionality | Limits |
| --- | --- | --- |
| `src/saal/agent.py`, `beliefs.py` | Online belief inference; expected-value-of-information clarification; correction overrides beliefs | Four synthetic objective classes |
| `src/saal/memory.py`, `runtime.py` | Consent-based preference memory with expiry/erasure; integrated synthetic session | Actor/evaluator/authority run in one demo process |
| `src/saal/train.py` | Offline SGD training and JSON weights for softmax classifier | Not LLM fine-tuning or beneficial-trait RL |
| `src/saal/sim.py`, `verifier.py` | Hidden-goal task generator, known synthetic outcome and constraint checks | Known reward is not real-world welfare |
| `src/saal/authority.py` | Static tool schemas, scoped delegation, human-credential approval, expiry, one-use HMAC, revocation | Key isolation requires distinct OS process/host |
| `src/saal/gate_server.py` | Unix-socket external authority backed by installed delegations | No trusted external execution adapter; no authenticated principal identity provider |
| `src/saal/monitor.py` | Independent low-value anomaly signal | Heuristic only; cannot certify a decision |
| `src/saal/ledger.py` | SQLite chain and optional independent external anchor digest | Without off-box anchor/WORM, attacker can rewrite chain |
| `src/saal/evaluation.py` | Six-arm matched-seed tests, proxy-vs-oracle outcomes, signed reports | Toy tests; no causal inference about humans |
| `src/saal/qualification.py` | Preregistration matching, paired bootstrap, Wilson upper critical-event checks, safety-regression checks | Synthetic qualification recommendation, not production proof |
| `src/saal/admission.py` | Ed25519 data admission attestation | Requires optional `cryptography`; signing key on separate trusted host |
| `scripts/train_lora_optional.py` | Real *optional* local PEFT/Transformers SFT training recipe with signed train input | Does not implement online RL, beneficial RL, or automatically qualify model |
| `src/saal/model_backends.py` | Optional OpenAI-compatible model **intent** API, strict JSON normalization | Not used by six-arm study; explicit network/consent needed |

## Source structure

```text
src/saal/
  agent.py           ask / propose / abstain reference actor
  beliefs.py         categorical belief and corrections
  memory.py          consent, TTL, deletion of explicit preferences
  runtime.py         integrated synthetic task flow
  train.py           train offline softmax; data separation
  sim.py             synthetic tasks, private scenario objective
  verifier.py        independent toy oracle outcome
  authority.py       capability and policy checks
  gate_server.py     isolated Unix-socket authority example
  monitor.py         monitoring signal (not enforcement)
  ledger.py          evidence chain and anchor validation
  admission.py       Ed25519 training-data admission
  evaluation.py      controlled paired six-arm trials
  qualification.py   safety-aware model release recommendation
  model_backends.py  optional backend, cannot issue grants
  cli.py             user commands
configs/             frozen proposed benchmark and example delegations
scripts/             study runner, optional LoRA, data admission
examples/            example synthetic-only training records
tests/               adversarial unit/integration tests
docs/                threat model, trust graph, APIs, experiments, roadmap
```

## Example authority process (Unix/Linux/macOS)

Run policy service as a **separate OS identity** with private keys, writable SQLite replay state and trusted-only delegation config; do NOT start it inside the actor's user account in production. First create two DIFFERENT secret files in a protected location:

```bash
saal init-key --out /secure/authority.key
saal init-key --out /secure/approval.key
saal gate-server --socket /secure/saal.sock \
  --authority-key-file /secure/authority.key \
  --human-key-file /secure/approval.key \
  --delegations configs/example_delegations.json \
  --replay-db /secure/replay.sqlite
```

The example delegation expires at a fixed future timestamp and is for demonstration only. The service responds to one JSON line `{"action":{"tool":"schedule.propose","args":{"slot_id":"A","cost_cents":0},"principal":"demo_user","task_id":"demo-1"}}`. A returned ALLOW ticket is **not an action**; a separate trusted executor must validate and consume it using the shared authority replay registry and must be the only component holding actual tool credentials. No real scheduling integration or human approval UI is provided. Control-plane protection requires isolated OS users, authentic principal assertions, authenticated admin endpoints, OS-backed credentials and hardened infrastructure—see `docs/deployment.md`.

## Experiment conditions

- **A:** Frozen, no goal inference, no questions, untrusted tool execution disabled in simulator.
- **B:** Frozen categorical uncertainty, rule-based cues, clarification.
- **C:** Offline-learned classifier, no questions.
- **D:** Offline-learned classifier plus questions.
- **E:** B plus simulated independent capability authority.
- **F:** D plus simulated independent capability authority.

None of these includes genuine beneficial-trait RL on an LLM, and the additional control layer is NOT an implicit improvement in intent inference. The six arms are useful diagnostic comparisons, **not a complete orthogonal causal factorial design**. See `docs/experiments.md` for needed extensions.

## Offline LoRA adapter (optional, explicitly not pre-run)

`pip install -e '.[train-lora,attestation]'` and follow `docs/training.md`. The trainer requires a separately signed train-only JSONL manifest and the signer's public key. It does not train on evaluation data or update the deployed actor. Models must have compatible attention projection module names (default `q_proj` and `v_proj`). Do not execute large-scale RL merely because the reference model passed toy tests.

## Non-negotiable boundaries

1. No learner authority over grant rules, human approvals, external credentials, evaluators, outcome facts, train-data admission, or model promotion.
2. Never present metric increases as evidence of human welfare without a warranted causal design.
3. Clarification and abstention must be possible; explicit correction beats inferred habits.
4. Each deployment receives only scoped privileges; destructive/model-control tools are denied outright here.
5. Evaluator secrets and hidden cases must be isolated in **deployment infrastructure**, not simply Python modules.
6. A rejection is a scientific result; do not tune on sealed failures and then continue claiming the same holdout is untouched.

See `docs/architecture.md`, `docs/threat_model.md`, `docs/experiments.md`, `docs/training.md`, `docs/deployment.md`, `docs/roadmap.md`, `docs/research_questions.md`, and `SECURITY.md`.

## v0.3 repeatable verification

```bash
python scripts/verify_release.py       # manifest consistency only
python -m unittest discover -s tests -q
python -m compileall -q src/saal
```

The public toy study found the offline-trained candidate **worse** than the
frozen controlled baseline in synthetic regret; see `docs/findings_v3.md` and
`reports/reference-study-v0.3.0-public-demo.json`. Do not reinterpret this
as a successful alignment experiment. No secret keys, hidden study seeds or
external credentials are distributed in this repository.
