# Human review and separate training admission

No human review has been performed or fabricated. The two CSV forms contain prompts and empty decision/rationale cells; model-proposed labels are in a separate `CORPUS.json`. Give each reviewer only their blind form and the policy, without the proposal file, the other review, model predictions or an instruction to agree with the oracle. This release cannot enforce that separation on an operator-controlled computer.

Enroll two actual reviewers and a distinct adjudicator by authenticating their public keys outside this archive. Fill the reviewer-policy template with 64-character Ed25519 public-key hex values. Do not bundle private keys. Distinct keys prove key separation, not independent humans or separate administration; identity and independence require external verification. All test review signatures are synthetic ephemeral fixtures and are never released as human evidence.

Reviewers enter CHEAP, FAST, ASK, REFUSE or REJECT, with a reason for every record. REJECT preserves a negative signed review and blocks admission. Fix ambiguous wording, wrong labels or rejected scenarios in a new corpus version, preserve the rejected version, and repeat its reviews. The builder refuses to overwrite a frozen candidate.

A label disagreement requires a signed, reasoned adjudicator batch. Agreement against the proposed oracle label also blocks admission and requires a corrected corpus version; the software never silently overwrites the frozen proposed label. Every review binds the complete corpus digest and exact prompt text.

The lexical queue has 17,262 cross-split flags, grouped into family-pair relationships, with the first 100 examples retained. It is a conservative screening lead, not proof of semantic leakage or independence. The curator must inspect the actual mechanisms and related examples. The semantic template binds the exact screen digest and has approval false by default. A family-group decision requiring rewriting, repartitioning or further review blocks this candidate. `ACCEPT_PUBLIC_SHARED_POLICY` is limited to disclosed public diagnostic use and must include reasoning; it does not declare independence or make the public test set secret.

From the candidate root, set PYTHONPATH to this `src` and, for native export only, the existing base release's `src`. Use the tested local environment. Commands never overwrite existing output files:

```text
python -m saal_corpus check corpus/CORPUS.json
python -m saal_corpus sign-review corpus/CORPUS.json corpus/reviewer_1-blind.csv --reviewer-id reviewer_1 --private-key /absolute/private/reviewer_1.key --output reviewer_1-signed.json
python -m saal_corpus sign-review corpus/CORPUS.json corpus/reviewer_2-blind.csv --reviewer-id reviewer_2 --private-key /absolute/private/reviewer_2.key --output reviewer_2-signed.json
```

The key paths are explicit placeholders; no such keys were created. Only the human reviewer should authorize signing their completed form. Use `--role adjudicator` for a completed adjudicator CSV when necessary. Complete the semantic template based on actual review, then sign it with `sign-semantic-review` and the enrolled adjudicator key.

```text
python -m saal_corpus assess corpus/CORPUS.json --policy enrolled-reviewers.json --review reviewer_1-signed.json --review reviewer_2-signed.json --semantic-review semantic-signed.json --output assessment.json
python -m saal_corpus prepare-data corpus/CORPUS.json --policy enrolled-reviewers.json --review reviewer_1-signed.json --review reviewer_2-signed.json --semantic-review semantic-signed.json --output native-data-bridge.json
```

Add `--adjudication adjudication-signed.json` if reviewers disagree. Assess/export currently fail because the delivered forms and enrollment are pending.

The native bridge preserves existing dataset-v3 custody flags conservatively and attaches review evidence separately. It maps public validation to development and public test to exposed regression. It never exports protected holdout data, reinterprets legacy job versions, creates a training budget or authorizes deployment. Feed a verified exported dataset through the existing source/config/budget-bound model admission, not directly into an unrestricted worker.

Before P1 training, independently verify the corpus, reviewer pins, review artifacts and their scope. A signed quorum is attribution evidence; it does not itself prove a human reviewed the text. For any protected qualification, establish separately controlled custody, sampling and attempt accounting. These gates remain unmet.
