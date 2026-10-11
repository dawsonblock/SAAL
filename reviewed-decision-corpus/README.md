# SAAL-RSI Decision Corpus v1 — review candidate

**600 assistant-authored draft records; zero completed human reviews. Training blocked. Qualification NOT_EVALUATED.**

This is preparation for the Reviewed Decision Corpus milestone, not a completed reviewed dataset or a new trained model. The existing source-data foundation and SAAL-RSI software are unchanged. ANLI, IFEval and NaturalInstructions labels were not converted into SAAL actions.

| Public partition | Records | CHEAP | FAST | ASK | REFUSE | Pairs |
|---|---:|---:|---:|---:|---:|---:|
| Train | 360 | 90 | 90 | 90 | 90 | 180 |
| Validation | 120 | 30 | 30 | 30 | 30 | 60 |
| Test | 120 | 30 | 30 | 30 | 30 | 60 |

There are **seven generator families**, not 600 independent scenarios. All current-preference, quoted-authorization, revocation, scope and expiry families are training-only. Spending-ceiling scenarios are validation-only; duration-cutoff scenarios are test-only. Noun substitutions are explicitly surface variations, not new independent families. A shared renderer/operator/oracle and related numerical concepts remain across splits. The 17,262 lexical flags and pending semantic review are published, rather than hidden by new IDs.

## Included

- Proposed decision semantics, bounded structured states, annotation schema and source/code lineage.
- Complete draft corpus, public partition files and 300 one-condition counterfactual pairs.
- Two blind review forms, enrollment and semantic-review templates. No fake human identities or approvals.
- Signature-bound review quorum, explicit rejection, adjudication and split-review checks.
- Review-gated native dataset bridge preserving conservative historical custody flags.
- Behavioral metrics: four-class confusion matrix/accuracy, macro-F1, class F1, family macro/worst-family accuracy, complete-pair correctness, action proposals on ASK/REFUSE, false refusal among gold CHEAP/FAST, and missing clarification among gold ASK.
- Negative tests for tampered labels/text, split/family substitution, stale/forged reviews, missing quorum, incomplete pairs, rejected records and unresolved semantic review.
- Qualification/gate report with human review, training and independent generalization explicitly pending.

The pinned local ModernBERT tokenizer inspected all 600 prompts without truncation: maximum **143 tokens**, zero exceeding 256. This loaded the tokenizer only, not model weights, and ran no training. Exact tokenizer file hashes and per-record counts are recorded. Recheck after any wording changes.

## Review and use

Read `POLICY.md`, then `REVIEW_WORKFLOW.md`. Use a separate authenticated workflow for actual human enrollment and approval. A public test partition can be held out from weight updates, but it is not an untouched independently custodied qualification population. This ZIP exposes all proposed labels to its operator. Repeated tuning against this public test must be disclosed as regression/development use.

`PYTHONPATH=src python -m saal_corpus check corpus/CORPUS.json` verifies the candidate. Source tests and native export tests additionally require the preserved base release's `src` and the already tested cryptography environment. `python -m saal_corpus metrics` accepts an exact record-ID prediction mapping for the selected partition, with strict labels and complete pair coverage; no model results were invented.

After actual reviews, compare frozen encoder/head training, full tuning and fixed LoRA under source- and budget-bound native jobs. Preserve initial, best-development and terminal models; select only from public development evidence. Run actual T4 training/recovery/fresh-process replay separately. Dynamic-rank adaptation stays deferred until a competent fixed baseline exists.

No baseline training, new Colab session, independent sampling, retention advantage, hostile-worker isolation or recursive improvement was established here. Signed review identity and independence remain external trust assumptions. The bridge cannot promote or execute tools. Original infrastructure/learning failures remain in their prior releases; this capsule does not overwrite them.
