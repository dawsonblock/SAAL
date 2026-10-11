# Decision Corpus v1 review candidate — release report

This additive capsule prepares the Reviewed Decision Corpus milestone. It does not complete human review, establish independent generalization, or upgrade a trained model.

600 assistant-authored drafts form 300 material counterfactual pairs. Public train/validation/test contain 360/120/120 records, balanced across CHEAP, FAST, ASK and REFUSE. Seven generator families remain grouped within their splits. Their shared policy, renderer and numerical concepts prevent treating them as independent populations. All labels remain proposed.

The mechanical validator verifies exact fields, policy outcomes, prompt reconstruction, digests, pair lineage, one changed condition, split grouping and balance. A deterministic rebuild reproduces the canonical corpus. Partition projections and blank review forms match the corpus. A pinned local ModernBERT tokenizer accepts all prompts within 256 tokens (maximum143); no weights were loaded or updated.

38 focused tests pass, including forged/stale reviews, identity reuse, rejection, adjudication, tampering, unresolved split review, invalid predictions and native admission. Test signing keys and reviews are synthetic fixtures, never human endorsements. The first bridge signature mismatch is preserved in tests-first-bridge-failure.log; the corrected bridge passes. These tests are distinct from prior full SAAL-RSI/Colab validation.

The 211 frozen source/config/test/tool files of the v0.9.4.2 base remain unchanged. This capsule requires that base for native bridge integration tests; it does not duplicate the full historical release. Base preservation evidence is included. Historical learning failures are not rewritten.

## Pending gates

- Human reviews completed: zero. Two independently enrolled reviewers and an adjudicator remain to be arranged.
- Semantic split review: pending, including 17,262 lexical similarity flags. Similarity flags need interpretation; accepting shared public policy does not certify independence.
- Policy acceptance: pending human review. The bounded synthetic policy omits realistic units, ties and many real-world actions.
- Training, real model behavior comparisons, retention and fresh-runtime Colab training/replay: not run.
- Independent sampling/custody and empirical qualification: not established; NOT_EVALUATED.
- Dynamic LoRA: deferred until a competent fixed-LoRA baseline exists.

Signatures bind review records to enrolled keys and the corpus. They do not prove a signer is human or operationally independent. The training bridge requires full reviews and separate signed job admission; it grants no qualification, promotion or tool authority. Public test labels are exposed in this package and cannot be relabelled as a protected holdout.

Research prototype; no guaranteed AI alignment. A review candidate is useful preparation, not evidence of behavioral learning.
