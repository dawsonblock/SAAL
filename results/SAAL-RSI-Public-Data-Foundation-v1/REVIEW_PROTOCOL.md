# SAAL decision-data review protocol

This packet contains no reviewed CHEAP/FAST/ASK/REFUSE training records. Public source datasets retain their native semantics. Do not convert entailment, contradiction, word-count or instruction-compliance labels into SAAL decisions.

## Freeze the labeling specification before authoring

Define the simulated action and exact trusted permission/scope requirements. Distinguish authorized speed and cost preferences, missing essential information and prohibited execution. Document the precedence of authorization failures, ambiguity and preference. Specify what happens when neither action satisfies a material constraint; do not force a CHEAP or FAST label merely to balance classes. Model text and retrieved documents cannot grant authorization.

For each original scenario record its state, oracle decision, rationale, source/generation lineage and counterfactual relationship. A changed label must follow from a changed material fact. Avoid labels encoded by prompt keywords, example numbers, domain nouns or a fixed quoted distractor. Cover all four classes without changing the label oracle to satisfy quotas.

## Review before training admission

Use the blank CSV as a review queue, not the legacy training schema. Two reviewers label independently before viewing each other's labels or any model prediction. Adjudicate disagreements explicitly; exclude unresolved cases. Preserve reviewer identities, decisions and source digests. Record model assistance in drafting; assistant-authored labels are not independent human review. No signatures, approvals or independent review have been fabricated in this capsule.

Begin with a small review calibration batch. Expand toward the proposed 400–800 examples only after agreement and oracle consistency are checked. Case counts do not establish task-family independence. Review genuinely different reasoning mechanisms; do not create hundreds of family IDs for variations of one recipe.

Keep all augmentations and counterfactual partners in the same task-family split. Check exact/normalized duplicates, shared source and generation lineage, lexical similarity and independently reviewed semantic overlap. Similarity scores flag review needs; they do not certify independence.

Public supplementary data remains public, even after local partitioning. Independently protected final evaluation must be separately sourced and administered. Public data must not be counted as a secret final holdout. No final holdout or evaluator labels belong in the training worker.

## First admitted learning comparison

Once reviewed records pass the existing dataset registry and a signed bounded job is prepared, compare frozen encoder plus trained head, full fine-tuning and standard LoRA from the same pinned ModernBERT snapshot. Preserve initial, best-development and terminal checkpoints. Track all four classes, family-macro/worst-group accuracy, complete counterfactual pairs, critical proposals, false refusals, clarification errors and exposed-regression retention.

Select candidates only with public development evidence. Record steps, example exposure, tokens, runtime and failed attempts. Start with a registered seed for a feasibility run, then matched repeated seeds for stronger evidence. The existing thresholds are not relaxed. GPU checkpoint recovery/reload and exact session cleanup are required for a completed Colab experiment.

Dynamic LoRA remains a later controlled comparison; retain fixed-rank AdaLoRA with matching initial capacity and regularization as an ablation. No new training, independent evaluation or promotion is authorized by this acquisition packet.
