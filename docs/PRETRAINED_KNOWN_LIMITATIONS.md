# Limits of the v0.9.4 research release

This release implements and tests native pretrained adaptation. It does not establish guaranteed alignment, independent generalization, transferable learning-to-learn, autonomous recursive improvement, or production readiness.

## Evaluation and data

The semantic pilot has 1,024 training, 128 development and 128 regression records. The generative pilot has 96 training, 12 development and 12 regression records. Correlated templates, counterfactuals and shared operators/oracles remain. Group partitioning and lexical screening prevent some obvious leakage; semantic review and independently sourced populations remain incomplete. The large, independently reviewed populations in the roadmap have not been constructed. Public regression is reusable diagnostic evidence, not sealed final evaluation.

Development data select checkpoints. Public regression metrics are recorded but the current fixed selector orders invalid outputs, unauthorized proposals, class coverage, worst-family accuracy, family-macro accuracy, pair accuracy and overall accuracy using development metrics. It is conservative relative to some utility gains and can retain step zero. Initial, selected and terminal predictions therefore remain mandatory. It does not enforce every proposed regression or retention threshold and cannot qualify a candidate.

Family-bootstrap intervals are descriptive and assume exchangeability; shared recipe lineage undermines independent population inference. No independent final holdout is evaluated. Historical family sign tests, alpha spending, one-use receipts and promotion controls are preserved, but their legacy evaluator rejects these v5 pretrained jobs. A separately reviewed pretrained custody/evaluation path is still required. No new job automatically gains compatibility with protected qualification.

## Learning and experiments

ModernBERT supports prediction-head training, full fine-tuning and fixed-rank LoRA. Qwen supports zero-update inference and fixed-rank LoRA. Trainable tensors are serialized as strict FP32 safetensors; base snapshots remain separate. LoRA alpha is fixed to twice the rank and dropout to 0.05 by source identity, rather than exposed as freely selectable job parameters. No QLoRA, dynamic rank, automatic architecture search, learned adapter routing or combined-model reward loop is admitted.

Five-seed head/LoRA and Qwen distributions do not replace independent task sampling. Full tuning is a single-seed comparison with the same example/step budget but a different learning rate and measured resource cost. The historical tiny Transformer and lexical controls are replayed on identical diagnostic inputs, but have different training/search histories. They cannot isolate pretraining causally.

The public sequential experiments reset the optimizer between stages, compare fresh learners with identical initialization and step schedules, and report actual tokens. Replay changes exposure; identical steps need not mean identical compute. The implementation can train shared and isolated adapters; routing isolated adapters by known task group is an oracle control. Continued optimizer-state recovery, pretrained training resume, dynamic allocation and independent transferable adaptation remain unqualified.

## Runtime and authority

Actual Colab T4 runs exercise a fresh hash-enforced Linux/Python 3.13 environment, native eager attention and selected precision. Mac CPU/Python 3.12 tests use a separate hash lock. These locks do not prove compatibility on other GPUs or operating systems. FP16 Qwen training produced a nonfinite gradient and was interrupted; separately signed FP32 jobs test the reference fallback. No silent precision change or automatic retry is permitted. Numerical and memory behavior of optional optimized DeltaNet kernels remains untested.

The controller, signer and evidence verifier operate under local owner trust. Private authority keys and holdouts stay out of workers, but workers are not a hostile-code OS sandbox and the roles are not independent machines or protected identities. Hashes, runtime declarations and controller signatures are integrity/attribution evidence, not GPU attestation or proof of an honest evaluator. Filesystem rollback and provider billing caps are not enforced by the local time ledger.

A fresh process on the same T4 host replays each completed ordinary candidate. Controller-host replay coverage is explicitly recorded. Both seed-17 Qwen candidates replay all 12 development outputs exactly; the ModernBERT seed-17 LoRA candidate replays all 128 predictions within tolerance. Every checkpoint of both public sequences restores on the CPU and reproduces recorded group accuracies. Sequence bundles did not record original per-record predictions, so that check establishes accuracy parity rather than identical per-record prediction parity. Other seeds have fresh same-T4 process replay, not controller-host CPU replay.

Named-session cleanup requires an actual absence response for the exact session. A network failure cannot prove teardown. Unconfirmed cleanup remains an incident. No model-generated `action_authorized` field, confidence score, training success, research report or release signature grants external execution permission. The inherited authority and promotion boundary remains authoritative; all new candidates remain `NOT_EVALUATED`.

The training attempt ledger enforces one-use reservations within an initialized local registry. It is not a global provider quota: a trusted owner can create a separate registry or roll back local storage. Controller signatures and declared budget IDs do not eliminate that local trust assumption. Protected evaluation attempts remain governed by the inherited separate custody machinery, which does not admit v5 pretrained candidates.
