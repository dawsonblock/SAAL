# SAAL-RSI Dynamic LoRA v4 — ModernBERT public research

Adds a versioned experimental PEFT AdaLoRA backend to the preserved SAAL-RSI v0.9.4.2 source and grammar-v3 experiment. Read REPORT.md and RESULTS.json for actual behavior, costs and failed gates. All outputs remain NOT_EVALUATED; no external deployment is authorized.

## Algorithm and scope

Official ModernBERT-base revision 11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88 is loaded through its native classification model and tokenizer. AdaLoRA targets the same inspected 22 Wqkv linear projections as fixed LoRA. Initial rank 8 is pruned by gradient-based importance to a global budget of at most 88 assigned directions (average rank 4). Warm-up is 100 updates, pruning runs through update 500, and the last 100 updates reuse the fixed mask. Alpha 16, dropout 0.05, importance smoothing 0.85/0.85, allocation interval 10 and orthogonal penalty 0.5 are frozen in the signed contract.

The callback runs after every optimizer step, before gradients are cleared. The explicit orthogonal penalty matches the installed PEFT formula; computing classification loss externally would otherwise omit it. Allocated ranks can differ per layer. Logs distinguish assigned directions, nonzero singular values, physically trainable parameters and tensor memory. Full initial-rank matrices remain allocated; adaptive masking does not compact checkpoints or reduce tensor-memory allocation.

Classification-head tensors and admitted AdaLoRA A/B/E tensors train. Backbone and rank-scaling constants remain frozen and hashed. This backend currently implements classification research, not Qwen generative dynamic adapters. It does not implement architecture search or autonomous learning-policy modification.

## Comparison, attempts and repair

Data, seed 47, 600 optimizer updates, batch 8, rate 0.0003 and native input bounds match the separately preserved fixed-rank-4 grammar-v3 run. Identical initial public predictions and total training token exposure are checked. Higher initial rank, SVD factorization, orthogonal loss, alpha and stochastic dropout streams differ, so this is a method comparison rather than a causal rank-only ablation. One CPU/FP32 seed and correlated operator-authored public cases do not establish independent generalization.

The first dynamic attempt was deliberately stopped after finding a replay-checker defect: PEFT's underlying-model mask names omit the wrapper prefix used by exported tensors. ATTEMPT_REPAIR.json, history/pre-repair-source/, original logs, partial tensor files and its INTERRUPTED ledger remain intact. A new source-bound admission starts from the upstream initialization with a fresh ledger after exact name mapping and regression tests. Original receipts are not reused or reclassified. No protected alpha is spent or reset.

Each admission reserves 2,300 local seconds, with a 1,800-second training/evaluation deadline and bounded fresh-process reconstruction. Both development reservations and measured consumption are disclosed. The local SQLite receipt resists accidental duplicate execution, not privileged filesystem rollback; it is not a billing cap or independent custody service.

Initial, selected and terminal checkpoints and metrics stay separate. The public screen remains 90% complete pairs, 80% worst group, correct coverage of all four decisions and no invalid/critical proposals. Public regression never selects checkpoints. Adding dynamic rank does not relax these gates or produce a promotion decision.

## Safe artifact identity and replay

The new saal-adalora-research-job-v1 schema binds the exact fixed-control signature/digest, native upstream model/tokenizer/dependency identities, dynamic schedule, worker source, budget and public screen. A native v5 configuration is a pinned comparison reference, never reinterpreted as an AdaLoRA execution contract. The AdaLoRA result has a distinct experimental schema and is not admissible as a native fixed-LoRA qualification record.

Full-shaped safetensors exports include all admitted trainable A/B/E and head tensors. Snapshot-specific mask metadata records every assignment. Fresh-process replay checks exact inventories, shape/dtype, finite values, mask names, zeroed pruned values and initial/best/terminal predictions with probability tolerance 0.0001. Controller signatures provide integrity relative to an authenticated key, not independent evaluator attestation or proof against a malicious host.

The main ZIP includes all dynamic tensor snapshots (including interrupted partial artifacts), source, tests, fixed-control evidence, signed jobs, logs and reports. No private keys are bundled. The complete historical base and earlier learning failures are included separately in the full delivery bundle. Heavy original upstream snapshots and full-fine-tuning tensors remain separately authenticated artifacts.

## Commands

Use the base release's tested macOS ARM64/Python 3.12 environment and the copied pinned dependency lock. Add this supplement and the base src directories to PYTHONPATH.

```sh
export PYTHONPATH=/absolute/dynamic/src:/absolute/SAAL-RSI-v0.9.4.2/src
python -m pytest /absolute/dynamic/tests
python /absolute/dynamic/tools/verify_dynamic.py --pin YOUR_AUTHENTICATED_CONTROLLER_PIN
python -m saal_adalora.worker replay /absolute/dynamic/experiment-r2 /absolute/model-cache
```

New training requires new output, ledger and controller-key paths. The sealed/private evaluator and custodian interfaces are not used by this experiment. No new Colab/T4 execution, GPU qualification, provider cleanup or external tool activation is claimed. Independent sampling, repeated seeds, matched-compute studies, retention evidence and operationally separated evaluation remain required before stronger research claims.

The implementation follows the official [PEFT AdaLoRA documentation](https://huggingface.co/docs/peft/package_reference/adalora) and inspected PEFT 0.21.2 source.
