# SAAL-RSI dynamic LoRA comparison

**NOT_EVALUATED. No independent generalization or external execution authorization.**

One corrected, separately signed PEFT AdaLoRA experiment is compared with the preserved fixed-rank-4 ModernBERT run. The frozen job binds the exact same native dataset, seed 47, 600 updates, batch 8, rate 0.0003 and 1,800-second training/evaluation deadline. Dynamic pruning starts after step 100 and fixes the final mask at step 500, followed by 100 terminal updates. Initial rank is 8 over 22 query/key/value projections; the final global assigned rank budget is at most 88 (average rank 4).

| Method | Initial dev | Selected dev | Terminal dev | Terminal complete pairs | Terminal train | Terminal exposed regression | Critical terminal dev |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-rank-4 | 23.44% | 39.06% | 41.15% | 16.67% | 87.76% | 27.27% | 6 |
| adalora-8-to-average-4 | 23.44% | 25.00% | 33.33% | 3.12% | 62.24% | 18.18% | 3 |

## Rank, work and checkpoints

- fixed-rank-4: 864,004 physically trainable parameters, 359,964 tokens, 528.64 measured training/evaluation seconds; selected step 400. Selected screen: {'pair_correctness': False, 'worst_family': False, 'class_coverage': False, 'critical_proposals': True, 'invalid_outputs': True, 'passed': False}. Terminal screen: {'pair_correctness': False, 'worst_family': False, 'class_coverage': True, 'critical_proposals': False, 'invalid_outputs': True, 'passed': False}.
- adalora-8-to-average-4: 1,134,516 physically trainable parameters, 359,964 tokens, 448.84 measured training/evaluation seconds; selected step 300. Selected screen: {'pair_correctness': False, 'worst_family': False, 'class_coverage': False, 'critical_proposals': True, 'invalid_outputs': True, 'passed': False}. Terminal screen: {'pair_correctness': False, 'worst_family': False, 'class_coverage': False, 'critical_proposals': False, 'invalid_outputs': True, 'passed': False}.
- Terminal assigned rank budget: 88; per-target assignment: {'model.layers.0.attn.Wqkv.lora_E.default': 8, 'model.layers.1.attn.Wqkv.lora_E.default': 6, 'model.layers.10.attn.Wqkv.lora_E.default': 4, 'model.layers.11.attn.Wqkv.lora_E.default': 5, 'model.layers.12.attn.Wqkv.lora_E.default': 5, 'model.layers.13.attn.Wqkv.lora_E.default': 2, 'model.layers.14.attn.Wqkv.lora_E.default': 2, 'model.layers.15.attn.Wqkv.lora_E.default': 4, 'model.layers.16.attn.Wqkv.lora_E.default': 4, 'model.layers.17.attn.Wqkv.lora_E.default': 3, 'model.layers.18.attn.Wqkv.lora_E.default': 3, 'model.layers.19.attn.Wqkv.lora_E.default': 3, 'model.layers.2.attn.Wqkv.lora_E.default': 6, 'model.layers.20.attn.Wqkv.lora_E.default': 4, 'model.layers.21.attn.Wqkv.lora_E.default': 1, 'model.layers.3.attn.Wqkv.lora_E.default': 5, 'model.layers.4.attn.Wqkv.lora_E.default': 6, 'model.layers.5.attn.Wqkv.lora_E.default': 1, 'model.layers.6.attn.Wqkv.lora_E.default': 4, 'model.layers.7.attn.Wqkv.lora_E.default': 6, 'model.layers.8.attn.Wqkv.lora_E.default': 1, 'model.layers.9.attn.Wqkv.lora_E.default': 5}.

The allocation callback executes after every optimizer update while gradients remain available. Orthogonal regularization is added explicitly because classification uses an external cross-entropy loss, which would otherwise bypass PEFT's loss-based penalty. At final-mask steps, pruned singular values are remasked after optimization. Initial/best/terminal safe tensor exports retain the full initial-rank matrix shapes plus signed mask metadata. This is adaptive masking, not physical storage compaction or reduced tensor memory.

Fresh-process replay validates every tensor name/shape/dtype, finite values, rank-mask inventory, pruned zeros and all three snapshots' recorded predictions. It also checks initial-prediction parity and identical training token exposure against the authenticated fixed control. Worker self-reports are not hostile-worker attestation; the local controller signs file integrity only.

## Research decision and limits

The preregistered public screen remains 90% complete pairs, 80% worst group, all four classes correctly represented and no invalid/critical proposals. It is unchanged by the request for dynamic rank. Failed screens remain failed. The new contract is explicitly experimental AdaLoRA; it never reinterprets or exports a native fixed-LoRA v5 result. No authority or promotion path is expanded.

Rank allocation, SVD factorization, orthogonal loss and higher initial physical capacity differ from fixed LoRA. One seed and equal update/token schedules do not establish equal compute, a causal rank-only improvement or a better efficiency frontier. The same operator-authored correlated public grammar is reused; no protected alpha or holdout is touched. The first dynamic attempt was stopped when a replay-key prefix defect was found; its source, partial tensors and spent ledger are preserved. A fresh source-bound admission reruns from the same upstream initialization after adding exact mask-name normalization and two regression tests. No interrupted receipt was reused or reclassified. Complete failed attempts and the earlier curriculum results remain in the delivery.

Implementation follows the [official PEFT AdaLoRA API](https://huggingface.co/docs/peft/package_reference/adalora) and the installed, version-pinned implementation; the frozen source includes the callback and regularization logic. No new Colab/T4 run, continual-learning proof, independent custody deployment or production qualification is claimed.
