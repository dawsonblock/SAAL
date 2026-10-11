# SAAL-RSI v0.9.4 pretrained research architecture

This release adds two separate native learning backends. It preserves the inherited SAAL authority controls, Dream-RSI harness, legacy neural workers, and negative research evidence. Training is distinct from evaluation and authorization. Every new result is `NOT_EVALUATED`; no model output, confidence score, JSON field, adapter, or development metric grants permission to use external tools.

## Official snapshot admission

`configs/pretrained/registry.json` binds exact upstream commits and SHA-256 hashes for every required file. The reviewed tensor inventories bind the native checkpoint layouts. Downloads use immutable commit URLs, bounded file sizes and SHA-256 verification. Cache files are regular files with an exact inventory; extra files and symlinks are rejected. No pickle checkpoint or unreviewed remote Python code is admitted.

| Backend | Repository | Revision |
| --- | --- | --- |
| Classification | answerdotai/ModernBERT-base | 11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88 |
| Generative | Qwen/Qwen3.5-0.8B | 2fc06364715b967f1860aea9cf38778875588b17 |
| Pretrained-only control | Qwen/Qwen3.5-0.8B-Base | dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 |

The requested ModernBERT revision's actual configuration is preserved, including its 512-position configuration. Current model-card statements about larger context are not silently substituted for this snapshot. The native tokenizer is used without truncation or token-type IDs. ModernBERT's existing MLM prediction transform is retained, and a new four-class linear classifier is initialized and trained. The frozen-head experiment trains the prediction transform and new classifier while freezing the encoder; it is not a zero-shot semantic classifier.

Qwen uses the official full `Qwen3_5ForConditionalGeneration` checkpoint with text-only native chat formatting. The vision weights are loaded and frozen; image/video inputs are rejected by the task contract. Processor configuration bytes and chat-template identity are authenticated even though image/video preprocessing is not invoked. No text-only conversion or model merger is used. LoRA targets include the actual DeltaNet `linear_attn.in_proj_qkv` projections and full-attention query/value projections, with an exact reviewed layer inventory. This does not adapt every possible DeltaNet projection.

Sources: [pinned ModernBERT repository](https://huggingface.co/answerdotai/ModernBERT-base/tree/11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88), [Qwen post-trained repository](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), [Qwen Base repository](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68), [native Qwen classes](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [native ModernBERT classes](https://huggingface.co/docs/transformers/model_doc/modernbert), [PEFT LoRA configuration](https://huggingface.co/docs/peft/package_reference/lora).

## Training and reconstruction

Version-5 jobs authenticate the model specification, complete public dataset, initialization seed, approved target modules, learning rate, initial loss scale, sampling budget, source identity, dependency lock and resource budget ID. Version-4 results preserve three safetensors files: initial, best development, and terminal. Exact tensor names, shapes, FP32 trainable values, finite values, hashes and selection history are verified. Frozen base parameters are hashed before and after adaptation; full tuning reports its smaller frozen scope rather than claiming backbone immutability.

ModernBERT supports frozen encoder/head training, full tuning, and fixed-rank LoRA plus head. Class-balanced cross-entropy is an explicitly admitted option. Qwen supports fixed-rank LoRA and a zero-update control. Prompt tokens are masked out of completion loss only after exact native-token prefix parity is checked. Malformed JSON, duplicate keys, extra fields, wrong types and invalid decisions fail the objective oracle without repair.

Mixed-precision training uses FP32 trainable tensors, FP16 autocast and a signed initial gradient scale. Nonfinite loss, nonfinite gradients or a skipped overflow update interrupts the attempt. The software does not silently change precision, targets, learning rate, batch size or checkpoint. QLoRA and unreviewed optimized kernels are not enabled. Deterministic generation uses native decoding; fresh-process replay measures actual checkpoint reconstruction, not merely successful loading.

The restoration identity binds base revision/files, architecture, tokenizer/processor/chat template, model specification, selected trainable checkpoint, separate classification-head tensor hashes, initialization seed, output schema, reviewed worker and runtime dependency versions. A standalone adapter is insufficient to reconstruct the model.

## Public datasets and inference scope

The bundled semantic pilot contains 1,280 examples grouped into recipe families and correlated counterfactual pairs. The generative pilot contains 120 examples covering structured decisions, extraction from trusted fields, and numeric sorting. These are deliberately small compatibility and learning pilots. They are not the independently reviewed 5,000–20,000-example population proposed in the roadmap. Every record includes provenance, generation mechanism, oracle version, rationale, difficulty, parent lineage, split and review status. Labels are oracle checked, not independently reviewed.

Family/cluster/template/source partition conflicts, normalized duplicates, forged labels, cyclic lineage and protected training inputs are rejected. Bounded lexical screening flags cross-partition similarities for review; semantic leakage review remains unperformed. Different recipe names do not establish independence. No protected population or custodian private key is present.

Measurements include before/best/terminal accuracy, family and category performance, four-class coverage, actual opposite-label pair correctness, same-label invariance controls, false refusals, missing clarification, unauthorized proposals, confusion, Brier score and descriptive calibration. Family bootstrap gain intervals are public diagnostics under exchangeability assumptions, not independently sampled generalization bounds. Published source/worker signatures establish attribution and integrity relative to their keys; they do not prove honest execution or honest evaluation.

## Compute monitoring and authority

Named Colab T4 sessions use a fresh hash-enforced Linux dependency environment. A durable local registry reserves every signed attempt before provisioning, including declared stage overhead. The inherited streaming monitor records loss, updates, token work, GPU memory and failures, with credential redaction. An independent local watchdog handles exact-session cleanup. Termination is confirmed through the named session's absence; uncertain cleanup remains an incident. Time reservations are not provider billing caps and are not rollback-resistant against a hostile local owner.

Workers receive public data, controller public keys and reviewed source. Controller/custodian/evaluator private keys and final-holdout labels are not transferred. A separate process reloads each completed candidate on the same compute host, then artifacts are downloaded and structurally validated on the controller. This is reproducibility under controller trust, not operationally independent evaluation or a GPU execution attestation. The local worker runs as the same OS user and is not an operating-system sandbox.

## Continual learning and qualification

`pretrained_continual.py` supports explicit signed public sequence protocols, shared/isolated LoRA, replay and matched fresh controls. It requires a positive recorded ordinary-learning signal before sequence execution, keeps complete retention matrices and records steps, tokens and trainable parameters. Isolated adapter routing by known task group is an oracle control, not a learned router. This API does not by itself establish transfer or grant a remote execution budget.

The inherited protected qualification and promotion boundaries remain fail closed. The legacy evaluator does not reinterpret v5 jobs as v4 or old synthetic models. This release supplies public evidence and reconstruction identities; operationally separated v5 custody/evaluation integration is still required before protected qualification. No synthetic or controller-owned review is presented as independent custody. No new promotion service, production execution path, dynamic-rank adaptation, automated model combination or recursive self-modification is introduced.

## Large-artifact transport

A separate reviewed transfer manifest binds job digest, exact total size, archive SHA-256 and ordered chunks of at most 32 MB. Downloads are bounded by a cumulative 900-second transfer deadline and per-chunk deadlines. Every chunk is authenticated before progress is reported, and the assembled ZIP is authenticated before extraction. Traversal, reordered chunks, total-size inconsistencies, cross-job manifests and corrupted chunks fail closed. Verified chunk pieces are disposable scratch; their manifest, transport logs, authenticated ZIP and reconstructed bundle remain. Transfer does not rerun training.
