# Pretrained model provenance

The release registers these official repositories and immutable revisions:

- `answerdotai/ModernBERT-base` — `11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88`.
- `Qwen/Qwen3.5-0.8B` — `2fc06364715b967f1860aea9cf38778875588b17`.
- `Qwen/Qwen3.5-0.8B-Base` — `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

Their official repository metadata identifies Apache-2.0 licensing; the Apache License is supplied in LICENSE. Exact downloaded model-file and native tokenizer identities appear in `configs/pretrained/registry.json`, with official metadata and reviewed safetensors headers beside it.

Base snapshots are retrieved separately. Included research heads, adapters and the separate full-tuning evidence are modified experimental derivatives: SAAL-RSI initializes a classification head or LoRA tensors and performs supervised updates on the declared public data. The upstream model publishers do not endorse or independently qualify this work. No model merger, third-party conversion or custom remote model code is used.

The separately distributed full-tuning evidence archive contains its own LICENSE, model provenance, inner bundle hash and the unchanged authenticated experiment export. Initial/selected/terminal snapshots remain distinct. No controller release signature grants independent qualification or deployment authority.

Pinned Python libraries are installed from their own hash-verified distributions; their source is not vendored as part of the new backends. Inherited SAAL, Dream-RSI and prior release notices remain preserved in the original source/archive hierarchy.
