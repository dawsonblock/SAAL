# Evidence map and reconstruction scope

The main source release preserves the original v0.9.3 ZIP and all inherited files. `tools/verify_pretrained_history.py` checks every incoming file and permits only the named release metadata changes. Original README/release reports remain under `docs/history/v0.9.3/`; historical failed learning and qualification evidence is not rewritten.

`configs/pretrained/registry.json` authenticates official model files and commits. `tensor_inventories.json` records the native upstream safetensors headers. Base weights are a separate verified cache, not bundled gigabytes in the source release.

`examples/pretrained_research/` preserves public research evidence. Completed ordinary bundles have exactly ten files and three trainable snapshots. Each signed job binds the dataset, source, base model, seed, training configuration and budget. Inventories hash every artifact; result metrics reconstruct from stored predictions. Controller signatures and pins authenticate attribution, not independent truth.

| Evidence | Scope |
| --- | --- |
| Campaigns 1–3 | Earlier interrupted/negative attempts and completed compatibility runs, original workers and cleanup |
| Campaign 4 | Seed-17 FP32 Qwen post-trained/Base pilots after explicitly admitted precision/data revision |
| Campaign 5 | Five-seed ModernBERT head/LoRA comparison plus one full-tuning trial |
| Campaigns 6–7 | Shared LoRA retention, no replay versus 25% replay, matched fresh controls |
| Campaign 8 | Four additional fixed-protocol seeds for each Qwen checkpoint |
| Controller replay JSON | Exact recorded subset/population, platform, tolerance and actual agreement |
| Final source compatibility evidence | Small admitted run/replay after final source hardening; not pooled with learning campaigns |
| Research summary and figures | Descriptive before/selected/terminal results and complete public retention matrices |

Large full-tuning snapshots are distributed separately as `SAAL-RSI-v0.9.4-ModernBERT-Full-Tuning-Evidence.zip`. Its size and SHA-256 appear in `full-tuning-external-archive.json`. The source ZIP contains that run's signed job, predictions, logs, metrics and hash inventory under `full-tuning-metadata/`, explicitly as a partial bundle. Validate the complete separate archive with the same pinned controller public key; a metadata-only directory is deliberately rejected by the ordinary complete-bundle validator.

Published historical worker archives have `VERIFY_AND_REPLAY_ONLY` scope. Their bytes and hashes are preserved for attribution and reconstruction. New execution requires the current worker identity and a newly signed, one-use admitted job. Source changes cannot silently authorize a historical job to run again.

The runtime monitor, signed job, controller verification, fresh same-T4 process replay and controller CPU replay are separate evidence. A released result or signed source capsule is not a custodian receipt, independent evaluator signature, protected qualification or deployment approval. All new candidates remain `NOT_EVALUATED`, and external deployment remains prohibited.

The full-tuning evidence ZIP is a licensed wrapper around the original authenticated `bundle.zip`. Extract the outer archive, then extract `bundle.zip` into a dedicated directory before complete-bundle verification. The main source evidence manifest records both wrapper and inner-export SHA-256 hashes.
