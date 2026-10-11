# SAAL-RSI v0.9.4 — Full Pretrained Learning Upgrade

A governed research platform for native ModernBERT classification and Qwen3.5 structured generative adaptation. This package preserves the SAAL authority system, Dream-RSI harness, historical neural workers, original release archives and negative evidence.

**All new candidates remain `NOT_EVALUATED`. External deployment is prohibited.** Public learning, engineering validation and independent qualification are separate outcomes. This is a research prototype with incomplete hardening, not guaranteed AI alignment or demonstrated recursive self-improvement.

## Implemented

- Exact official ModernBERT and Qwen post-trained/Base snapshots, native tokenizers and reviewed architecture/adapter targets. Upstream weights stay in a separately authenticated cache.
- Versioned signed model/job contracts, source and dependency identity, four-class ModernBERT head/full/LoRA training, and Qwen completion-only supervised LoRA with strict JSON oracles.
- Initial, selected and terminal safetensors snapshots; frozen-weight checks, exact tensor inventories, prediction/metric reconstruction and fresh-process replay.
- Public provenance/lineage checks, family-group partitions, counterfactual pairs, lexical screening, class coverage, calibration diagnostics and field-level generative scoring. Dataset independence is explicitly unestablished.
- Matched multi-seed public experiments, shared/isolated adapter research APIs, replay, full retention matrices and fresh-learner controls.
- Actual named Colab T4 execution with hash-enforced dependencies, durable attempt reservations, streamed progress, deadlines, independent local watchdog and exact-session shutdown confirmation.
- Fail-closed historical qualification/promotion boundaries. A v5 pretrained job cannot be silently treated as an older qualified model.

## Evidence and results

Read [the research report](docs/PRETRAINED_RESEARCH_REPORT.md), [release report](RELEASE_REPORT.md) and [known limitations](docs/PRETRAINED_KNOWN_LIMITATIONS.md) before interpreting the scores. Results include unsuccessful training, checkpoint-selection regressions, malformed outputs and forgetting.

On the public semantic pilot, the five terminal ModernBERT LoRA checkpoints average **66.4%** development accuracy, starting at **23.0%**. The safety-oriented selector retains several initial checkpoints; selected-checkpoint mean accuracy is **39.5%**. The lexical control scores **42.2%** on the same development inputs. These findings show why terminal learning gains must be reported alongside selected-candidate failures. The proposed counterfactual, safety and independent-generalization goals remain unmet.

One shared-adapter retention pilot loses all refusal/clarification accuracy after the final task without replay. The matched replay pilot retains that group's accuracy, with other residual losses. This is one-seed, correlated public evidence and does not prove robust transfer.

## Start here

Use the tested Python/platform hash lock in `configs/dependency-locks/`; see [reproduction commands](docs/PRETRAINED_REPRODUCTION.md). The new entry point is:

```sh
PYTHONPATH=src python -m saal_rsi.pretrained_cli --help
```

It supports dataset generation, pinned snapshot retrieval, signed bounded jobs, local execution, monitored Colab campaigns, explicit sequence protocols and artifact verification. Controller keys remain private and outside the package. No evaluator or custodian secret is transferred to training.

Verify inherited preservation with `tools/verify_pretrained_history.py`, the complete package with `tools/verify_manifest.py`, and the controller source/evidence capsule with `tools/verify_pretrained_release.py --pin ...`. `tools/verify_archive.py` performs a real fresh extraction, manifest verification and the full test suite.

The main ZIP includes source, tests, locks, adapters, public datasets, runtime logs, replay evidence, failure records, signed research attribution, reports and a hash manifest. Large full-tuning snapshots are a separate authenticated evidence ZIP; [the evidence map](docs/PRETRAINED_EVIDENCE_MAP.md) explains which bundles are complete or intentionally metadata-only.

## Scope and migration

[Architecture](docs/PRETRAINED_UPGRADE.md) · [Migration](docs/PRETRAINED_MIGRATION.md) · [Reproduction](docs/PRETRAINED_REPRODUCTION.md) · [Evidence](docs/PRETRAINED_EVIDENCE_MAP.md)

The original v0.9.3 ZIP is unchanged under `provenance/`, with its original source, checkpoints, manifests and reports. New code is additive and versioned. Final source admission repairs do not rewrite past jobs; approved historical workers are verification/replay-only.

Independent protected pretrained evaluation, independently operated authority services, a large reviewed task population, QLoRA, dynamic rank, learned routing, combined-model grading and recursive improvement remain outside the validated scope. No development result authorizes external execution.
