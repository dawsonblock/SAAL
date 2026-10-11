# Reproduce v0.9.4 experiments

Use Python 3.12 on macOS ARM64 or Python 3.13 on Linux x86_64. Install the corresponding `configs/dependency-locks/pretrained-*.txt` with `uv pip sync --require-hashes`. The legacy platform locks remain unchanged for historical reproduction. The upstream weights occupy about 4.1 GB and are kept outside the source release.

```sh
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --require-hashes configs/dependency-locks/pretrained-macos-arm64-py312.txt
PYTHONPATH=src .venv/bin/python -m saal_rsi.pretrained_cli --help
PYTHONPATH=src .venv/bin/python -m pytest tests
```

Fetch each exact repository/revision from `configs/pretrained/registry.json` into a separate model cache. For example:

```sh
PYTHONPATH=src .venv/bin/python -m saal_rsi.pretrained_cli fetch \
  --repository answerdotai/ModernBERT-base \
  --revision 11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88 \
  --cache /absolute/path/to/model-cache
```

Use `model-spec`, `dataset` and `sign-job` to construct a new bounded research trial. The signing key must be locally owned, outside the package and excluded from worker uploads. Existing signed examples authenticate their original source identity; source edits require new admission rather than rewriting old evidence.

`saal-pretrained colab` accepts signed job files and an exact public-key pin, a new workspace, artifact destination, durable ledger and total deadline. The total must cover at least 1,200 seconds for session/setup stages plus each admitted training deadline and 1,400 seconds of replay/transfer allowance. No implicit compute retry or resume is implemented. Failed runs keep their logs and budget records.

Use `verify` for structure, identity, hashes, finite tensor inventory and metric reconstruction. Actual prediction reproduction additionally requires the verified base snapshot, exact environment and `pretrained_worker --replay` in a reviewed staged worker. The replay worker operates in a new process; same-host replay is not an independent evaluator. GPU parity and CPU parity must be reported separately rather than inferred from load success.

Historical v0.9.3 is preserved as an unchanged ZIP under `provenance/`. Its original lock, manifest, signed jobs and source are authoritative for old results. Draft v0.9.4 failed trials include their own staged worker archive, signed jobs and cleanup evidence; do not apply a later validator to rewrite their original results.

## Explicit pretrained job commands

Generate a Qwen FP32 specification (the default for Qwen after the recorded FP16 failure) and the public compatibility population:

```sh
PYTHONPATH=src .venv/bin/python -m saal_rsi.pretrained_cli model-spec \
  --repository Qwen/Qwen3.5-0.8B --method lora --rank 8 \
  --backend colab --precision float32 --max-input-tokens 256 --out qwen-spec.json
PYTHONPATH=src .venv/bin/python -m saal_rsi.pretrained_cli dataset \
  --kind generative --out generative.json
PYTHONPATH=src .venv/bin/python -m saal_rsi.pretrained_cli sign-job \
  --spec qwen-spec.json --dataset generative.json \
  --private-key /private/controller.key --out qwen-job.json \
  --steps 20 --seed 29 --batch-size 1 --accumulation 2 \
  --checkpoint-interval 20 --max-new-tokens 80 \
  --learning-rate 0.0001 --initial-loss-scale 1 --max-seconds 1800
PYTHONPATH=src .venv/bin/python -m saal_rsi.pretrained_cli colab \
  --jobs qwen-job.json --pin YOUR_64_HEX_PUBLIC_KEY \
  --workspace /new/colab-workspace --artifacts /new/artifacts \
  --ledger /private/experiment-budget.db --max-seconds 4400
```

The example admits a new compute attempt; it does not reuse a consumed job or qualify the result. Use actual paths and a locally generated raw Ed25519 key. The CLI cannot generate a production deployment approval. For ModernBERT, select `head`, `full` or `lora`, specify the batch/learning rate/budget explicitly and add `--class-balance` only for the registered ablation.

`sequence-protocol` signs a separate shared/isolated adaptation protocol. Supply an already signed LoRA job with enough total steps for **both** the experimental and fresh-control branches, a positive completed ordinary-learning report, repeated `--group` category lists, explicit `--steps-per-stage`, `--replay-ratio` and `--max-seconds`. The `sequence` command accepts that signed protocol and uses an exact monitored Colab session. The exported result preserves the full retention matrix, fresh-control performance, work and all stage adapters; sequence-stage cross-process prediction parity remains separate evidence.

Use `tools/replay_pretrained_bundle.py` to reconstruct an ordinary candidate on the controller from verified official snapshots. The default replays the full development population. An explicit `--records` limit produces a report marked partial. Use `tools/analyze_pretrained_research.py --artifacts ... --pin ... --out ...` for authenticated public distributions; include the separately distributed full-tuning bundle if checking that run.

The full-tuning evidence ZIP is a licensed wrapper around the original authenticated `bundle.zip`. Extract the outer archive, then extract `bundle.zip` into a dedicated directory before complete-bundle verification. The main source evidence manifest records both wrapper and inner-export SHA-256 hashes.
