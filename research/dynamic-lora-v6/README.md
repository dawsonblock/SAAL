# SAAL-RSI Dynamic Adapter Research v6

Read [the implementation and limitations](../../docs/DYNAMIC_LORA_V6.md)
and the release report before interpreting scores. Qualification is
**NOT_EVALUATED**; external deployment is prohibited.

## Reproducible environment and checks

On macOS ARM64 / Python 3.12, install the existing platform lock in a fresh
environment. Colab uses the separate Linux x86_64 / Python 3.13 lock.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes --only-binary :all: \
  -r configs/dependency-locks/pretrained-macos-arm64-py312.txt
PYTHONPATH=src .venv/bin/python -m pytest research/dynamic-lora-v6/tests -q
PYTHONPATH=src .venv/bin/python research/dynamic-lora-v6/prepare.py check
```

The tests include real PEFT 1,000-step mechanism fixtures and archived official
ModernBERT CPU artifacts. The fixtures are not scientific experiments. For
fresh native reproduction, use a new output name and the exact registered
model cache. Keep the private raw Ed25519 controller key outside the repository.

```sh
PYTHONPATH=src .venv/bin/python research/dynamic-lora-v6/native_smoke.py \
  --key /private/location/controller.ed25519 \
  --cache /path/to/verified-model-cache --attempt native-engineering-new
```

## Monitored T4 pilot

Preparing and running are separate operations. Preparation creates three
signed admissions and a fresh durable public budget. Existing directories and
budgets cannot be reused or reset. Preserve every failed attempt. The new v6
controller identity is separate from the unchanged historical v5 public key.

```sh
PYTHONPATH=src .venv/bin/python research/dynamic-lora-v6/prepare.py prepare \
  --stage PILOT --key /private/location/controller.ed25519
PYTHONPATH=src .venv/bin/python research/dynamic-lora-v6/prepare.py run \
  --stage PILOT --cache /path/to/verified-model-cache
```

The run performs one provision request and no automatic training retries.
`pilot-experiment/colab/monitor-state.json`, `events.jsonl`, per-stage logs,
`registry-export.json`, recovered native bundles and controller replay receipts
record what happened. A failed provisioning request does not count as training.

Only an accepted complete pilot permits `--stage STUDY` preparation. Its 15
public comparison runs remain unqualified. `--stage DATA_CONTROL` defines the
separate exploratory two-seed parent-training comparison. Do not restart or
choose a checkpoint based on protected evaluation.

The documented cross-platform head reconstruction repair has its own explicit
`--pilot pilot-repair-02` admission. Both `prepare` and `run` accept that option.
It requires the retained original failed admission and successful repaired
checkpoint replay. Study admission must name the accepted pilot evidence.

## Artifact inspection

```sh
PYTHONPATH=src .venv/bin/python -m saal_rsi.adaptive_cli inspect \
  --job signed-job.json --public-key OWNER_PUBLIC_KEY_HEX
PYTHONPATH=src .venv/bin/python -m saal_rsi.adaptive_cli validate \
  --job signed-job.json --public-key OWNER_PUBLIC_KEY_HEX --bundle experiment/run
PYTHONPATH=src .venv/bin/python -m saal_rsi.adaptive_cli replay \
  --job signed-job.json --public-key OWNER_PUBLIC_KEY_HEX --bundle experiment/run \
  --cache /path/to/verified-model-cache --report new-controller-replay.json
```

The main learning/evaluation/promotion systems are preserved. The adaptive CLI
adds no promotion command. Upstream pretrained snapshots are separate from the
release ZIP; `configs/pretrained/registry.json` authenticates their revisions
and actual file hashes. Never substitute an LFS pointer for a tensor payload.

After a campaign ends, report its validated training and replay evidence with
`tools/adaptive_campaign_report.py research/dynamic-lora-v6/pilot-experiment new-report.json`. A completed
training run can still fail CPU/GPU replay or pilot admission; these outcomes
remain separate in the report. Neither outcome grants qualification.

The physical ZIP has its own signed inventory. Use its reported SHA-256 and
the controller public key supplied with the delivery:

```sh
.venv/bin/python tools/adaptive_capsule.py extract RELEASE.zip NEW_DIRECTORY ZIP_SHA256
.venv/bin/python tools/adaptive_capsule.py verify NEW_DIRECTORY CONTROLLER_PUBLIC_KEY_HEX
```
