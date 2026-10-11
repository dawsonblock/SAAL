# SAAL-RSI v0.9.3 — Robust Neural Learning

A position-aware, four-class supervised research model, matched frozen/full/LoRA experiments, versioned artifacts and monitored Google Colab execution. This release preserves SAAL, Dream-RSI and the inherited authority system. It is not an LLM, an autonomous recursive improvement system or a production-qualified alignment result.

The new `tiny-transformer-lora-v2` uses UTF-8 byte embeddings, learned positions, a classification token, two 64-dimensional Transformer layers and explicit query/value LoRA projections. The default input bound is 256 bytes; longer or empty prompts are rejected without truncation. The historical 128-byte worker and archived negative evidence remain unchanged.

Training results are **NOT_EVALUATED**. Public development scores, successful tests and valid signatures never authorize deployment. The existing protected-family gates, one-use custody receipts and alpha-spending rules remain in place. No operationally independent final population is available in this release.

[Release report](RELEASE_REPORT.md) records actual measurements and the state of each gate. [Implementation and migration](docs/V0.9.3_ROBUST_NEURAL_LEARNING.md) describes contracts, experiment controls, reproduction and remaining limitations. Historical v0.9.2 failures are retained under `docs/history/v0.9.2/`.

## Run public research

Use the packaged platform-specific hash lock. macOS ARM64 requires Python 3.12; the Colab GPU lock requires Linux x86_64 and Python 3.13. These locks pin Torch 2.10.0, Safetensors 0.8.0 and NumPy 2.4.6. Colab CLI authentication is performed locally; controller keys and final-evaluation data must remain local.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r configs/dependency-locks/macos-arm64-py312.txt
export PYTHONPATH="$PWD/src"
.venv/bin/python -m saal_rsi.model_cli neural-data --out public-dataset.json
.venv/bin/python -m saal_rsi.cli keygen --help
.venv/bin/python -m saal_rsi.model_cli neural-campaign --dataset public-dataset.json --controller-key /absolute/private/controller.key --workspace /absolute/new/campaign --backend local
```

Use `--backend colab` to execute the same bounded campaign in one fresh named T4 session. All per-run jobs are signed locally; frozen baseline bytes are validated before signing adaptation jobs. A private local budget ledger reserves each run once. The GPU installer creates a fresh environment and enforces the packaged wheel-hash lock. Native monitoring streams metrics, enforces deadlines, runs a separate cleanup watchdog, and requires a named-session absence response before reporting cleanup complete. This ledger is not a financial billing cap or a rollback-proof service.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
.venv/bin/python tools/verify_manifest.py
.venv/bin/python tools/inspect_neural_release.py
```

The full source tree is required for source-bound evidence checks. Archived reports refer to their original versions and must not be interpreted as current-source attestations.
