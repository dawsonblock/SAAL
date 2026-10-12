# Reproduce this public exploratory experiment

This supplement contains unmodified v0.9.4.3 source/configurations plus the new corpus, frozen signed jobs and recovered results. `SOURCE_ORIGIN.json` binds the native implementation to main commit b145c09c16aa0ca10901c2a1e2a5382d3e95913a. The complete historical release remains separate; this supplement does not bundle upstream pretrained weights.

Set `PYTHONPATH=src`. Install the platform-specific hash-locked dependencies from `configs/dependency-locks/` using the matching Python and architecture, with `--require-hashes`. Do not apply the Linux Python 3.13 lock to another platform. The recorded Mac CPU verification used Python 3.12 ARM64; GPU training used Linux Python 3.13 and the native locked installer. Google Colab CLI authentication and an admitted T4 runtime are required for new cloud training.

Fetch the official pinned model into a chosen cache:

```sh
PYTHONPATH=src python -m saal_rsi.pretrained_cli fetch \
  --repository answerdotai/ModernBERT-base \
  --revision 11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88 \
  --cache /path/to/model-cache
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python tools/report_results.py
PYTHONPATH=src python tools/replay_results.py 0 /path/to/model-cache
```

Replay indexes 0–4 correspond to the five preregistered jobs. The replay validates all three checkpoint inventories and reconstructs initial, selected and terminal predictions on both public splits. It verifies probability differences against the recorded GPU outputs at tolerance 1e-4 and preserves failure evidence. The recovered candidate and source must remain exactly as signed. The bundled controller key authenticates self-consistency; establish its identity separately for external trust.

A new training experiment needs a new signed job, resource-budget identity, preregistration and output directory. Completed attempts and the original ledger must not be reset/reused to disguise additional attempts. The corpus generator and policy are available for review; rendering a state does not establish that the natural-language labels are independently correct. A public-data run consumes no protected evaluation alpha and cannot authorize promotion.

See REPORT.md for measured results, tests, cleanup and known limits. No human reviews or independent final evaluation are claimed.
