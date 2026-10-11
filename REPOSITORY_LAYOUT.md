# SAAL repository branches

`main` preserves the complete SAAL-RSI v0.9.4.2 Validation Repairs source release, including its inherited examples, checkpoints, negative evidence and provenance. Its original MANIFEST.sha256 remains unchanged. The additive `reviewed-decision-corpus/` directory is a 600-record,300-pair review candidate with its own source,tests,manifest and review workflow. It has zero completed human reviews and is not training-ready.

`results/saal-rsi-20261011` adds published experiment and Colab validation evidence under `results/`. Results are research evidence, not model qualification or external execution authorization. Current corpus validation passed38tests on local CPU and Colab CPU. The prior full Colab suite and GPU probe are distinct from actual pretrained GPU training.

Clone with Git LFS installed and run `git lfs pull` to materialize historical archives/checkpoints. Root release verification needs the full materialized files. Do not interpret LFS pointer files as valid checkpoint bytes.

Use the pinned runtime documented in docs/CORRECTIVE_REPRODUCTION.md for the inherited tests. Corpus integration tests: `PYTHONPATH=reviewed-decision-corpus/src:src python -m pytest reviewed-decision-corpus/tests -q`. Corpus verification: `python reviewed-decision-corpus/tools/verify_capsule.py`.

Independent qualification remains NOT_EVALUATED; external deployment remains prohibited. This publication does not generate human approvals, new training results or fresh qualification signatures.
