# Validation record

The source branch was tested in a temporary, isolated CPython 3.12.0 arm64 environment using the repository `configs/dependency-locks/pretrained-macos-arm64-py312.txt` with hash enforcement. The host torch distribution was 2.10.0; Transformers, PEFT, safetensors, NumPy, Hugging Face Hub and cryptography matched the lock.

- `PYTHONPATH=src python -m pytest tests -q`: passed (exit 0) in the pinned environment.
- `PYTHONPATH=src python -m pytest research/dynamic-lora-v6/tests -q`: passed (exit 0) after adding the exception-trace rejection case.
- Results inventory: every listed file size and SHA-256 verified; compressed telemetry files opened successfully; no checkpoint tensors, ZIPs, SQLite ledgers, or private-key files are in this results package.
- `git diff --check`: passed before publication.

The unpinned default Python environment had PEFT 0.20.0 and Transformers 4.57.6 and correctly rejected its own dependency identity; it was not used for the final suite. An initial attempt to install the arm64 lock in an x86_64 interpreter was rejected because the Torch wheel is architecture-specific. Neither environment mismatch changed the campaign or its artifacts.

This is software/regression validation only. It does not upgrade the campaign's assurance status: controller hardware preflight lacks an accepted structured receipt, the worker runtime is self-reported, all trained checkpoints failed the behavioral selection gate, and qualification remains `NOT_EVALUATED`.
