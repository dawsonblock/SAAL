# Reproduce v0.9.4.1

Use Python 3.12 on macOS ARM64 with the unchanged hash lock
`configs/dependency-locks/pretrained-macos-arm64-py312.txt`. The package retains
the separately tested Linux Python 3.13 T4 lock; do not apply it to another image.
The optional pretrained environment is needed for full tests and replay.

```sh
python -m pip install --require-hashes -r configs/dependency-locks/pretrained-macos-arm64-py312.txt
python -m pip install --no-deps -e .
python -m pytest tests
python tools/verify_corrective_history.py
python tools/verify_pretrained_history.py
```

Use the actual controller public pin published with the archive to verify the
new source/evidence capsule. Reading a public key from the same untrusted
download is not independent key authentication.

```sh
python tools/verify_corrective_release.py --pin YOUR_PINNED_CONTROLLER_PUBLIC_KEY
python tools/verify_manifest.py
python tools/verify_archive.py /absolute/path/to/SAAL-RSI-v0.9.4.1-Full-Corrective-Upgrade.zip
```

The last command extracts into a new directory, verifies the manifest, runs the
full suite in that extraction and verifies the manifest again. Its adjacent
verification receipt records the real post-build result.

`saal-repair public-dataset` and `screen` create public-only datasets and review
flags. `research-selection` consumes a JSON object containing `snapshots`, each
with step, checkpoint digest and development metrics. It never qualifies a model.
See `saal-repair --help` for the complete arguments.

Existing pinned model snapshots can be fetched with `saal-pretrained fetch` and
validated by the preserved registry. Heavy official base weights are not in the
ZIP. The optional 1.56 GiB full-tuning evidence remains separately distributed;
its original exact inner bundle and licensed wrapper hashes remain recorded.

Replay the new local sequence in a new process, using the public training
controller pin in `examples/corrective_research/controller.pub` and an already
authenticated model cache:

```sh
saal-repair sequence-replay --bundle examples/corrective_research/sequence-v2-smoke \
  --pin YOUR_PINNED_TRAINING_CONTROLLER_PUBLIC_KEY --cache /absolute/model-cache \
  --out /new/path/replay.json
```

New sequence training requires a newly signed local-FP32 v5 job, a separately
signed sequence-v2 protocol, positive ordinary-learning evidence, a new output
directory and a previously initialized one-use development budget. `init-development-budget`
does not reset an existing ledger. The released completed job is evidence, not
authorization to repeat its already consumed run. Create a fresh admitted job
and budget for reproduction. Do not reuse private keys from historical examples;
none are bundled.

Witness deployment templates deliberately contain invalid placeholders and no
keys. A separately authorized administrator enrolls the witness once:

```sh
python -m saal_rsi.attempt_service init-witness --config /private/witness.json --database /private/witness.db
python -m saal_rsi.attempt_service serve-witness --config /private/witness.json --database /private/witness.db
```

On the separately managed custodian host, configure a distinct mutual-TLS client,
the exact witness public and certificate pins, and the matching enrollment.
Initialize the mirror only against witness genesis, then run `serve-custodian`.
Certificate rotation/re-enrollment and recovery are audited administrative work,
not automatic client fallbacks. Do not use test certificates, colocate private
keys with training data, expose a Docker socket or claim independent custody
from a successful localhost demonstration. No protected evaluator is deployed by
these commands.

The original v0.9.4 capsule remains unchanged at `release-evidence/` and validates
against the original ZIP's extraction; metadata changes in this corrective tree
mean it is not the current release capsule. Historical qualification claims must
be read with their original source and stated limits.
