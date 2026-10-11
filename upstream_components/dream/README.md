# Dream-RSI Governed Reference Build v1.0.0

This archive upgrades the paper-only Dream-RSI repository into a runnable **independent reference implementation** of the replay/control-plane ideas. It is not the unreleased official Dream-RSI code and does not claim experimental reproduction of the paper.

The original user-supplied repository is preserved byte-for-byte under `reference/upstream/`.

## What this build adds

- stochastic multi-outcome replay rather than one frozen child per action;
- prefix-only policy context and strict parallelism/budget guards;
- development / selection / sealed-holdout dataset separation;
- explicit support and uncertainty penalties;
- adaptive risk-aware UCB reference policy plus fixed baseline;
- independent qualification gates against the incumbent baseline;
- Ed25519-signed promotion manifests;
- SHA-256 content-addressed artifact store with strict digest parser;
- append-only hash-chained SQLite audit ledger;
- static policy-source safety inspection;
- live-canary comparison helper;
- deterministic end-to-end demo and pytest suite;
- architecture, threat-model and validation documentation.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest

dream-rsi demo --worlds 30 --seed 7 --beta 1.0
dream-rsi demo --worlds 30 --seed 7 --beta 1.0 --promote --out run-artifacts
```

## Authority model

```text
Discovery / evidence
        |
        v
Policy proposer  --NO PROMOTION AUTHORITY--
        |
        v
Replay evaluation
        |
        v
Independent qualification on sealed holdout
        |
        v
Fresh live canary (deployment-specific)
        |
        v
Independent signer / promotion authority
        |
        v
Signed runtime manifest
```

The design refuses the assumption that a policy which scores well on the histories used to create it has proven live improvement. Replay is evidence, not authority.

## Important limitation

The upstream repository has no explicit software license in the supplied archive. `reference/upstream/` is included only because it was supplied by the user. Do not assume rights to redistribute or commercially reuse upstream material. The new reference implementation in `src/`, `tests/`, `docs/`, and `examples/` was generated independently for this build.
