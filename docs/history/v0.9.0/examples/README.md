# Public negative example

The v0.8 example is produced by the final upgraded source and a locally anchored fresh holdout plan. It uses separate custodian/evaluator keys in one process controlled by one operator: it does **not** prove independent operator custody, an external timestamp, or sealed holdout confidentiality. Private plans, keys, and mutable registries are not shipped.

Portable cryptographic/source inspection:

```sh
python tools/inspect_example.py
```

That script separately reports whether the inspecting runtime matches. It does not certify a different host. Live verification on the evaluating runtime/source tree requires pinned keys:

```sh
saal-rsi verify --report examples/example_negative_qualification.json \
  --proposal examples/public_development_proposal.json \
  --public-key "$(cat examples/example_evaluator_public_key.txt)" \
  --anchor-public-key "$(cat examples/example_custodian_public_key.txt)"
```

Valid signatures are negative evidence; `qualification_passed` and `promotion_allowed` are false. Public demo keys are identifiers for this demo, not trusted production authorities. Earlier v0.7 examples are retained in `docs/history/examples/` and cannot satisfy v0.8 verification or activation.
