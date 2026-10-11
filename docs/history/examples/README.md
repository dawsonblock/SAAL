# Public demonstration artifacts

These public examples are NOT sealed or independently timestamped benchmarks and
have **no promotion authority**. The private evaluator holdout plan and Ed25519
private signing key were intentionally not packaged.

The report's signature can be inspected using the provided public key:

```bash
saal-rsi verify --report examples/example_negative_qualification.json \
  --proposal examples/public_development_proposal.json \
  --public-key "$(cat examples/example_evaluator_public_key.txt)"
```

`verified: true, qualification_passed: false` is expected. Valid signature
means the bytes were signed by the pinned demo key, not that improvement passed.
