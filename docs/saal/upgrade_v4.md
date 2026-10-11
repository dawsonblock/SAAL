# v0.4.0: evaluator private/public-key separation and frozen study commitments

## New trust boundary

In v0.3, the evaluator and independent qualifier both needed the evaluator's HMAC secret. A compromised qualifier could therefore forge evaluator signatures. v0.4 provides an Ed25519 evaluator-specific keypair. **Only the evaluator holds the private key.** The qualifier holds only the public key and checks its SHA-256 fingerprint against a preregistered trust anchor. The actor/trainer must receive neither the private key nor the sealed evaluation seeds. This is an authentication improvement, not proof of outcome validity.

**Breaking v4 study gate:** Evaluations whose preregistration sets `signature_scheme=ed25519` must be signed with Ed25519. The qualifier must have the pinned public key, the complete private preregistration, an external preregistration commitment and the identical candidate model. It recomputes all arms independently. Downgrading an Ed25519 study to HMAC or disabling replay rejects qualification. The v3 HMAC path is retained **only for legacy research reproducibility**; do not use it for new studies.

### Distinct hosts / users

Evaluator host (private study contents, private signing key, independently controlled model intake):

```sh
python -m saal evaluator-keygen --private /private/evaluator.ed25519 --public /public/evaluator.pub
python -m saal prereg-create --seeds 12 --episodes 100 \
    --evaluator-public-key /public/evaluator.pub --out /private/prereg.json
```

**Before model evaluation**, publish the `prereg_digest` receipt to a timestamped external append-only journal. Merely copying the hash into the qualifier request does not demonstrate its earlier existence. Store the exact public key fingerprint securely through an independently controlled provenance channel.

After candidate weights are fixed and transferred in:

```sh
python -m saal evaluate --model /incoming/intent-model.json \
    --prereg /private/prereg.json \
    --eval-private-key /private/evaluator.ed25519 \
    --out /private/evaluation.json
```

Qualifier host (trusted source release, no evaluator private key):

```sh
python -m saal qualify --report /incoming/evaluation.json \
    --prereg /private/prereg.json --model /incoming/intent-model.json \
    --eval-public-key /public/evaluator.pub \
    --expected-prereg-digest <EXTERNALLY_TIMESTAMPED_SHA256> \
    --out /private/qualification.json
```

Replace `<EXTERNALLY_TIMESTAMPED_SHA256>` with the real external commitment. Qualification may legitimately return **REJECT** when the trained candidate has not statistically surpassed the frozen baseline; rejection is expected, not a tooling failure.

## Changes in this release

- Ed25519 `sign_evaluation`, `verify_evaluation`, pinned `EvaluatorPublicKey`, domain separated statement covering report, model, prereg, implementation and partition.
- `implementation_digest` hashes each shipped SAAL Python module; the qualifier compares against its independently installed source tree and recomputes signed outcomes. The digest is **not** a certified runtime image measurement or secure supply-chain proof.
- CLI `evaluator-keygen`, `evaluate --eval-private-key`, `qualify --eval-public-key --expected-prereg-digest`, and `prereg-create --evaluator-public-key`.
- Public-key reading uses `O_NOFOLLOW` and a file-descriptor regular-file check on POSIX platforms. On unsupported platforms, fails closed.
- Authority socket sets a bounded read timeout for incomplete requests. The research server remains a toy, single-process server; it is not production DoS protection.
- Release verifier now detects unlisted files, not just incorrect checksums.
- All v0.1–v0.3 synthetic experiments, budgets, single-use capabilities, signed staging and adversarial tests remain in place.

## Remaining blockers

Independent credential custody, OS identity separation, actual trusted executor, safe runtime image digest, OS-level resource control, secure remote authentication, signed/recorded external preregistration witnessing, independent evidence collection, cross-domain evaluation, live model learning experiments, and human-subject outcome methodology remain unsolved here. The v4 CLI is not a safe deployment authority.
