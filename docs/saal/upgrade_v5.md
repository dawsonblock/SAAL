# SAAL v0.5.0 — independent witness and training-data security

## Summary

This is a source-complete successor to v0.4.0, including its prior tests, six-arm synthetic studies, gated synthetic capabilities, offline model training and candidate staging-only release path.

**v5 change:** a signed evaluator report is no longer the only independently authenticatable item in the new evaluation workflow. An Ed25519 **witness authority**, which must be separate from the evaluator, signs the preregistration commitment and the installed implementation digest. A v5 evaluation binds the full witness receipt digest, and a v5 qualifier fails closed if the witness receipt or independently pinned witness public key is absent, altered, swapped, or mismatched. The qualifier independently replays the same synthetic study.

The witness should be an independent organization, service, host, or operator with exclusive custody of its private key. Merely running all commands on one machine under the same user **does not** create real separation of powers.

## Full CLI demonstration

Install from this checkout:

```bash
python -m pip install -e '.[attestation]'
python -m unittest discover -s tests -q
python scripts/verify_release.py
```

**Independent witness machine**, holding the witness private key; evaluator and trainer must never possess it:

```bash
saal witness-keygen --private /witness/witness.priv --public /witness/witness.pub
```

**Evaluator machine**, before candidate training and evaluation:

```bash
saal evaluator-keygen --private /evaluator/evaluator.priv --public /evaluator/evaluator.pub
saal prereg-create --evaluator-public-key /evaluator/evaluator.pub \
  --witness-public-key /public/witness.pub --seeds 12 --episodes 100 \
  --out /evaluator/private-prereg.json
```

Send the preregistration to the witness over a trusted channel. It includes private test seeds: do not expose it to a trainer or acting model.

**Witness machine** signs and externally anchors the receipt *before evaluation*:

```bash
saal prereg-witness --prereg /witness/private-prereg.json \
  --witness-private-key /witness/witness.priv \
  --witness-public-key /witness/witness.pub \
  --witness-identity independent-study-notary --out /witness/witness-receipt.json
```

**Independent archive:** publish the receipt or its digest to a separately controlled timestamped append-only store. Signed `issued_at_ns` is an assertion by the witness, NOT independently verified chronology; the script does not implement a public timestamping/notary service. A private database under evaluator control is not sufficient.

**Evaluator machine** accepts immutable candidate weights and conducts study:

```bash
saal evaluate --model /incoming/intent-model.json \
  --prereg /evaluator/private-prereg.json \
  --eval-private-key /evaluator/evaluator.priv \
  --witness-receipt /evaluator/witness-receipt.json \
  --witness-public-key /public/witness.pub \
  --out /evaluator/private-report.json
```

**Independent qualifier** verifies a separately received witness receipt, an *independently trusted* witness public key, a frozen preregistration digest from the out-of-band archive, the evaluator public key, the exact candidate, the source digest, and full deterministic replay:

```bash
saal qualify --report /incoming/private-report.json \
  --prereg /qualifier/private-prereg.json \
  --eval-public-key /public/evaluator.pub \
  --witness-receipt /qualifier/witness-receipt.json \
  --witness-public-key /public/witness.pub \
  --expected-prereg-digest <EXTERNALLY_ANCHORED_PREREG_SHA256> \
  --model /incoming/intent-model.json --out /qualifier/decision.json
```

An evaluation rejection is entirely normal. This code **does not** automatically promote a model on qualification success; qualification is only a recommendation for further review.

## New explicit fail-closed conditions

- Reject missing or unauthenticated preregistration witness for new v5 studies.
- Reject witness and evaluator key reuse; keys are pinned by distinct public fingerprints in the study.
- Reject a report that omits or alters the witnessed receipt digest, even when re-signed by a genuine evaluator.
- Reject mismatched study ID, preregistration fields, source digest, witness identity or signature contents.
- Reject altered evaluator reports, downgraded evaluation signature purposes and mismatched full deterministic replay.
- Reject extra fields and duplicate JSON keys in independently admitted training JSONL; refuse symbolic links and oversized training files, validate the same byte snapshot that is hashed, pin the data authority verification key and require a domain-specific signature.

## What v0.5 does NOT establish

1. A witness signing receipt does not by itself establish historical order of receipt creation relative to model development; that needs an independently governed append-only log with a trustworthy timestamp.
2. A verified signature authenticates an authorized signer, not the truth of a human welfare outcome.
3. The code digest covers SAAL Python module bytes, not wheel provenance, binary dependencies, kernel, interpreter, microcode, or host/hardware attestation.
4. The signer and evaluator are distinct *roles in code*, not different physical machines or principals unless deployed that way.
5. The synthetic simulator is not an LLM alignment benchmark, a general corrigibility proof, or a safe production agent. There are no live financial, medical, robotic, scheduling, or messaging effectors.
6. Existing v4 study verification remains for historical reproducibility. Organizations should establish an external policy that disallows legacy studies for new promotion decisions. The new v5 witness requirement is enforced whenever `witness_scheme=ed25519`.
7. Offline LoRA remains optional and requires additional large-model-specific evaluations and independent admission oversight.

## Suggested follow-up release

Prioritize independent OS identities and a dedicated least-privilege executor on a separate host or VM; externally anchored study commitments; safe dependency/runtime supply-chain measurement; authenticated human approval; verifiable outcome evidence from actual bounded services; blinded human study methodology. Do not add unrestricted online neural weight updates.
