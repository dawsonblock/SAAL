# SAAL v0.2.0 — Independent candidate qualification and staged release

## Separate authority domains

- **Untrusted actor / trainer:** submits candidate artifacts, never holds verifier, qualifier, or human keys.
- **Independent evaluator:** holds the private/secret evaluator signing key, sealed test generator, oracle and trusted measurements. The included public test seeds are for demonstration and are NOT a true sealed holdout.
- **Independent qualifier:** checks test provenance, evaluator signature, preregistration hash, statistical and regression gates, then attests an `ELIGIBLE_FOR_HUMAN_REVIEW` verdict using its **Ed25519 private key**.
- **Authorized human:** views the evidence and signs a specific candidate file plus qualification/preregistration commitments with a different **Ed25519 private key**.
- **Staging host:** verifies both public signatures, rereads and hashes candidate bytes, then writes exclusively created immutable candidate files. It does **not** activate them.

Key generation for each separate trusted signer (in separate protected environments):

```bash
python -c "from saal.admission import keygen; keygen('/secure/qualifier.secret','/secure/qualifier.pub')"
python -c "from saal.admission import keygen; keygen('/secure/human.secret','/secure/human.pub')"
saal init-key --out /secure/evaluator.key
```

Do not run all three on one machine in production. No secret should enter the ZIP, source repository, training host, or actor environment.

## Evaluator run on a trusted host

```bash
saal evaluate --model /exports/model.json --prereg configs/preregistered_eval.json \
  --eval-key-file /secure/evaluator.key --out /secure/evaluation.json
```

The preregistration and evaluation environment should be independently frozen **before** candidate exploration, with an off-host timestamped digest. Hash-binding alone does not prove timing.

## Qualifier step

```bash
saal qualify --report /secure/evaluation.json --prereg configs/preregistered_eval.json \
  --eval-key-file /secure/evaluator.key --out /secure/qualification.json \
  --qualification-key-file /secure/qualifier.secret \
  --attestation-out /secure/qualification-attestation.json \
  --reviewer independent-reviewer
```

This command fails without a verified result eligible for human review. The example classifier may fail the improvement gate; that is a valid research outcome. **Do not weaken thresholds to force promotion.**

## Human sign-off, if and only if qualified

```bash
saal candidate-authorize \
  --qualification-attestation /secure/qualification-attestation.json \
  --qualifier-public-key /secure/qualifier.pub \
  --human-private-key /secure/human.secret \
  --model /exports/model.json --prereg configs/preregistered_eval.json \
  --approver responsible-human \
  --rationale 'Reviewed sealed results and staged for later technical inspection' \
  --out /secure/authorization.json
```

## Stage without deployment

```bash
saal candidate-stage \
  --qualification-attestation /secure/qualification-attestation.json \
  --qualifier-public-key /secure/qualifier.pub \
  --human-public-key /secure/human.pub \
  --candidate-authorization /secure/authorization.json \
  --model /exports/model.json --prereg configs/preregistered_eval.json \
  --stage-root /secure/staging
```

This writes a digest-derived `candidate.json` and `release_record.json`. It never sets `current`, executes weights, or grants the model any additional privilege. Staging is a recommendation for separate future deployment governance, not activation.

## Explicit limitations

The qualifier signature proves only a key-holder signed a verdict, not that the verdict or model behavior is correct. The evaluator uses HMAC shared with the qualifier; the deployment needs independent public-key evaluators and strong identity management. Filesystem safety assumes trusted administrative ownership; code-only separation within one process or UID is not a security boundary. A real large-model release needs the **complete closure** of weights, tokenizer, config, training code, dependencies, evaluations, dataset identifiers, authorization records, and image hashes. This prototype binds only its tiny JSON model and the qualification/preregistration evidence.
