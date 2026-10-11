# Weight training and independently admitted data

## Working part

The `saal train` command genuinely optimizes a four-class softmax classifier by SGD using a synthetic **train** partition. During each evaluation episode all weights are frozen. No gradients, LoRA adapters, or self-modifying code run in the actor process. The six-arm report isolates whether adding the offline-trained classifier helps relative to a frozen cue-based belief estimator; it is not evidence that training a frontier LLM makes it value-aligned.

## Optional local transformer LoRA SFT (separate, heavier install)

Install `python -m pip install -e '.[train-lora,attestation]'`. Validate your local checkpoint's model card, license, memory requirements, exact `target_modules`, tokenizer and load policy. *Do not* load remote code from the model repository. Training dataset JSONL objects require fields `prompt`, `ideal_response`, `source_partition="train"`, `admitted_by="independent_data_authority"`.

**Approval is not established by the JSON fields.** A trusted data-reviewer process must examine source rights, consent, privacy, correctness, conflicting values, and holdout exclusion, then sign an exact digest with its private Ed25519 key. Suggested local demonstration (all paths are examples, not a production trust separation):

```
python scripts/admit_training_data.py keygen --private-key /secure/admission-private.key --public-key /secure/admission-public.key
python scripts/admit_training_data.py sign --private-key /secure/admission-private.key \
  --data examples/qualified_training_example.jsonl --manifest artifacts/admission.json \
  --reviewer "independent reviewer" --rationale "example synthetic data only"
python scripts/admit_training_data.py verify --public-key /secure/admission-public.key \
  --data examples/qualified_training_example.jsonl --manifest artifacts/admission.json
python scripts/train_lora_optional.py --local-base-model /models/trusted-small-lm \
  --qualified-train-jsonl examples/qualified_training_example.jsonl \
  --admission-manifest artifacts/admission.json \
  --admission-public-key /secure/admission-public.key \
  --output-dir artifacts/lora-candidate --max-steps 50
```

The two provided samples demonstrate **format** only, not a large enough or independent dataset to produce a useful behavioral change. Training requires a locally cached compatible base model and likely a GPU. LoRA script is optional and has not been validated against every model family; architecture-specific projection names differ. Training outputs candidate weights; no promotion or deployment automation is included.

## Proposed beneficial-trait objective (not currently implemented)

For an outcome and behavior vector `r=[goal_match, honesty, calibration, corrigibility, externality, autonomy]`, use constrained qualification rather than a single unrestricted reward. Independent scorers should preserve evaluator disagreement and timing effects. Admitted training targets must never teach the model to falsify evidence or optimize grader artifacts. Never grant a learned reward model permissions to set its own scoring rules.

Investigate offline DPO/SFT first, then constrained RL on tasks with objective outcomes, then independent adversarial tests. Compare trained and frozen systems with identical tools and prompting; only claim added value after a preregistered positive out-of-distribution result. The optional script is plain SFT **not** beneficial trait RL.

## Data retention, privacy and reversibility

By default the prototype stores only synthetic data. Real user trajectories must be purpose-limited; redacted before admission, access-controlled, encrypted, retained on a written schedule and deletable where legally required. An append-only audit log and a right to deletion can conflict; preserve necessary audit commitments and erase the underlying sensitive payload through cryptographic or retention-layer mechanisms with appropriate governance.
