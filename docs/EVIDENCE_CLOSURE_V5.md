# Corrective source entry points

The v5 source repair retains the historical workers, jobs and failed models.
Use `saal_rsi.epoch_backend_v6` for new v6-classifier experiments and
`saal_rsi.epoch_artifacts_v3.validate_bundle(bundle, job, controller_pin)` for
strict artifact consistency checks. Do not treat verification as qualification.
`saal_rsi.behavior_screen_v2.error_audit` includes invalid-score pairs in its
failure list. `saal_rsi.family_gain_v2.family_gain` gives order-stable descriptive
public effects. The previous interfaces remain available for historical replay.

The research project is `research/evidence-closure-v5/`. Its corpus is generated
by `compositional_corpus_v5`, contains the complete unmodified public parent, and
remains unreviewed. The signed pilot contains four matched LoRA jobs at seeds 47
and 83 and one exploratory full-tuning job. Its controller source map, worker
identity, original reservations, budgets and selection rules must remain intact.
No automatic training retry or protected evaluation is admitted.

The original Colab run and two explicit provisioning continuations could not
obtain a T4. All session absences are confirmed. Results publication preserves
those failures and the five never-started RESERVED jobs. The final admission
prohibits further provisioning in that batch; resumption needs a new explicitly
signed provisioning admission that verifies the original ledger and exact jobs.
Never reinitialize the budget to erase failed or started attempts.

`frozen_challenge.py CACHE_ROOT` measures existing terminal checkpoints against
the new public diagnostics. `numerical_probe.py CACHE_ROOT DESTINATION` reproduces
the two retained numerical replay failures. Output files are created exclusively;
existing evidence is not overwritten. These commands require the complete
results artifacts and the exact verified upstream ModernBERT cache.

`structured_choice.plan(state)` is an exact synthetic planning control, and
`structured_choice.check_proposal(state, decision)` compares a raw model proposal
with it. Both require independently trusted structured facts. They do not parse
language, validate model-supplied permission claims, or grant execution authority.
Any future language-to-facts model needs its own field-level evaluation and
independent authentication before this control can use its extracted facts.

The full physical ZIP includes source, tests, public data, native historical
tensors, raw failures and release evidence. Verify its externally supplied
SHA-256, use the trusted `extract_capsule.py`, then run the trusted v5
`verify_capsule.py EXTRACTED_ROOT KNOWN_CONTROLLER_PIN`. The known controller pin
for this release is
`ccc792a3f49c372eaa3c66a9089e02e4350e19e63f4e677768561298caf32fc2`.
A signature verifies controller attribution and integrity; it does not create
independent data custody or prove worker/evaluator honesty.

Source, release reports, fresh Colab test evidence, local/extraction tests,
public learning diagnostics and protected qualification are separate records.
External activation remains prohibited. The source repair does not establish
generalization, production containment or recursive improvement.
