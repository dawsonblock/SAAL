# Dynamic Adapter Research v6

Target package: **0.9.5.dev6**, based on authenticated Evidence Closure v5.
This is a research implementation. Training, artifact validation, qualification,
and execution authorization remain separate operations. Fixed LoRA remains the
existing default. This path cannot authorize external deployment.

## Foundation and compatibility

`research/dynamic-lora-v6/FOUNDATION.json` records the clean extraction of the
final v5 physical ZIP. Its 2,615 logical files and 253 frozen historical files
verified, and its seven local test suites passed all 790 cases. There were no
outer LFS placeholders. The five original v5 jobs remain RESERVED; their jobs,
protocol, database, and exported state are protected by recorded hashes.

The historical two numerical replay failures remain failures at `1e-4`.
Historical full inference was not repeated for this foundation check. V6
starts from the official registered ModernBERT snapshot, not from a failed
historical checkpoint. The legacy worker code, contract decoding, signatures,
predictions, and negative findings are preserved. The frozen package
`__version__` symbol retains its historical value; distribution metadata and
the new adaptive CLI identify this development branch separately.

## Registered comparison

| Method | Allocated rank per layer | Target active aggregate rank |
| --- | ---: | ---: |
| Fixed LoRA-r8 | 8 | 176 |
| Fixed LoRA-r12 | 12 | 264 |
| AdaLoRA 12 to 8 | 12 | At most 176 after final allocation |

All methods adapt the same 22 native `model.layers.N.attn.Wqkv` projections.
The existing `head` **and** `classifier` modules are trained and saved for all
methods. The signed model specification describes an unadapted official
snapshot; a separate closed adapter contract admits the method, rank, targets,
loss, schedule, tensor format and budget. Old jobs cannot execute this path.

Default campaign configuration: official ModernBERT revision
`11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88`, native tokenizer, 256 tokens with
explicit rejection rather than truncation, FP32, AdamW, 1,000 successful
optimizer updates, microbatch 8 with accumulation 2, and 16,000 training
sample exposures. Head, sampler, and training dropout seeds are controlled.
Wrapping consumes different RNG draws, so the dropout stream is reset.

The mandatory T4 pilot uses seed **60606**, excluded from the five-seed study.
Study seeds are **47, 83, 109, 157, 211**. Admission of the 15 study runs
requires all three full pilot runs, tensor validation, fresh remote and local
checkpoint replay, matching initial model/head identity and sampling, and
confirmed cleanup. A two-seed parent-training control retains exactly the
same expanded public development population.

## AdaLoRA details that matter

The implementation pins PEFT **0.21.2**, including hashes of its AdaLoRA
configuration, layer, and model source. It invokes `update_and_allocate()`
once after a successful optimizer update while gradients are still present.
Unscale, finite checks and clipping precede the update. A skipped AMP step
or nonfinite value interrupts the run without advancing the rank schedule.

Updates 1–100 initialize; redistribution occurs through update 800;
updates 801–1,000 stabilize. The reference schedule allocates every ten
updates, with a forced final allocation at 800. Importance buffers are cleared
by PEFT at that boundary; subsequent telemetry retains the last allocation's
importance estimates rather than claiming newly measured importance.

**SAAL adds a disclosed persistent-mask policy.** PEFT's periodic pruning
allows zeroed singular values to regrow between allocation events. After
PEFT updates importance and allocation, SAAL reapplies the current mask.
Future allocation can reactivate components. This policy is signed as
`persist-after-completed-update-v1`; results must not be described as an
unmodified default PEFT experiment.

Orthogonal regularization is computed explicitly once, at weight 0.5. It does
not rely on the outer PEFT wrapper automatically including the tuner penalty.
Cross-entropy and regularization are reported separately. Telemetry records
completed optimizer steps, scheduled and last-applied budgets, each layer's
mask, nonzero singular count, importance values, adapter gradient norms,
global gradient checks, GPU memory and elapsed time. Hash chains detect edits
relative to the final authenticated inventory; they do not attest an honest host.

## Checkpoints and evaluation

Initial, best-development, and terminal snapshots are retained separately.
Full allocated A/B/E tensors and immutable `ranknum` values are stored in
Safetensors. They are not compressed through stock PEFT export. Masks are
separate versioned artifacts, bound to the matching history step. Restoring
does not resize tensors or silently change scaling. Unknown tensor names,
missing singular values, incorrect shapes/dtypes, nonfinite data, altered
mask patterns, or same-step substitution fail validation.

The new worker receives **training and development records only**. Public
regression is evaluated only after candidate freeze. No protected examples,
signing secrets, or promotion credentials are uploaded. The research selector
uses public invalid/critical errors, class coverage, complete pairs and family
metrics; it never performs qualification. Initial and terminal results remain
visible even when the best-development checkpoint is step zero.

The public comparison requires all five matched seeds. Its primary endpoint
is terminal family-macro complete-pair gain over fixed-r8. Seed/family paired
bootstrap intervals are descriptive for this correlated public corpus.
Thresholds are +5 percentage points, interval lower bound above zero,
nondecreasing worst-family accuracy, and no increased critical proposals.
Retention requires a separate sequential-learning protocol and a maximum
two-point prior-task loss. Public screening does not satisfy that requirement.

AdaLoRA also changes SVD parameterization, regularization and starting capacity.
A positive primary comparison alone cannot attribute a gain to rank allocation.
No-allocation and random-allocation controls require subsequent separate
admissions. Retention, independently authored tasks and protected qualification
are further gates, not conclusions from successful training.

## Compute and remaining trust limits

Colab runs use an exact named T4 session, Python-3.13 hash-locked installation,
separate controller `nvidia-smi` observation, a bounded forward/backward probe,
streamed progress, durable attempts, artifact recovery and a watchdog. Only
immutable transport reads may retry. Provisioning and training never retry
automatically or change batch, precision, rank, revision or parent weights.
Termination requires an actual exact-session absence response. An uncertain
shutdown remains an incident; network failure does not establish cleanup.

These are trusted, closed worker programs in separate processes and a dedicated
provider session. They are **not a hostile-code sandbox** or GPU attestation.
Worker measurements and controller commands on the same provider host do not
establish operationally independent custody. Local SQLite development quotas
are not administrator rollback resistance or a financial billing cap.

The public corpus still has shared generators/oracles and zero human reviews.
The repository cannot manufacture independently administered services or
independently reviewed final tasks. Protected sampling, external attempt
anchors, independent credentials and qualification evidence remain required.
No transfer, retention advantage, independent generalization, recursive
self-improvement or production alignment is established by this release.

## Physical release verification

`tools/adaptive_capsule.py` authenticates the v5 foundation, retains its signed
delivery metadata under provenance, packages actual LFS payload bytes, and
binds source and evidence commits to a new controller-signed inventory. After
clean extraction it verifies every file before restoring deduplicated tensor
paths. The pinned official upstream model cache remains a separate download.

`tools/adaptive_qa_result.py` requires all eight suites, source/input identity,
consistent JUnit cases and confirmed runtime teardown. Missing adapter results
cannot be reported as a successful historical seven-suite run.

The pilot gate resolves artifacts inside the extracted package. Absolute
controller paths are provenance and are never used to borrow files from the
original checkout. The exact admission-time source is retained in
`pilot-experiment/preregistered-source`; subsequent controller reporting and
packaging fixes do not alter the submitted worker, data, optimizer, rank
schedule, numerical tolerances or acceptance thresholds.

The first T4 admission exposed a cross-platform initialization difference:
the classifier weight differed by at most `1.49e-8` between ARM and x86.
Requiring a freshly randomized head to hash identically was incorrect replay
reconstruction. Controller replay now restores the exact archived initial head
before hashing, then checks the unchanged model/frozen hashes and `1e-4`
prediction tolerance. Its restored predictions passed for all three snapshots.
The failed original admission remains failed. A separate signed
`pilot-repair-02` admission references that failure and repaired replay; it
does not reset or reuse the original quota. The training worker is unchanged.

Reference: [official PEFT AdaLoRA documentation](https://huggingface.co/docs/peft/en/package_reference/adalora).
