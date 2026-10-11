# SAAL-RSI v0.9.4.2 validation repair report

Status: NOT_EVALUATED. Independent generalization NOT_ESTABLISHED. External deployment prohibited.

## Reproduced defects and fixes

1. Sequence replay previously accepted identical probability vectors whose entries summed to four, and NaN tolerance allowed changed predictions to compare equal. None/string probability vectors could raise incidental errors. Replay now checks bounded finite normalized vectors, allowlisted argmax-consistent decisions and finite tolerances no greater than 1e-4. Malformed evidence returns disagreement.
2. Public research selection accepted absent complete_pair_accuracy during validation, then crashed with KeyError during ranking. It also compared measured pairs against unmeasured pairs as though both were measured. Missing fields and mixed coverage now raise explicit validation errors; consistent null coverage remains supported.
3. Evidence-chain receipt checks used a fixed 0.025 alpha ceiling for every attempt. They now enforce the maximum 0.05 / 2**attempt and at least attempt+1 events before consumption, rejecting an inflated late-attempt allowance or impossible counter. Smaller enrolled total-alpha allocations remain valid. These checks do not reconstruct witness history or establish operational custody independence.

The new regression tests exercise malformed vectors, inconsistent labels, NaN tolerance, legitimate numerical agreement, missing/mixed pair evidence, inflated alpha and impossible event counters, including a valid later-attempt receipt that issues no runtime capability.

## Preservation and scope

The original v0.9.4.1 directory and delivered ZIP are unchanged. Every inherited file digest is recorded in provenance/v0.9.4.2-historical-input.json; exact changes are enumerated in corrective-evidence/validation-repair-changes.json. Historical release statements/reports are preserved separately. Current source/evidence receives a new controller signature; old signatures are not presented as validation of changed source.

Historical neural training workers, signed model jobs, checkpoints, predictions, previous negative findings and Colab cleanup evidence remain unchanged. No new training, GPU provisioning or protected evaluation is performed. The recorded LoRA 0/36 complete counterfactual pairs remains a failure; this release makes no improved learning claim.

The 0.9.4 pretrained worker identity and sequence-v2 training identity are unchanged. The repaired replay verifier belongs to the controller evidence path. Qualification services still require independent administration, defensible sampling and protected evaluation before empirical qualification. This patch does not make colocated keys or controller-operated services independent.

## Verification

Results are recorded below after execution. Platform hash locks remain inherited; this patch does not claim a newly tested Linux/T4 installation.

- Full source suite: **492 passed, zero failures/skips**, 94.59 seconds, macOS ARM64/Python 3.12.
- 28 new negative/positive regression cases; 73 targeted integration cases passed.
- Fresh-process existing ModernBERT sequence checkpoint replay: complete per-record agreement within 1e-4. No new training.
- Prior v0.9.4 history verifier: 2,012 files unchanged; five enumerated metadata changes.
- The delivered ZIP is checked separately after clean extraction; its adjacent verification report records the actual result.
