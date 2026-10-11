# Verified repairs in v0.8.1

This patch preserves the bounded candidate language, replay-based selection, synthetic behavior gates, fresh holdout ledger, and fail-closed promotion behavior.

- Public keys are compared as decoded bytes. Hexadecimal letter case cannot create a second custodian, evaluator, or human identity. Registry pins use canonical lowercase encoding.
- Anchor receipt schema v3 signs the public evaluation rules, thresholds, cluster count, and configuration alongside the sealed plan digest. Verification checks the pinned custodian identity, fixed study family, scheduled alpha, attempt limits, and receipt structure.
- Qualification report schema v3 verifies the baseline identity and human-approval requirement. It recomputes paired means, the deterministic exploratory bootstrap bound, and gate conclusions against anchored rules. It rejects inconsistent summary counts and checks behavior conclusions against recorded proposals and control denial/process conclusions against their records.
- Actor responses have a closed, typed proposal schema. Unknown decisions and malformed replies fail evaluation; an unknown decision cannot earn appropriate-refusal credit.
- Worker deadlines cover nonblocking writes and reads. Oversized, unsolicited, malformed, and wrong-PID replies fail closed and terminate the worker. Negative tests use actual stalled and forged-reply child processes.
- CLI reads reject symlinks and nonregular files and bound input size. Output creation is exclusive and reserved before anchoring or consuming a holdout, so an existing output file cannot waste an attempt. Interrupted writes remove only the reserved inode. This is an exclusive reservation with a final flush, not atomic publication; observers may see the reserved empty or partial file while work runs.
- New SQLite stores are private regular files; symlinks, FIFOs, foreign ownership, group/other permissions, and hard links are rejected at opening. SQLite subsequently reopens by pathname: this preflight does not defeat a same-UID parent-directory replacement race. Use independently protected custody and directories.

The 29 additional regression cases include re-signed inconsistent reports, invalid receipts, decoded-key impersonation, malformed actor decisions, real-child IPC deadlines, and negative filesystem/CLI cases. A positive approval unit fixture has deliberately constructed coherent statistics; it is not measured qualification evidence.

## Migration and limits

v0.8.0 receipts/reports remain historical signature artifacts under docs/history/v0.8.0. Current operational verification requires schema v3 and matching current source/runtime identity. Create a new proposal and plan under v0.8.1; never edit an old signed receipt or reset a study ledger to regain alpha. Existing registry history is preserved, but consumed old attempts cannot be reused. No automatic evidence migration or inferred deployment grant is provided.

Consistency verification is not independent reconstruction of holdout outcomes: the signed evaluator remains trusted to report truthful measurements. Capability challenge booleans require that trust; arbitrary fabricated but internally consistent evidence cannot be distinguished cryptographically. The public demo is negative evidence and uses public seeds.

Same-UID child processes are not an OS sandbox. Custodian anchoring is local, not a remote service or trusted timestamp. A privileged or same-UID operator can modify keys, source, or reset stores. Bootstrap bounds are exploratory and alpha bookkeeping does not establish nominal statistical coverage. No real language-model learning, alignment guarantee, real-world side effects, external activation, or production qualification is claimed.
