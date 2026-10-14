# Numerical contract repair — engineering only

This branch adds a new numerical contract and an unsigned campaign review
draft. It does not authorize, reserve or execute another research campaign.
The completed dev6 evidence archive and its source/results branches remain
the historical release. No previously consumed admission is amended.

## Failure and repair

The fixed-r12 seed157 study attempt trained 1,000 worker-reported updates, but
its bundle was rejected by the admitted controller. The stored Brier score was
`0.0881157183233161`; local reconstruction was `0.08811571832331609`, one binary64
unit apart. Physical inventory checks and recorded GPU prediction replay were
consistent. The exact operation responsible for the platform-dependent last
bit has not been established. This does not establish checkpoint corruption
or behavioral qualification.

The old v1/v2 validators still reject that bundle. Its state is `INTERRUPTED`;
the study budget stays `FAILED`, with ten `COMPLETED`, one `INTERRUPTED` and four
`RESERVED` attempts. No missing result is imputed. All previously recorded
behavioral failures remain failures.

New jobs use `saal-adaptive-job-v3`, a different worker identity and a signed
`numerical_metric_policy` plus its digest. New result, inventory and replay
schemas distinguish these artifacts from historical ones. Both directions of
legacy/new campaign substitution are rejected before attempt reservation.

## Numerical policy

`saal-numerical-metric-policy-v1` specifies the following arithmetic:

* Input probabilities are their exact binary64 values, in the registered
  CHEAP/FAST/ASK/REFUSE order. Values are never renormalized. The sum must differ
  from one by strictly less than exact `0.0001`; ties use the first class.
* Counts, confusion entries and labels remain exact. Accuracy ratios and family
  means use exact rational arithmetic followed by one binary64 rounding.
* Brier score and ten-bin calibration error use exact rational aggregation.
  Negative log likelihood uses an explicitly private 80-digit, half-even Decimal
  context and an exact decimal `0.000000000001` probability floor, followed by
  exact aggregation of the computed losses.
* Brier, calibration error, NLL and per-class NLL are canonical decimal strings
  with eighteen fractional digits and exact half-even final rounding. The
  metrics include `saal-classification-diagnostics-v1` as a schema discriminator.
* Validation still compares canonical metrics exactly. There is no broad
  floating-point tolerance. Malformed inputs, changed digits, float substitutions,
  missing fields and invalid policy identities fail validation.

The interrupted predictions have a new-policy diagnostic Brier string of
`"0.088115718323316100"`. That is a diagnostic reconstruction, not conversion of
the historical attempt into an admitted v3 result.

The new exact family-macro mean can differ by a final rounding bit from a legacy
sum of rounded family accuracies. Checkpoint tie selection can consequently
differ in a future experiment. This change is explicitly part of the new policy;
no old checkpoint selection is rewritten. Numerical acceptance at a probability
normalization boundary can also differ because the new policy tests the exact
sum rather than a rounded sum. These differences require new admission.

Selection criteria, behavioral thresholds, frozen tensor checks, rank budgets,
completed-update accounting and prediction replay tolerance (`1e-4`) retain their
registered meanings. The decimal-string diagnostics do not authorize execution
and are not used as qualification thresholds.

The arithmetic uses documented Python behavior: [exact rational construction](https://docs.python.org/3.12/library/fractions.html)
and [Decimal logarithm and rounding](https://docs.python.org/3.12/library/decimal.html#decimal.Decimal.ln).
Portability of the implementation is tested locally; a fresh Linux/T4 validation
of this new worker remains a separate engineering gate.

## Reviewable next campaign

`CAMPAIGN_DRAFT.json` is a deterministic review document, not a signed protocol.
It contains no signature, reservation or executable admission. It binds the new
worker/policy/source identities and proposes separately budgeted stages:

1. Three complete engineering pilots using excluded seed60606.
2. Fifteen matched runs: fixed-r8, fixed-r12 and AdaLoRA12-to-8 across seeds47,
   83,109,157,211, after the pilot and cleanup pass.
3. Four exploratory parent-data controls across seeds47/83, after every study
   run and cleanup pass and the separate parent-data budget is authorized.

Each proposed run preserves 1,000 completed updates, 16,000 sample exposures,
initial/best-development/terminal checkpoints, no automatic retries and no
resume. Equal updates/exposures do not imply equal compute. Frozen public
regression uses the new versioned analysis path after candidate freeze;
training workers receive no regression or protected final tasks.

A separate explicit authorization and newly signed admission are still needed
before any of these stages. A draft generated here must never be treated as a
reservation or as evidence that its prerequisite stages passed. Future signing
and execution must recheck source commitments and prerequisite evidence.

Retention, mechanism attribution and independently reviewed novel tasks need
their own protocols and evidence. Operational custody independence, hostile-code
containment, independent qualification and recursive self-improvement remain
unestablished. Qualification stays `NOT_EVALUATED`; external deployment is false.

## Verification

`validate_repair.py NEW_EVIDENCE_DIRECTORY` runs the eight inherited suites and
the new repair suite with the pinned local Python environment. It creates logs,
JUnit XML and `RESULT.json`; it has no cloud/signing/admission operation.

The new tests exercise the actual interrupted fixture, deterministic aggregation,
rounding boundaries, ambient Decimal contexts, strict policy/schema binding,
legacy preservation, temporary test ledgers, all three checkpoint roles and
optimizer interruption. A tiny test model exercises real local optimizer updates
and save/reload inside unit tests. Re-encoded artifact fixtures exist only in
temporary test directories with ephemeral test keys; they are not new training
evidence, protected evaluation or scientific results.

`PRESERVATION.json` binds frozen legacy source files, old authority/evidence
records and the failed budget bytes. The old campaign's full source commitment
belongs to its archived source tree; new controller/reporting code intentionally
has a new identity on this branch.
