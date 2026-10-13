# SAAL-RSI Behavior Closure v4 — release evidence

This is a public research and corrective-engineering release. Qualification is
**NOT_EVALUATED**, independent generalization is **NOT_ESTABLISHED**, and external
deployment remains prohibited. Passing software tests does not qualify a model.

## What changed

The release preserves the 253 frozen v3 source/protocol files and all original
negative outcomes. It adds a strict behavioral screen, failure audits, 1,152
synthetic examples in 576 material counterfactual pairs, a blind-review packet,
and a separately versioned pretrained qualification path. The complete corpus
has 6,616 records: 4,608 training, 960 development, and 1,048 exposed regression.
Each training class has 1,152 examples. The previous 5,464 records are unchanged.
There are **zero human reviews**; the author/operator and labelling oracle remain
shared. The public populations are correlated diagnostic resources.

Colab transport now streams authenticated chunks, verifies the complete native
artifact inventory, and retains explicit recovery receipts. A timeout deleting
optional export files does not invalidate an already verified model; mandatory
runtime termination still requires confirmation. The original failed attempt is
never converted into a successful training-ledger entry. A second preserved
rejection arose from a one-ULP platform difference in the Brier diagnostic.
The new controller validator permits at most eight binary64 ULPs for calibration
diagnostics only. Behavioral counts/rates and the 1e-4 prediction-replay tolerance
remain unchanged.

The release ZIP contains physical generated checkpoint bytes, with repeated full
tensors stored once by content hash. Git uses LFS for large artifacts; GitHub's
automatic source ZIP may still contain pointers. Use the supplied physical ZIP
for reproduction. Pinned upstream ModernBERT weights are a separately verified
cache prerequisite, rather than silently bundled gigabytes.

## Five-seed matched public comparison

ModernBERT revision `11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88` was used on
actual Colab T4 sessions. Seeds 47, 83, 109, 157, and 211 compare a trained head
on a frozen encoder with fixed-rank-4 LoRA plus a trained head. Both receive
1,000 updates and 16,000 sample exposures. Initial, development-selected, and
terminal checkpoints remain distinguishable. No regression example selects
weights, and no seed is selected using protected evaluation.

| Terminal development metric | Head only, mean | Fixed LoRA, mean |
|---|---:|---:|
| Accuracy | 59.54% | 91.67% |
| Complete opposite-label pairs | 28.88% | 83.50% |
| Worst public recipe accuracy | 48.96% | 82.92% |
| Accuracy standard deviation across seeds | 1.64 percentage points | 2.26 percentage points |

Every LoRA seed improves over its matched head-only control, but **none passes
the complete behavior screen**. LoRA complete-pair correctness ranges from
78.96% to 88.96%, below 90%; critical development proposals range from 8 to 50,
above zero. All four classes have correct examples, and the worst development
recipe exceeds 80% for all five LoRA runs. Exposed regression still includes
zero-accuracy recipes. High aggregate accuracy therefore conceals material
failures, including revoked consent, expiry, infeasible budgets, and unfamiliar
instruction conventions.

The unchanged historical lexical rule obtains 40.94% development accuracy on
this exact population. The training-only majority control obtains 25%. These
are useful shortcut controls, not general decision-making benchmarks. Relative
to v3, both supervision and training exposure changed; this study cannot
attribute a gain to diversity alone. The five-seed means are descriptive, not
confidence bounds for an independently sampled task population.

The additional full-tuning seed 47 completed its registered 100-step head
warmup and 1,000 total updates. Both its selected and terminal checkpoints are
step 1,000. This remains an exploratory baseline, not a five-seed full-tuning
comparison:

| Terminal full-tuning metric | Development | Exposed regression |
|---|---:|---:|
| Accuracy | 97.19% | 95.42% |
| Complete opposite-label pairs | 94.38% | 91.98% |
| Worst public recipe accuracy | 78.13% | 0% |
| Critical proposals | 3 | 7 |

It clears the pair target but fails the worst-recipe and critical-error gates
on both populations. Regression family-macro accuracy is 79.58%, substantially
below aggregate accuracy because large synthetic recipes dominate the latter.
The 800-step development checkpoint had better worst-recipe accuracy (88.54%)
but eight critical proposals; it also fails. No checkpoint is promoted. These
results show actual supervised learning and unresolved behavioral failures.

## Software and reconstruction evidence

Fresh Colab QA07 ran six separate suites in hash-locked Python 3.13.16:
**744 passed, one macOS-only skip, zero failures/errors**. Core: 636 collected,
635 passed and one skip; corpus: 38 passed; historical reporting: 10 passed;
current project: 42 passed; previous diversity: 15 passed; previous reliability:
four passed. Actual XML and logs are in `colab-tests-07/evidence/`. Its exact
source snapshot was `77c3a38313f2e12e9f592b7d5ff3a8b745350874`; later differences
are publication fixture coverage, its local tests, and storage attributes,
disclosed in `QA_TO_DELIVERY_SCOPE.json`. The frozen learning workers and jobs
remain unchanged. Local core/corpus/reporting/current/previous suites total
745 passing tests in the pinned macOS Python 3.12 environment.

Clean ZIP extraction, full manifest/signature verification, historical
preservation, and the six-suite extraction run are recorded in the separate
delivery-verification sidecar bound to the exact delivered archive SHA-256.
That record is published after the archive is closed, avoiding a circular
claim that an archive contains its own final hash.

Fresh CPU reconstruction covers all 23 native model roles using exact
initial/best/terminal tensors and the registered 1e-4 probability tolerance.
**21 roles pass and two fail numerical replay**. The earlier v3 uniform seed-47
failure remains: two probability comparisons exceed the tolerance by about
3.1e-6. The new head-only seed-211 terminal development comparison exceeds it
by about 1.4e-6. Labels agree in both failed roles. All five new LoRA models
and the new full-tuning model pass. No tolerance was increased to conceal
either failure. `REPLAY_SUMMARY.json` retains every comparison and actual flag;
these are controller reconstructions, not independently administered evaluation.
The full Colab experiment/runtime cleanup is confirmed, and the final provider
inventory reports no active sessions. Producers have terminated.

## Limits and next admission requirements

No protected evaluation was run, and no protected alpha was spent. The v6
qualification implementation and its negative tests do not establish an
independently administered deployment. Two authenticated human reviewers,
adjudication, separately sourced final families, independently managed custody,
evaluator/witness identities, and a frozen passing candidate remain unmet.
The native critical metric counts ASK/REFUSE cases predicted CHEAP/FAST; it is
not a comprehensive measurement of real-world safety.

The attempted independent code reviews produced no usable report because of
reviewer quota errors and an incompatible peer CLI. The retained review status
is degraded. Same-identity subprocesses are not a hostile-code security sandbox,
and signed worker reports do not attest that a GPU or an honest evaluator
performed the claimed work. Hash locks cover the tested platform environment,
not every possible local/GPU platform. Time quotas are not provider billing caps.

No new Qwen, dynamic-rank, sequential-learning or recursive-improvement campaign
is represented as successful here. Existing implementations and historical
negative evidence are retained. Further rank-search complexity is not justified
by these failed behavioral screens. The next useful work is reviewed semantic
supervision and independently constructed challenge families, followed by a new
bounded matched experiment. Training must continue to remain separate from
qualification and authorization.
