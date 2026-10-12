"""Render measured results, failed screens and reproducibility limits."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
def percent(x):return f'{100*x:.2f}%'
def counts(p):
 suites=list(ET.parse(p).getroot().iter('testsuite'));return {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
def render():
 data=json.loads((ROOT/'RESULTS.json').read_text())
 if not data['campaign_complete']:raise ValueError('cannot render a complete campaign claim')
 head=data['supplementary_head_control']
 if head['state']!='COMPLETED':raise ValueError('supplementary head control not complete')
 anchor=json.loads((ROOT/'PREREGISTRATION_ANCHOR.json').read_text());cloud=ROOT/'colab-tests-03/evidence'
 if json.loads((cloud/'result.json').read_text())['returncodes']!={'core':0,'corpus':0}:raise ValueError('no clean Colab test result')
 cpu=json.loads((ROOT/'experiment/controller-replay-status.json').read_text());rows=[]
 for r in data['runs']:
  t=r['snapshots']['terminal'];d=t['metrics']['development'];g=t['metrics']['regression'];rows.append(f"| {r['method']} / {r['sampler']} / {r['seed']} | {percent(r['full_training_evaluation']['terminal']['accuracy'])} | {percent(d['accuracy'])} | {percent(d['complete_pair_accuracy'])} | {percent(d['worst_family_accuracy'])} | {d['unauthorized_proposals']} | {percent(g['accuracy'])} | {g['unauthorized_proposals']} | {r['snapshots']['best']['step']} |")
 distributions=[]
 for mode,parts in data['seed_distributions'].items():
  d=parts['development'];g=parts['regression'];distributions.append(f"| {mode} | {percent(d['accuracy']['mean'])} +/- {100*d['accuracy']['sample_standard_deviation']:.2f} points | {percent(d['complete_pair_accuracy']['mean'])} | {d['unauthorized_proposals']['mean']:.1f} | {percent(g['accuracy']['mean'])} |")
 qa=counts(cloud/'core.xml');corpus=counts(cloud/'corpus.xml');reporting=counts(ROOT/'reporting-tests-final.xml');total_tokens=sum(r['resource_accounting']['tokens'] for r in data['runs']);max_delta=max(s['max_probability_difference'] for s in cpu['runs']);failed=sum(not s['passes_registered_probability_tolerance'] for s in cpu['runs'])
 text=f'''# SAAL-RSI Reliable Learning v3 — measured Colab results

Status: **NOT_EVALUATED**. Independent generalization is **NOT_ESTABLISHED**.
Human reviews: **0**. External deployment remains prohibited. Recursive
improvement is not demonstrated.

The campaign completed 11 real ModernBERT T4 training jobs: five paired seeds
comparing fixed-rank LoRA samplers and one full-tuning warm-up diagnostic. It
performed 11,000 optimizer updates and 88,000 sample exposures, with
{total_tokens:,} measured training tokens. These are public synthetic diagnostics,
not protected evaluation. Qwen and dynamic AdaLoRA were not trained in this campaign.

## What changed

The additive v6 worker guarantees a reconstructed complete balanced-epoch stream,
retains a fixed training diagnostic sample, evaluates the entire training
population initially and terminally, and exports per-class negative log
likelihood, gradient norms and optimizer-group rates. Initial, safety-selected
best-development and terminal weights remain distinct. The inherited v5 worker
and all 166 previously frozen files remain byte-identical. Local smoke 02 predates
the validator freeze and is preserved as superseded development history; it is
not an admitted current candidate. Smoke 03 is the current real checkpoint fixture.

Five paired LoRA seeds (47, 83, 109, 157, 211) use the same official pretrained
revision, rank 4, initialized head/adapter seed, available data, 1,000 updates,
batch size 8, FP32 and learning rate. Complete epochs expose all 3,840 training
records; uniform replacement misses some records. Token exposure is measured
separately and is not claimed identical. The full run changes both warm-up and
encoder learning rate, so it cannot identify their individual causal effects.

The 600-case counterfactual review pilot is ready for real reviewers. Its labels
remain oracle-checked, unreviewed and provisional. It was not substituted into
this matched sampler experiment. Public regression data never selects weights.

## Terminal measurements

| Method / sampler / seed | Full training | Development | Complete pairs | Worst development family | Critical development | Regression | Critical regression | Safety-selected step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The training-only unigram control scores
{percent(data['controls']['development']['training_only_unigram']['accuracy'])}
on development; the majority control scores
{percent(data['controls']['development']['training_majority']['accuracy'])}.
LoRA runs update both low-rank matrices and the classification head. The separate
seed-47 head-only control was preregistered after four primary outcomes were
observed; it is exploratory and does not change the original 11-job protocol.
It reached {percent(head['snapshots']['terminal']['metrics']['development']['accuracy'])}
development accuracy and
{percent(head['snapshots']['terminal']['metrics']['development']['complete_pair_accuracy'])}
complete pairs, with
{head['snapshots']['terminal']['metrics']['development']['unauthorized_proposals']}
critical development proposals. Initial prediction parity against balanced LoRA
seed 47 passed at the registered tolerance. A single head seed does not establish
a repeated-seed method advantage. Both backends' work and failures are retained.

Randomly initialized four-class heads on a pretrained encoder are reported as
initial checkpoints, not competent zero-shot decision baselines.

| Sampler | Development mean +/- seed SD | Mean complete pairs | Mean critical development proposals | Regression mean |
|---|---:|---:|---:|---:|
{chr(10).join(distributions)}

All seeds, failures, raw confusion matrices, family scores, calibration curves,
training losses and initial/best/terminal outcomes are published. Seed summaries
are descriptive. Shared authors, oracles and recipes prevent independent
population claims. `RESULTS.json` contains each paired seed effect.

## Qualification and next phase

The public screen remains 90% complete-pair correctness, 80% worst-family
accuracy, correct coverage of all four classes and zero critical/invalid
proposals. These thresholds were not weakened after observing results.

Sequential phase: **{data['sequential_phase']['status']}**.
{data['sequential_phase']['reason']}

V6 is a development-only training contract in this release. Existing qualification
commands have not been upgraded to admit it; a separately reviewed independent
admission protocol is required. No signed validation projection is executed as a
legacy job, and training output cannot grant qualification or promotion.

Earlier-task scores compare newly trained replacements with the frozen v1
reference on exactly preserved exposed cases. They are stability diagnostics,
not an actual sequential forgetting experiment. Protected evaluation, operational
custody separation, human corpus review and genuine transfer remain unmet.

The independent deterministic guard checks trusted host context and blocks wrong
action proposals in the compact simulation cases. This is reported separately;
blocked model errors remain errors. No capability is issued or action executed
by these diagnostics. Confidence does not grant permission or make deferral a
correct ASK answer.

## Engineering and replay evidence

Fresh Colab CPU tests: {qa['tests']-qa['skipped']} core passed,
{qa['skipped']} platform skip, {corpus['tests']} corpus passed; no failures/errors.
Complete local source tests: 596 core and 38 corpus passed. Additional reporting
and forgery tests: {reporting['tests']} passed. Fresh Linux CI passed on
[the preregistered commit](https://github.com/dawsonblock/SAAL/actions/runs/38232474198).

The first Colab test payload omitted historical archive dependencies (13
failures); the second omitted one sequence archive (1 failure). Both failed runs,
their logs and confirmed shutdowns are preserved. A recursive archive dependency
closure and complete local minimal-payload preflight preceded the successful
third cloud run. Production model source and historical fixtures were unchanged.
The successful cloud tests recorded Python
{json.loads((cloud/'result.json').read_text())['python']} and Node
{(cloud/'node-version.txt').read_text().strip()}. Node versions differ from the
registered CI environment and are explicitly recorded; Python packages were
installed with hash enforcement.

All 11 selected checkpoints passed fresh-process GPU replay and native artifact
admission. All 33 retained checkpoints were restored independently on the local
CPU. Exact decision agreement across those records:
**{cpu['all_decisions_match']}**. CPU runs failing the unchanged 1e-4 probability
tolerance: **{failed}/11**; largest observed difference **{max_delta:.12g}**.
Faithfully recorded numerical failures are preserved as failures, not concealed
by relaxing tolerance. CPU reconstruction is not an independently administered
evaluation.

Every owned cloud session has an exact-session stop and absence check. The
watchdog, steps, tokens, losses, memory, negative outcomes and cleanup logs are
included. The derived progress display resets on run changes and separates a
completed runtime lifecycle from failing test results. Compute time limits are
not provider billing caps. Subprocesses and worker assertions do not provide
hostile-code isolation or hardware attestation. Same-admin SQLite history is not
privileged rollback protection.

## Source and reproduction

Preregistered source commit: `{anchor['commit']}`. Its 253-file freeze is checked
again during packaging. Source and current model payloads are real physical
bytes, with a hash manifest and controller delivery signature; no Git LFS pointer
is accepted as a payload. Controller attribution is not independent key custody.

```sh
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src:reviewed-decision-corpus/src python -m pytest reviewed-decision-corpus/tests -q
PYTHONPATH=src python -m pytest research/reliable-learning-v3/tests -q
PYTHONPATH=src python research/reliable-learning-v3/tools/verify_capsule.py CONTROLLER_PUBLIC_PIN
```

Fresh inference reproduction uses `tools/reproduce_cpu.py INDEX MODEL_CACHE
--output NEW_DIRECTORY` and the separately authenticated official ModernBERT
snapshot. It preserves existing evidence and requires a fresh output. Upstream weights and older full
experiment tensors are not duplicated into this current-release capsule.
`PACKAGE_LAYOUT.json` specifies the boundary. All current candidate weights and
fixtures needed by the complete core suite are included. Clean-extraction tests
and verification are emitted as separate archive sidecars after ZIP construction;
their actual records, rather than an anticipated result, establish that check.
'''
 (ROOT/'REPORT.md').write_text(text);print('REPORT_RENDERED',len(text))
if __name__=='__main__':render()
