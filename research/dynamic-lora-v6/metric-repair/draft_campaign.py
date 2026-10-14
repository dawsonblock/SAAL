"""Render an unsigned review document. No signing, reservation or execution API."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.adaptive_contract import adapter_spec
from saal_rsi.adaptive_contract_v3 import new_job, worker_identity, SOURCES
from saal_rsi.adaptive_dataset import training_projection, validate_training_data
from saal_rsi.numerical_metrics_v1 import POLICY, POLICY_DIGEST
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.spec import canonical, sha

METHODS = ('fixed-r8', 'fixed-r12', 'adalora-12-to-8')
SEEDS = (47, 83, 109, 157, 211)


def build_draft():
    preservation = json.loads(Path(__file__).with_name('PRESERVATION.json').read_text())
    for name, digest in preservation['frozen_files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('preserved historical file changed: '+name)
    population = json.loads((ROOT/'research/evidence-closure-v5/dataset.json').read_text())
    expanded = training_projection(population)
    parent = json.loads((ROOT/'research/behavior-closure-v4/dataset.json').read_text())
    parent_ids = {r['record_id'] for r in parent['records'] if r['split'] == 'train'}
    parent_data = validate_training_data(dict(expanded, records=[r for r in expanded['records']
        if r['split'] == 'development' or r['record_id'] in parent_ids]))
    plans = []
    for stage, pairs, data, prerequisite in (
        ('PILOT', [(m, 60606) for m in METHODS], expanded, 'Fresh explicit authorization and signed admission'),
        ('STUDY', [(m, s) for s in SEEDS for m in METHODS], expanded, 'All three v3 pilot runs, artifacts, CPU/GPU replay and exact-session cleanup pass'),
        ('DATA_CONTROL', [(m, s) for s in (47, 83) for m in ('fixed-r8', 'adalora-12-to-8')], parent_data,
         'All fifteen v3 study runs and cleanup pass; separate parent-data budget authorization'),
    ):
        campaign_id = 'adaptive_v6_v3_proposed_'+stage.lower()
        budget = {'campaign_id': campaign_id, 'max_runs': len(pairs),
                  'total_lifecycle_seconds': 5000*len(pairs)+1200,
                  'max_provisioning_attempts': 1, 'automatic_training_retries': 0,
                  'financial_cap': False}
        runs = []
        for method, seed in pairs:
            name = method.replace('-', '_')+'_'+str(seed)
            spec = model_spec('answerdotai/ModernBERT-base', method='none', rank=0,
                backend='colab', precision='float32', budget=campaign_id+'_'+name, max_tokens=256)
            job_budget = {k: budget[k] for k in ('campaign_id', 'max_runs', 'total_lifecycle_seconds',
                'automatic_training_retries', 'financial_cap')}
            job_budget.update(run_id=name, reserved_seconds=5000)
            job = new_job(spec, data, adapter_spec(method), seed=seed, budget=job_budget,
                          campaign_stage=stage)
            runs.append({'name': name, 'method': method, 'seed': seed,
                         'prospective_unsigned_job_digest': sha(job), 'adapter': job['adapter'],
                         'model_spec': spec})
        plans.append({'stage': stage, 'authorization_status': 'NOT_AUTHORIZED',
            'execution_status': 'NOT_RUN', 'prerequisite': prerequisite, 'budget': budget,
            'training_dataset_digest': sha(data),
            'common_training': {'steps': 1000, 'batch_size': 8, 'accumulation': 2,
                'sample_exposures': 16000, 'learning_rate': .0003, 'head_learning_rate': .0003,
                'max_seconds': 3600, 'checkpoint_interval': 100, 'class_balance': False,
                'sampler_seed': 'same as run seed', 'dropout_seed': 'same as run seed',
                'selection_policy': 'public-guardrails-pair-family-v1',
                'replay_tolerance': 1e-4, 'resume_policy': 'NO_RESUME'}, 'runs': runs})
    controller_sources = list((ROOT/'src/saal_rsi').glob('*.py'))
    controller_sources += list(Path(__file__).parent.glob('*.py'))
    controller_sources += list((ROOT/'research/dynamic-lora-v6').glob('*.py'))
    controller_sources += [ROOT/'configs/dependency-locks'/name for name in (
        'pretrained-linux-x86_64-py313.txt', 'pretrained-macos-arm64-py312.txt')]
    return {'schema': 'saal-adaptive-v3-review-draft-v1', 'authorization_status': 'NOT_AUTHORIZED',
        'executable_admission': False, 'signatures': [], 'budget_reserved': False,
        'worker_identity': worker_identity(), 'numerical_metric_policy': copy.deepcopy(POLICY),
        'numerical_metric_policy_digest': POLICY_DIGEST,
        'worker_source_files': {n: hashlib.sha256((ROOT/'src/saal_rsi'/n).read_bytes()).hexdigest() for n in SOURCES},
        'source_files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(controller_sources)},
        'preservation_digest': sha(preservation),
        'prior_campaign': {'identity': 'study-v2', 'state': 'FAILED', 'budget_reset_permitted': False,
                           'missing_run_imputation_permitted': False, 'reuse_as_completed_permitted': False},
        'plans': plans, 'primary_endpoint': 'terminal family-macro complete counterfactual-pair gain over fixed-r8',
        'proposed_thresholds': {'pair_gain': .05, 'paired_interval_lower_above': 0.,
            'worst_family_nonregression': True, 'critical_error_nonincrease': True,
            'retention_max_loss_under_separate_protocol': .02},
        'analysis_implementations': {
            'comparison': 'research/dynamic-lora-v6/metric-repair/report_study_v3.py',
            'frozen_public_regression': 'research/dynamic-lora-v6/metric-repair/frozen_regression_v3.py',
            'numerical_diagnostics': 'src/saal_rsi/numerical_metrics_v1.py'},
        'analysis_scope': 'Public correlated diagnostics; resource consumption reported separately. No independent population inference.',
        'regression_policy': 'Evaluate only frozen candidates after training; never send regression/protected tasks to worker.',
        'retention': 'NOT_MEASURED_SEPARATE_PROTOCOL_REQUIRED',
        'mechanism_attribution': 'NOT_ESTABLISHED_SEPARATE_ABLATIONS_REQUIRED',
        'independently_reviewed_novel_tasks': 'REQUIRED_NOT_AVAILABLE',
        'operational_custody_independence': 'NOT_ESTABLISHED',
        'hostile_code_containment': 'NOT_ESTABLISHED',
        'recursive_self_improvement': 'NOT_ESTABLISHED',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}


if __name__ == '__main__':
    # Deliberately stdout-only; the caller may preserve this review document.
    sys.stdout.buffer.write(canonical(build_draft())+b'\n')
