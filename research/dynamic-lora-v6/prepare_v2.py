"""Fresh CUDA repair admissions; consumed v1 budgets are never reset.

The full pilot explicitly reuses the two already completed fixed controls.
Their original failed campaign remains failed. Only their individually
completed, authenticated and replayed runs can enter the composite gate.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(PROJECT))
from prepare import check_foundation, source_map, ENGINEERING_SEED, SEEDS, METHODS
from native_smoke import small_population
from saal_rsi.adaptive_artifacts import read_json, file_hash
from saal_rsi.adaptive_contract import adapter_spec, new_job as v1_job, worker_identity as v1_identity
from saal_rsi.adaptive_contract_v2 import new_job as v2_job, worker_identity as v2_identity, RANKNUM_POLICY
from saal_rsi.adaptive_dataset import training_projection, validate_training_data
from saal_rsi.adaptive_dispatch import validate_bundle
from saal_rsi.adaptive_registry import AdaptiveRegistry
from saal_rsi.adaptive_campaign import execute
from saal_rsi.attempt_service import private_key
from saal_rsi.epoch_metrics import prediction_equal
from saal_rsi.model_contract import sign, verify
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.spec import canonical, sha

DIRECTORIES = {'PROBE': 'cuda-v2-probe', 'PILOT': 'pilot-v2-01',
               'STUDY': 'study-v2', 'DATA_CONTROL': 'data-control-v2'}
FIXED_CONTROL = PROJECT/'pilot-repair-02'


def completed_runs(folder, methods, *, allow_failed_campaign=False, steps=1000):
    """Reconstruct completion from real bundles and bound replay receipts."""
    folder = Path(folder)
    pin = (folder/'controller.pub').read_text().strip()
    if pin != (PROJECT/'controller.pub').read_text().strip():
        raise ValueError('unexpected controller identity')
    protocol = verify(read_json(folder/'signed_protocol.json'), pin)
    export = read_json(folder/'colab/registry-export.json')
    state = read_json(folder/'colab/monitor-state.json')
    if (export['protocol_digest'] != sha(protocol) or state['cleanup_verified'] is not True
            or export['provisioning_attempts'] != 1):
        raise ValueError('consumed admission, authenticated export and confirmed cleanup required')
    if allow_failed_campaign:
        if export['campaign_state'] != 'FAILED' or state['phase'] != 'FAILED':
            raise ValueError('preserve the failed parent campaign')
    elif export['campaign_state'] != 'COMPLETED' or state['phase'] != 'COMPLETED':
        raise ValueError('new repair admission did not complete')
    if len(export['attempts']) != len(protocol['runs']):
        raise ValueError('incomplete attempt accounting')
    found = {}
    for descriptor, row in zip(protocol['runs'], export['attempts']):
        if descriptor['method'] not in methods:
            continue
        job = verify(read_json(folder/(descriptor['name']+'-job.json')), pin)
        digest = sha(job)
        outcome = row['outcome']
        if (row['state'] != 'COMPLETED' or not outcome or row['digest'] != digest
                or descriptor['job_digest'] != digest or outcome['job_digest'] != digest
                or outcome['method'] != descriptor['method'] or outcome['seed'] != job['seed']):
            raise ValueError('a failed or substituted run cannot become a control')
        import re
        name = outcome['run']
        if type(name) is not str or not re.fullmatch(r'run-[0-9]{2}-'+digest[7:19], name):
            raise ValueError('run artifact identity')
        artifact = folder/'artifacts'/name
        native = validate_bundle(artifact, job, pin, require_replay=True)
        receipt = read_json(folder/'colab'/(name+'-controller-replay.json'))
        if (receipt['verified'] is not True or receipt['job_digest'] != digest
                or receipt['inventory_sha256'] != file_hash(artifact/'artifact_inventory.json')
                or outcome['result_digest'] != sha(native) or outcome['controller_replay_digest'] != sha(receipt)
                or any(not prediction_equal(receipt['predictions'][role], native['snapshots'][role]['predictions'],
                    'modernbert-seqcls-v1', 1e-4) for role in ('initial', 'best', 'terminal'))):
            raise ValueError('control checkpoint replay or lineage mismatch')
        if job['steps'] != steps or job['seed'] != ENGINEERING_SEED:
            raise ValueError('excluded engineering seed and exact update count required')
        if job['adapter']['method'] == 'adalora-12-to-8':
            if (job['schema'] != 'saal-adaptive-job-v2' or native['ranknum_policy'] != RANKNUM_POLICY
                    or native['ranknum_normalized'] != 22 or native['runtime']['device'] != 'cuda:0'):
                raise ValueError('real CUDA repair evidence required')
        method = job['adapter']['method']
        if method in found:
            raise ValueError('duplicate method in engineering gate')
        found[method] = {'job': job, 'result': native, 'binding': {
            'method': method, 'job_digest': digest, 'protocol_digest': sha(protocol),
            'export_digest': sha(export), 'inventory_sha256': file_hash(artifact/'artifact_inventory.json'),
            'controller_replay_digest': sha(receipt), 'unadapted_model_hash': native['unadapted_model_hash'],
            'sample_stream_digest': sha(read_json(artifact/'sampler_manifest.json')['record_ids']),
            'artifact_path': artifact.relative_to(ROOT).as_posix(),
            'original_campaign_state': export['campaign_state']}}
    if set(found) != set(methods):
        raise ValueError('missing completed engineering method')
    return found


def fixed_controls():
    return completed_runs(FIXED_CONTROL, ('fixed-r8', 'fixed-r12'), allow_failed_campaign=True)


def probe_accepted():
    value = completed_runs(PROJECT/DIRECTORIES['PROBE'], ('adalora-12-to-8',), steps=4)
    return sha(value['adalora-12-to-8']['binding'])


def composite_pilot():
    fixed = fixed_controls()
    folder = PROJECT/DIRECTORIES['PILOT']
    protocol = verify(read_json(folder/'signed_protocol.json'), (folder/'controller.pub').read_text().strip())
    if protocol['frozen_fixed_controls'] != [fixed[m]['binding'] for m in METHODS[:2]]:
        raise ValueError('fixed controls differ from the repair preregistration')
    if protocol['prior_probe_digest'] != probe_accepted():
        raise ValueError('CUDA engineering probe changed')
    runs = fixed | completed_runs(folder, ('adalora-12-to-8',))
    hashes = {r['binding']['unadapted_model_hash'] for r in runs.values()}
    streams = {r['binding']['sample_stream_digest'] for r in runs.values()}
    common = ('dataset_digest', 'seed', 'sampler_seed', 'dropout_seed', 'steps', 'batch_size',
              'accumulation', 'learning_rate', 'head_learning_rate', 'class_balance',
              'checkpoint_interval', 'selection_policy', 'peft_implementation', 'replay_tolerance')
    for field in common:
        if len({sha(r['job'][field]) for r in runs.values()}) != 1:
            raise ValueError('composite pilot unmatched condition: '+field)
    if len(hashes) != 1 or len(streams) != 1 or any(r['result']['resource_accounting']['sample_exposures'] != 16000 for r in runs.values()):
        raise ValueError('matched base/head, stream and 16000 exposures required')
    return {'schema': 'saal-adaptive-composite-pilot-v2', 'accepted': True,
            'bindings': [runs[m]['binding'] for m in METHODS],
            'scope': 'Engineering only. Completed v1 fixed controls plus separately admitted v2 CUDA AdaLoRA; failed parent campaigns remain failed.',
            'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}


def prepare(stage, key_path):
    foundation = check_foundation()
    fixed = fixed_controls() if stage == 'PILOT' else None
    probe = probe_accepted() if stage == 'PILOT' else None
    pilot = composite_pilot() if stage in ('STUDY', 'DATA_CONTROL') else None
    if stage == 'DATA_CONTROL':
        # No parent-population study before every main-study run and teardown.
        folder = PROJECT/DIRECTORIES['STUDY']
        export = read_json(folder/'colab/registry-export.json')
        state = read_json(folder/'colab/monitor-state.json')
        if export['campaign_state'] != 'COMPLETED' or len(export['attempts']) != 15 or any(r['state'] != 'COMPLETED' for r in export['attempts']) or state['cleanup_verified'] is not True:
            raise ValueError('complete 15-run study before exploratory data control')
    population = read_json(ROOT/'research/evidence-closure-v5/dataset.json')
    data = small_population(population) if stage == 'PROBE' else training_projection(population)
    if stage == 'DATA_CONTROL':
        parent = read_json(ROOT/'research/behavior-closure-v4/dataset.json')
        ids = {r['record_id'] for r in parent['records'] if r['split'] == 'train'}
        data = validate_training_data(dict(data, records=[r for r in data['records'] if r['split'] == 'development' or r['record_id'] in ids]))
    runs = [('adalora-12-to-8', ENGINEERING_SEED)] if stage in ('PROBE', 'PILOT') else (
        [(m, s) for s in SEEDS for m in METHODS] if stage == 'STUDY' else
        [(m, s) for s in (47, 83) for m in ('fixed-r8', 'adalora-12-to-8')])
    folder = PROJECT/DIRECTORIES[stage]
    folder.mkdir(exist_ok=False)
    key = private_key(key_path)
    pin = key.public_key().public_bytes_raw().hex()
    if pin != (PROJECT/'controller.pub').read_text().strip():
        raise ValueError('new admission controller identity')
    campaign_id = 'adaptive_v6_v2_'+stage.lower()
    reserved = 2300 if stage == 'PROBE' else 5000
    budget = {'campaign_id': campaign_id, 'max_runs': len(runs), 'total_lifecycle_seconds': len(runs)*reserved+1200,
              'max_provisioning_attempts': 1, 'automatic_training_retries': 0, 'financial_cap': False}
    jobs, descriptors = [], []
    for method, seed in runs:
        name = method.replace('-', '_')+'_'+str(seed)
        spec = model_spec('answerdotai/ModernBERT-base', method='none', rank=0, backend='colab', precision='float32', budget=campaign_id+'_'+name, max_tokens=256)
        b = {k: budget[k] for k in ('campaign_id', 'max_runs', 'total_lifecycle_seconds', 'automatic_training_retries', 'financial_cap')}
        b.update(run_id=name, reserved_seconds=reserved)
        options = dict(seed=seed, budget=b, campaign_stage='ENGINEERING' if stage == 'PROBE' else stage)
        adapter = adapter_spec(method)
        if stage == 'PROBE':
            adapter = adapter_spec(method, total_steps=4, tinit=1, tfinal=1, delta=1)
            options.update(steps=4, max_seconds=900, checkpoint_interval=2)
        job = (v2_job if method == 'adalora-12-to-8' else v1_job)(spec, data, adapter, **options)
        envelope = sign(job, key)
        (folder/(name+'-job.json')).write_bytes(canonical(envelope))
        jobs.append(envelope)
        descriptors.append({'name': name, 'method': method, 'seed': seed, 'job_digest': sha(job)})
    protocol = {'schema': 'saal-adaptive-campaign-v2', 'stage': 'ENGINEERING' if stage == 'PROBE' else stage,
        'source_files': source_map(), 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'foundation_digest': foundation, 'worker_identities': {'v1': v1_identity(), 'v2': v2_identity()},
        'ranknum_policy': RANKNUM_POLICY, 'frozen_fixed_controls': [fixed[m]['binding'] for m in METHODS[:2]] if fixed else [],
        'prior_probe_digest': probe, 'composite_pilot_digest': sha(pilot) if pilot else None,
        'runs': descriptors, 'job_digests': [d['job_digest'] for d in descriptors], 'budget': budget,
        'engineering_seed_excluded': ENGINEERING_SEED, 'comparison_seeds': list(SEEDS),
        'primary_endpoint': 'terminal family-macro complete counterfactual-pair gain over fixed-r8',
        'decision_thresholds': {'pair_gain': .05, 'paired_interval_lower_above': 0., 'worst_family_nonregression': True, 'retention_max_loss': .02, 'critical_error_nonincrease': True},
        'selection': 'PUBLIC_DEVELOPMENT_ONLY; preserve initial, best and terminal; compare terminal endpoints',
        'interpretation': 'Public correlated diagnostics. Equal updates/exposures are not equal compute. AdaLoRA also differs in parameterization, regularization and initial capacity.',
        'regression_policy': 'No regression or protected labels in worker. Never tune or retry using protected results.',
        'protected_alpha_consumed': False, 'human_reviews': 0, 'independent_generalization': 'NOT_ESTABLISHED',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    signed = sign(protocol, key)
    for name, expected in protocol['source_files'].items():
        source = ROOT/name
        if file_hash(source) != expected:
            raise ValueError('source changed during admission')
        target = folder/'preregistered-source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    (folder/'controller.pub').write_text(pin+'\n')
    (folder/'signed_protocol.json').write_bytes(canonical(signed))
    ledger = AdaptiveRegistry(folder/'budget.sqlite3', pin, signed, create=True)
    try:
        for envelope in jobs: ledger.reserve(envelope)
        (folder/'registry-export.json').write_bytes(canonical(ledger.records()))
    finally: ledger.close()
    print(json.dumps({'stage': stage, 'reserved_jobs': len(jobs), 'protocol_digest': sha(protocol)}))


def run(stage, cache):
    check_foundation()
    folder = PROJECT/DIRECTORIES[stage]
    pin = (folder/'controller.pub').read_text().strip()
    p = verify(read_json(folder/'signed_protocol.json'), pin)
    if p['source_files'] != source_map() or p['worker_identities'] != {'v1': v1_identity(), 'v2': v2_identity()}:
        raise ValueError('preregistered source changed')
    if stage == 'PILOT' and (p['prior_probe_digest'] != probe_accepted() or p['frozen_fixed_controls'] != [fixed_controls()[m]['binding'] for m in METHODS[:2]]):
        raise ValueError('preregistered repair controls changed')
    if stage in ('STUDY', 'DATA_CONTROL') and p['composite_pilot_digest'] != sha(composite_pilot()):
        raise ValueError('composite engineering pilot changed')
    return execute(folder/'colab', [folder/(d['name']+'-job.json') for d in p['runs']], folder/'signed_protocol.json', folder/'budget.sqlite3', pin, folder/'artifacts', cache)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'gate'))
    parser.add_argument('--stage', choices=tuple(DIRECTORIES), default='PROBE')
    parser.add_argument('--key'); parser.add_argument('--cache')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.stage, args.key)
    elif args.action == 'run': run(args.stage, args.cache)
    else: print(json.dumps(composite_pilot()))
