"""Fresh v6 public admission. V5 reservations and signatures are untouched."""
import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'src'))
from saal_rsi.adaptive_contract import adapter_spec, new_job, worker_identity
from saal_rsi.adaptive_dataset import training_projection, validate_training_data
from saal_rsi.adaptive_registry import AdaptiveRegistry
from saal_rsi.adaptive_campaign import execute
from saal_rsi.attempt_service import private_key
from saal_rsi.model_contract import sign, verify
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.spec import canonical, sha

METHODS = ('fixed-r8', 'fixed-r12', 'adalora-12-to-8')
SEEDS = (47, 83, 109, 157, 211)
ENGINEERING_SEED = 60606


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def check_foundation():
    f = json.loads((PROJECT/'FOUNDATION.json').read_text())
    if digest(f['archive']) != f['archive_sha256']:
        raise ValueError('authenticated v5 baseline ZIP changed')
    for name, value in f['protected_v5_files'].items():
        if digest(ROOT/name) != value: raise ValueError('preserved v5 authority record changed: '+name)
    frozen = json.loads((ROOT/'research/reliable-learning-v3/PREREGISTERED_FILES.json').read_text())
    if any(digest(ROOT/name) != value for name, value in frozen.items()):
        raise ValueError('historical frozen source changed')
    if any(s['returncode'] or s['failures'] or s['errors'] for s in f['clean_extraction_tests']['suites']):
        raise ValueError('foundation tests failed')
    connection = sqlite3.connect('file:'+str(ROOT/'research/evidence-closure-v5/experiment/budget.sqlite3')+'?mode=ro', uri=True)
    try: states = connection.execute('SELECT digest,state FROM attempts ORDER BY rowid').fetchall()
    finally: connection.close()
    if [{'digest': d, 'state': s} for d, s in states] != f['preserved_v5_jobs']:
        raise ValueError('v5 reservations changed')
    return sha(f)


def source_map():
    files = list((ROOT/'src/saal_rsi').glob('*.py'))+list(PROJECT.glob('*.py'))
    files += [PROJECT/'FOUNDATION.json', ROOT/'configs/dependency-locks/pretrained-linux-x86_64-py313.txt',
              ROOT/'configs/dependency-locks/pretrained-macos-arm64-py312.txt']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


def pilot_accepted():
    """No study admission based solely on untrusted passing flags."""
    from saal_rsi.adaptive_artifacts import validate_bundle, read_json, file_hash
    from saal_rsi.epoch_metrics import prediction_equal
    base = PROJECT/'pilot-experiment'
    pin = (base/'controller.pub').read_text().strip()
    protocol = verify(read_json(base/'signed_protocol.json'), pin)
    export = read_json(base/'colab/registry-export.json')
    state = read_json(base/'colab/monitor-state.json')
    if export['campaign_state'] != 'COMPLETED' or state['cleanup_verified'] is not True or len(export['attempts']) != 3:
        raise ValueError('complete pilot and verified cleanup required before scientific admission')
    hashes, streams, methods = set(), set(), set()
    for row, descriptor in zip(export['attempts'], protocol['runs']):
        if row['state'] != 'COMPLETED': raise ValueError('pilot has failed / incomplete attempts')
        job = verify(read_json(base/(descriptor['name']+'-job.json')), pin)
        if job['steps'] != 1000 or job['seed'] != ENGINEERING_SEED: raise ValueError('full excluded-seed pilot required')
        native = validate_bundle(row['outcome']['artifacts'], job, pin, require_replay=True)
        receipt = read_json(row['outcome']['controller_replay'])
        if receipt['inventory_sha256'] != file_hash(Path(row['outcome']['artifacts'])/'artifact_inventory.json') or receipt['job_digest'] != sha(job) or receipt['verified'] is not True:
            raise ValueError('fresh CPU replay identity failed')
        for role in ('initial', 'best', 'terminal'):
            if not prediction_equal(receipt['predictions'][role], native['snapshots'][role]['predictions'], 'modernbert-seqcls-v1', 1e-4):
                raise ValueError('pilot checkpoint replay failed')
        hashes.add(native['unadapted_model_hash'])
        streams.add(sha(read_json(Path(row['outcome']['artifacts'])/'sampler_manifest.json')['record_ids']))
        methods.add(job['adapter']['method'])
    if len(hashes) != 1 or len(streams) != 1 or methods != set(METHODS):
        raise ValueError('pilot model / classifier / sampling mismatch')
    return sha(export)


def prepare(stage, key_path):
    foundation_digest = check_foundation()
    prior_pilot = pilot_accepted() if stage != 'PILOT' else None
    if stage not in ('PILOT', 'STUDY', 'DATA_CONTROL'): raise ValueError('campaign phase')
    # These matched public diagnostic comparisons cannot establish independent
    # generalization. Separately authored/adjudicated final tasks remain absent.
    population = json.loads((ROOT/'research/evidence-closure-v5/dataset.json').read_text())
    parent = json.loads((ROOT/'research/behavior-closure-v4/dataset.json').read_text())
    data = training_projection(population)
    parent_training_ids = {r['record_id'] for r in parent['records'] if r['split'] == 'train'}
    control = validate_training_data(dict(data, records=[r for r in data['records']
        if r['split'] == 'development' or r['record_id'] in parent_training_ids]))
    if stage == 'PILOT': runs = [(m, ENGINEERING_SEED, 'expanded') for m in METHODS]
    elif stage == 'STUDY': runs = [(m, s, 'expanded') for s in SEEDS for m in METHODS]
    else: runs = [(m, s, 'parent') for s in (47, 83) for m in ('fixed-r8', 'adalora-12-to-8')]
    folder = PROJECT/({'PILOT': 'pilot-experiment', 'STUDY': 'study-experiment', 'DATA_CONTROL': 'data-control-experiment'}[stage])
    folder.mkdir(exist_ok=False)
    key = private_key(key_path)
    pin = key.public_key().public_bytes_raw().hex()
    campaign_id = 'adaptive_v6_'+stage.lower()
    budget = {'campaign_id': campaign_id, 'max_runs': len(runs),
              'total_lifecycle_seconds': len(runs)*5000+1200,
              'max_provisioning_attempts': 1, 'automatic_training_retries': 0, 'financial_cap': False}
    if budget['total_lifecycle_seconds'] > 100000: raise ValueError('campaign budget bound')
    jobs, descriptors = [], []
    for method, seed, source in runs:
        name = method.replace('-', '_')+'_'+str(seed)
        spec = model_spec('answerdotai/ModernBERT-base', method='none', rank=0, backend='colab',
                          precision='float32', budget=campaign_id+'_'+name, max_tokens=256)
        job_budget = {k: budget[k] for k in ('campaign_id', 'max_runs', 'total_lifecycle_seconds', 'automatic_training_retries', 'financial_cap')}
        job_budget.update(run_id=name, reserved_seconds=5000)
        job = new_job(spec, control if source == 'parent' else data,
                      adapter_spec(method), seed=seed, budget=job_budget, campaign_stage=stage)
        envelope = sign(job, key)
        (folder/(name+'-job.json')).write_bytes(canonical(envelope))
        jobs.append(envelope)
        descriptors.append({'name': name, 'method': method, 'seed': seed, 'population': source, 'job_digest': sha(job)})
    protocol = {'schema': 'saal-adaptive-campaign-v1', 'stage': stage, 'source_files': source_map(),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'foundation_digest': foundation_digest, 'worker_identity': worker_identity(),
                'prior_pilot_digest': prior_pilot, 'runs': descriptors, 'job_digests': [d['job_digest'] for d in descriptors],
                'parent_training_population_digest': sha([r for r in parent['records'] if r['split'] == 'train']),
                'budget': budget, 'engineering_seed_excluded': ENGINEERING_SEED,
                'comparison_seeds': list(SEEDS), 'selection': 'PUBLIC_DEVELOPMENT_ONLY; retain initial, best, terminal',
                'matched_conditions': ['official base weights and initialized head', 'public data', 'training/dropout and sampling seeds',
                                       '1000 completed optimizer updates', '16000 training sample exposures', 'head and classifier policy',
                                       'AdamW rates', '22 Wqkv target modules', '256 native tokens', 'FP32', 'checkpoint schedule'],
                'important_difference': 'AdaLoRA uses SVD parameterization, orthogonal regularization, init-r12 and persistent masks. A gain alone does not isolate adaptive allocation; no-allocation and random-allocation ablations are subsequent separately admitted tests.',
                'preflight': 'one identical 8-example forward/backward probe with zero optimizer updates; discarded gradients and reset training RNG; cost reported separately',
                'primary_endpoint': 'terminal family-macro complete counterfactual-pair gain over fixed-r8',
                'decision_thresholds': {'pair_gain': .05, 'paired_interval_lower_above': 0., 'worst_family_nonregression': True,
                                        'retention_max_loss': .02, 'critical_error_nonincrease': True},
                'regression_policy': 'Not included in job or worker. Evaluate only frozen candidates; never restart/tune on protected data.',
                'resources': 'Equal updates and sample exposures do not imply equal compute or stored tensor capacity. Measure time, peak memory and parameters.',
                'protected_alpha_consumed': False, 'human_reviews': 0, 'independent_generalization': 'NOT_ESTABLISHED',
                'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}
    signed = sign(protocol, key)
    (folder/'controller.pub').write_text(pin+'\n')
    (folder/'signed_protocol.json').write_bytes(canonical(signed))
    ledger = AdaptiveRegistry(folder/'budget.sqlite3', pin, signed, create=True)
    try:
        for envelope in jobs: ledger.reserve(envelope)
        (folder/'registry-export.json').write_bytes(canonical(ledger.records()))
    finally: ledger.close()
    print(json.dumps({'stage': stage, 'jobs_reserved': len(jobs), 'protocol_digest': sha(protocol), 'qualification_status': 'NOT_EVALUATED'}))


def run(stage, cache):
    check_foundation()
    if stage != 'PILOT': pilot_accepted()
    folder = PROJECT/({'PILOT': 'pilot-experiment', 'STUDY': 'study-experiment', 'DATA_CONTROL': 'data-control-experiment'}[stage])
    pin = (folder/'controller.pub').read_text().strip()
    protocol = verify(json.loads((folder/'signed_protocol.json').read_text()), pin)
    if protocol['source_files'] != source_map() or protocol['worker_identity'] != worker_identity():
        raise ValueError('preregistered implementation changed')
    return execute(folder/'colab', [folder/(d['name']+'-job.json') for d in protocol['runs']],
                   folder/'signed_protocol.json', folder/'budget.sqlite3', pin, folder/'artifacts', cache)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'run', 'check'))
    p.add_argument('--stage', choices=('PILOT', 'STUDY', 'DATA_CONTROL'), default='PILOT')
    p.add_argument('--key'); p.add_argument('--cache')
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.stage, a.key)
    elif a.action == 'run': run(a.stage, a.cache)
    else: print(json.dumps({'foundation_digest': check_foundation()}))
