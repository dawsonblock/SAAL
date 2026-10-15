"""Create a fresh signed v3 engineering-pilot admission; never provisions Colab."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))

from saal_rsi.adaptive_contract import adapter_spec
from saal_rsi.adaptive_contract_v3 import new_job, worker_identity
from saal_rsi.adaptive_dataset import training_projection
from saal_rsi.adaptive_evaluator_v3 import validate_protocol, validate_sources
from saal_rsi.adaptive_registry import AdaptiveRegistry
from saal_rsi.model_contract import sign, verify
from saal_rsi.numerical_metrics_v1 import POLICY, POLICY_DIGEST
from saal_rsi.pretrained_contract import model_spec
from saal_rsi.spec import canonical, sha


def _draft_inputs():
    draft = json.loads((Path(__file__).with_name('CAMPAIGN_DRAFT.json')).read_text())
    population = json.loads((ROOT/'research/evidence-closure-v5/dataset.json').read_text())
    return draft, training_projection(population)


def _pilot_jobs(draft, dataset):
    stage = next(p for p in draft['plans'] if p['stage'] == 'PILOT')
    if stage['authorization_status'] != 'NOT_AUTHORIZED' or stage['execution_status'] != 'NOT_RUN':
        raise ValueError('pilot draft must remain an unsigned, unexecuted proposal')
    jobs = []
    for run in stage['runs']:
        spec = model_spec('answerdotai/ModernBERT-base', method='none', rank=0,
            backend='colab', precision='float32', budget=run['model_spec']['resource_budget_id'], max_tokens=256)
        if spec != run['model_spec']:
            raise ValueError('registered pretrained model identity changed')
        budget = {key: stage['budget'][key] for key in ('campaign_id', 'max_runs',
            'total_lifecycle_seconds', 'automatic_training_retries', 'financial_cap')}
        budget.update(run_id=run['name'], reserved_seconds=5000)
        job = new_job(spec, dataset, adapter_spec(run['method']), seed=run['seed'],
                      budget=budget, campaign_stage='PILOT')
        if (sha(job) != run['prospective_unsigned_job_digest']
                or job['adapter'] != run['adapter']):
            raise ValueError('unsigned pilot job differs from frozen draft')
        jobs.append(job)
    return stage, jobs


def build_payload(draft, stage, jobs):
    from saal_rsi.adaptive_evaluator_v3 import METHODS
    budget = dict(stage['budget'])
    budget['max_provisioning_attempts'] = 1
    descriptors = [{'name': run['name'], 'method': run['method'], 'seed': run['seed'],
                    'job_digest': sha(job)} for run, job in zip(stage['runs'], jobs)]
    return {
        'schema': 'saal-adaptive-campaign-v3',
        'campaign_id': budget['campaign_id'],
        'stage': 'PILOT',
        'runs': descriptors,
        'job_digests': [row['job_digest'] for row in descriptors],
        'budget': budget,
        'worker_identities': {'v3': worker_identity()},
        'numerical_metric_policy': POLICY,
        'numerical_metric_policy_digest': POLICY_DIGEST,
        'colab_cli_version': draft['colab_cli_version'],
        'source_files': draft['source_files'],
        'preservation_digest': draft['preservation_digest'],
        'prior_campaign': draft['prior_campaign'],
        'training_dataset_digest': stage['training_dataset_digest'],
        'primary_endpoint': draft['primary_endpoint'],
        'proposed_thresholds': draft['proposed_thresholds'],
        'scope': 'Engineering pilot on public correlated diagnostics; no qualification or deployment claim.',
        'qualification_status': 'NOT_EVALUATED',
        'external_deployment_allowed': False,
    }


def _read_signing_key(path, expected_public):
    path = Path(path)
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size != 32
            or info.st_uid != os.getuid()):
        raise ValueError('controller signing key must be an owner-only regular 32-byte file')
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.from_private_bytes(path.read_bytes())
    public = key.public_key().public_bytes_raw()
    if public.hex() != expected_public:
        raise ValueError('controller signing key does not match the pinned controller identity')
    return key, hashlib.sha256(public).hexdigest()


def prepare(output, key_path, pin_path):
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError('preserve previous admissions and campaign evidence')
    pin = Path(pin_path).read_text().strip()
    if len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin):
        raise ValueError('pinned controller public key required')
    key, key_fingerprint = _read_signing_key(key_path, pin)
    draft, dataset = _draft_inputs()
    stage, jobs = _pilot_jobs(draft, dataset)
    payload = build_payload(draft, stage, jobs)
    validate_protocol(payload)
    required = {'src/saal_rsi/'+name for name in __import__(
        'saal_rsi.adaptive_contract_v3', fromlist=['SOURCES']).SOURCES} | {
        'src/saal_rsi/adaptive_campaign.py', 'src/saal_rsi/adaptive_registry.py',
        'src/saal_rsi/adaptive_dispatch.py', 'src/saal_rsi/adaptive_replay_v3.py',
        'src/saal_rsi/adaptive_evaluator_v3.py',
        'research/dynamic-lora-v6/metric-repair/report_study_v3.py',
        'research/dynamic-lora-v6/metric-repair/frozen_regression_v3.py'}
    validate_sources(payload, ROOT, required)
    signed_jobs = [sign(job, key) for job in jobs]
    signed_protocol = sign(payload, key)
    if verify(signed_protocol, pin) != payload or any(verify(e, pin) != j for e, j in zip(signed_jobs, jobs)):
        raise ValueError('fresh signature self-verification failed')

    output.mkdir(mode=0o700, parents=True)
    (output/'jobs').mkdir(mode=0o700)
    for run, envelope in zip(stage['runs'], signed_jobs):
        (output/'jobs'/f"{run['name']}-job.json").write_bytes(canonical(envelope))
    (output/'signed_protocol.json').write_bytes(canonical(signed_protocol))
    (output/'controller.pub').write_text(pin+'\n')
    authorization = {
        'schema': 'saal-user-authorization-record-v1',
        'request': 'you go do that',
        'scope': 'Create and sign only the newly budgeted three-run PILOT admission; execute only its admitted public engineering pilot.',
        'prior_campaign_reused': False,
        'recorded_key_fingerprint': key_fingerprint,
        'approval_source': 'direct user instruction in this conversation',
    }
    (output/'authorization-record.json').write_bytes(canonical(authorization))
    (output/'admission-manifest.json').write_bytes(canonical({
        'schema': 'saal-adaptive-pilot-admission-manifest-v1',
        'campaign_digest': sha(payload), 'signed_protocol_digest': sha(signed_protocol),
        'job_digests': [sha(job) for job in jobs], 'job_envelope_digests': [sha(e) for e in signed_jobs],
        'controller_key_fingerprint': key_fingerprint,
        'numerical_metric_policy_digest': POLICY_DIGEST,
        'colab_cli_version': draft['colab_cli_version'],
        'state': 'SIGNED_NOT_PROVISIONED',
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False,
    }))
    registry_path = output/'budget.sqlite3'
    registry = AdaptiveRegistry(registry_path, pin, signed_protocol, create=True)
    try:
        for envelope in signed_jobs:
            registry.reserve(envelope)
        registry_export = registry.records()
        if registry_export['campaign_state'] != 'PREPARED' or len(registry_export['attempts']) != len(jobs):
            raise ValueError('fresh pilot budget reservation incomplete')
        (output/'preprovision-registry.json').write_bytes(canonical(registry_export))
    finally:
        registry.close()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True, help='local owner-only controller key; never copied')
    parser.add_argument('--pin', required=True, help='pinned controller public key file')
    parser.add_argument('--out', required=True, help='new, empty campaign evidence directory')
    args = parser.parse_args()
    target = prepare(args.out, args.key, args.pin)
    print(json.dumps({'campaign': str(target), 'state': 'SIGNED_NOT_PROVISIONED',
                      'run_count': 3, 'remote_execution_started': False,
                      'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}))
