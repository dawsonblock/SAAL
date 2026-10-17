"""Fresh v2-metric campaign admission checks; public research only."""
from pathlib import Path
import re

from .adaptive_contract_v4 import validate_job, worker_identity
from .adaptive_runtime_v4 import validate_runtime_requirements
from .numerical_metrics_v2 import POLICY_DIGEST, validate_policy
from .model_contract import verify
from .spec import canonical, sha


def validate_protocol(protocol):
    """Validate a *new* v4 protocol without spending or creating a budget."""
    protocol_fields = {
        'schema', 'stage', 'budget', 'job_digests', 'runs', 'worker_identities',
        'numerical_metric_policy', 'numerical_metric_policy_digest', 'colab_cli_version',
        'source_files', 'required_source_files', 'independent_generalization',
        'qualification_status', 'external_deployment_allowed', 'runtime_requirements',
        'data_review_status', 'scope', 'signer_status',
    }
    if (type(protocol) is not dict or set(protocol) != protocol_fields
            or protocol.get('schema') != 'saal-adaptive-campaign-v4'
            or protocol.get('qualification_status') != 'NOT_EVALUATED'
            or protocol.get('external_deployment_allowed') is not False
            or protocol.get('worker_identities') != {'v4': worker_identity()}
            or protocol.get('numerical_metric_policy_digest') != POLICY_DIGEST
            or type(protocol.get('colab_cli_version')) is not str
            or not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', protocol['colab_cli_version'])):
        raise ValueError('fresh v4 public campaign, metric policy and worker identity required')
    validate_policy(protocol.get('numerical_metric_policy'))
    budget = protocol.get('budget')
    if (type(budget) is not dict
            or set(budget) != {'campaign_id', 'max_runs', 'total_lifecycle_seconds',
                               'max_provisioning_attempts', 'automatic_training_retries',
                               'financial_cap', 'max_compute_units'}
            or type(budget['campaign_id']) is not str
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', budget['campaign_id'])
            or type(budget['max_runs']) is not int or not 1 <= budget['max_runs'] <= 25
            or type(budget['total_lifecycle_seconds']) is not int
            or not 30 <= budget['total_lifecycle_seconds'] <= 100000
            or type(budget['max_provisioning_attempts']) is not int
            or budget['max_provisioning_attempts'] != 1
            or type(budget['automatic_training_retries']) is not int
            or budget['automatic_training_retries'] != 0
            or budget['financial_cap'] is not False
            or type(budget['max_compute_units']) is not int
            or not 1 <= budget['max_compute_units'] <= 100):
        raise ValueError('one-provision, no-retry bounded public budget required')
    validate_runtime_requirements(protocol.get('runtime_requirements'))
    if (protocol.get('data_review_status') != 'UNREVIEWED_PUBLIC_DIAGNOSTIC'
            or protocol.get('scope') != 'ENGINEERING_ONLY_NO_BEHAVIORAL_CLAIM'
            or protocol.get('signer_status') != 'SELF_SIGNED_DEVELOPMENT_ONLY_NOT_INDEPENDENT'):
        raise ValueError('unreviewed data may be used only for a bounded engineering diagnostic')
    runs = protocol.get('runs')
    if type(runs) is not list or not 1 <= len(runs) <= budget['max_runs']:
        raise ValueError('bounded exact run descriptors required')
    names, conditions, digests = set(), set(), []
    for run in runs:
        if (type(run) is not dict or set(run) != {'name', 'method', 'seed', 'job_digest'}
                or type(run['name']) is not str
                or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run['name'])
                or run['name'] in names
                or run['method'] not in ('fixed-r8', 'fixed-r12', 'adalora-static-r12', 'adalora-12-to-8')
                or type(run['seed']) is not int or not 0 <= run['seed'] <= 2**31-1
                or type(run['job_digest']) is not str
                or not re.fullmatch(r'sha256:[0-9a-f]{64}', run['job_digest'])
                or (run['seed'], run['method']) in conditions):
            raise ValueError('unique exact v4 method, seed and run identities required')
        names.add(run['name'])
        conditions.add((run['seed'], run['method']))
        digests.append(run['job_digest'])
    if (protocol.get('job_digests') != digests
            or len(set(digests)) != len(digests)
            or protocol.get('stage') not in ('ENGINEERING', 'PILOT', 'STUDY', 'DATA_CONTROL')):
        raise ValueError('ordered job digest / stage identity mismatch')
    sources = protocol.get('source_files')
    required = protocol.get('required_source_files')
    if (type(sources) is not dict or not sources or type(required) is not list or not required
            or any(type(item) is not str or not item for item in required)
            or len(set(required)) != len(required) or not set(required) <= set(sources)):
        raise ValueError('explicit v4 controller and analysis source identities required')
    for name, digest in sources.items():
        if (type(name) is not str or not name or Path(name).is_absolute() or '..' in Path(name).parts
                or type(digest) is not str or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('safe relative source digest inventory required')
    if protocol.get('independent_generalization') != 'NOT_ESTABLISHED':
        raise ValueError('public campaign cannot claim independent generalization')
    return protocol


def validate_jobs(protocol, envelopes, pin):
    """Verify all fresh signed v4 jobs and exact protocol bindings."""
    validate_protocol(protocol)
    if type(envelopes) is not list or len(envelopes) != len(protocol['runs']):
        raise ValueError('complete ordered signed v4 job set required')
    jobs = []
    for run, envelope in zip(protocol['runs'], envelopes):
        job = validate_job(verify(envelope, pin))
        budget = job['budget']
        if (sha(job) != run['job_digest'] or job['adapter']['method'] != run['method']
                or job['seed'] != run['seed'] or job['campaign_stage'] != protocol['stage']
                or budget['campaign_id'] != protocol['budget']['campaign_id']
                or budget['max_runs'] != protocol['budget']['max_runs']
                or budget['total_lifecycle_seconds'] != protocol['budget']['total_lifecycle_seconds']
                or budget['automatic_training_retries'] != 0
                or budget['financial_cap'] is not False
                or canonical(job['runtime_requirements']) != canonical(protocol['runtime_requirements'])
                or job['numerical_metric_policy_digest'] != protocol['numerical_metric_policy_digest']
                or canonical(job['numerical_metric_policy']) != canonical(protocol['numerical_metric_policy'])):
            raise ValueError('signed v4 job does not match protocol admission')
        jobs.append(job)
    return jobs


def validate_sources(protocol, root):
    """Verify admitted controller/analysis bytes before campaign execution."""
    from .adaptive_artifacts import file_hash
    validate_protocol(protocol)
    root = Path(root).resolve()
    for name, expected in protocol['source_files'].items():
        path = root/name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or file_hash(path) != expected):
            raise ValueError('admitted v4 controller/analysis source changed: '+name)
    return True
