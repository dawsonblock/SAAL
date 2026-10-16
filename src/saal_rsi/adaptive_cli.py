"""Inspect, validate and reproduce bounded adaptive artifacts; no promotion."""
import argparse
import json
from pathlib import Path

from . import __version__
from .adaptive_artifacts import read_json
from .adaptive_dispatch import validate_job, validate_bundle
from .adaptive_campaign import fresh_local_replay
from .model_contract import verify
from .spec import canonical, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('inspect', 'validate', 'replay'))
    p.add_argument('--job', required=True); p.add_argument('--public-key', required=True)
    p.add_argument('--bundle'); p.add_argument('--cache'); p.add_argument('--report')
    a = p.parse_args()
    job = validate_job(verify(read_json(a.job), a.public_key))
    if a.action == 'inspect':
        value = {'research_version': __version__, 'job_digest': sha(job), 'adapter': job['adapter'],
                 'steps': job['steps'], 'sample_exposures': job['steps']*job['batch_size']*job['accumulation'],
                 'dataset_digest': job['dataset_digest'], 'qualification_status': 'NOT_EVALUATED',
                 'external_deployment_allowed': False}
    elif a.action == 'validate':
        result = validate_bundle(a.bundle, job, a.public_key, require_replay=True)
        value = {'job_digest': sha(job), 'artifact_and_replay_evidence_validated': True,
                 'allocation': result['allocation'], 'qualification_status': 'NOT_EVALUATED'}
    else:
        value = fresh_local_replay(a.bundle, job, a.public_key, a.cache)
        if not a.report: raise ValueError('fresh replay requires a new report path')
        with Path(a.report).open('xb') as f: f.write(canonical(value))
        if value['verified'] is not True: raise ValueError('fresh local replay failed')
    print(json.dumps(value, allow_nan=False, indent=2))


if __name__ == '__main__': main()
