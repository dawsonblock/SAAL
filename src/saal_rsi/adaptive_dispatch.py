"""Allowlisted controller dispatch; signed v1/v2 identities never substitute."""


def validate_job(job):
    from .adaptive_contract import validate_job as v1
    from .adaptive_contract_v2 import validate_job as v2
    if type(job) is not dict:
        raise ValueError('versioned adaptive job required')
    if job.get('schema') == 'saal-adaptive-job-v1':
        return v1(job)
    if job.get('schema') == 'saal-adaptive-job-v2':
        return v2(job)
    raise ValueError('unknown adaptive job version')


def validate_bundle(root, job, pin, *, require_replay=False):
    validate_job(job)
    if job['schema'] == 'saal-adaptive-job-v1':
        from .adaptive_artifacts import validate_bundle as validate
    else:
        from .adaptive_artifacts_v2 import validate_bundle as validate
    return validate(root, job, pin, require_replay=require_replay)
