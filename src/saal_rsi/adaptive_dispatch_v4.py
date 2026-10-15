"""Closed v4 dispatch; v1-v3 worker contracts remain unchanged."""


def validate_job(job):
    from .adaptive_contract_v4 import validate_job as validate
    return validate(job)


def validate_bundle(root, job, pin, *, require_replay=False):
    validate_job(job)
    from .adaptive_artifacts_v4 import validate_bundle as validate
    return validate(root, job, pin, require_replay=require_replay)
