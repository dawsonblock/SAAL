"""Final bounded provisioning attempt after CPU QA teardown. No training reset."""
from pathlib import Path
import hashlib
import json
import sys
import continue_provisioning as original
from saal_rsi.spec import canonical, sha
from saal_rsi.model_contract import sign, verify
from saal_rsi.attempt_service import private_key
from saal_rsi.epoch_backend_v6 import setup_plan
from saal_rsi.model_monitor_v3 import execute_colab_plan

PROJECT, ROOT = original.PROJECT, original.ROOT


def body():
    experiment, pin, protocol, state, records = original.context()
    prior = json.loads((experiment/'provisioning-continuation/monitor-state.json').read_text())
    qa = json.loads((PROJECT/'colab-tests/monitor-state.json').read_text())
    result = json.loads((PROJECT/'colab-tests/evidence/result.json').read_text())
    if (prior['terminal'] is not True or prior['cleanup_verified'] is not True
            or 'Service Unavailable' not in prior['error']['detail']
            or (experiment/'provisioning-continuation/outcomes.json').exists()):
        raise ValueError('only confirmed pre-training provisioning failure can continue')
    if qa['terminal'] is not True or qa['cleanup_verified'] is not True or any(result['returncodes'].values()):
        raise ValueError('CPU test runtime must be successfully stopped first')
    return {'schema': 'saal-final-provisioning-after-qa-admission-v1',
        'protocol_digest': sha(protocol), 'original_monitor_digest': sha(state),
        'continuation_monitor_digest': sha(prior), 'registry_digest': sha(records),
        'qa_state_digest': sha(qa), 'qa_result_digest': sha(result),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'previous_driver_sha256': hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        'job_digests': [r['job_digest'] for r in protocol['runs']],
        'additional_training_reservations': 0, 'training_retries': 0,
        'additional_provisioning_attempts': 1, 'no_further_provisioning_in_this_batch': True,
        'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}


def prepare(path):
    payload = body()
    key = private_key(path)
    pin = (PROJECT/'experiment/controller.pub').read_text().strip()
    if key.public_key().public_bytes_raw().hex() != pin: raise PermissionError('controller identity')
    with (PROJECT/'experiment/signed_final_provisioning.json').open('xb') as out:
        out.write(canonical(sign(payload, key)))


def run():
    experiment, pin, protocol, _, _ = original.context()
    if verify(json.loads((experiment/'signed_final_provisioning.json').read_text()), pin) != body():
        raise ValueError('final source/admission substitution')
    root = experiment/'provisioning-after-qa'
    plan = setup_plan(root, protocol['budget']['lifecycle_seconds'])
    config = {'workspace': str(root.resolve()), 'session': plan['session'], 'pin': pin,
        'jobs': [str((experiment/(r['name']+'-job.json')).resolve()) for r in protocol['runs']],
        'artifacts': str((experiment/'artifacts').resolve()), 'ledger': str((experiment/'budget.sqlite3').resolve())}
    (root/'coordinator.json').write_bytes(canonical(config))
    launcher = root/'coordinator.py'
    launcher.write_text('import sys\nsys.path.insert(0,'+repr(str(ROOT/'src'))+')\n'
        'from saal_rsi.epoch_backend_v6 import coordinator\ncoordinator('+repr(str((root/'coordinator.json').resolve()))+')\n')
    plan['commands'] += [[sys.executable, '-I', str(launcher)], ['colab', 'stop', '-s', plan['session']]]
    plan['phases'] += ['CAMPAIGN']
    (root/'submission-plan.json').write_bytes(canonical(plan))
    return execute_colab_plan(plan, root)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare(sys.argv[2])
    elif sys.argv[1] == 'run': run()
    else: raise ValueError('prepare or run required')
