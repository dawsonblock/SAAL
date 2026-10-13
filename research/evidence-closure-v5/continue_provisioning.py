"""One explicit pre-training provisioning continuation; reuse original reservations."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'src'))
import campaign
from saal_rsi.spec import canonical, sha
from saal_rsi.model_contract import sign, verify
from saal_rsi.attempt_service import private_key
from saal_rsi.epoch_registry import EpochExperimentRegistry
from saal_rsi.epoch_backend_v6 import setup_plan
from saal_rsi.model_monitor_v3 import execute_colab_plan


def context():
    experiment = PROJECT/'experiment'
    pin = (experiment/'controller.pub').read_text().strip()
    protocol = verify(json.loads((experiment/'signed_protocol.json').read_text()), pin)
    if protocol['source_files'] != campaign.source_map():
        raise ValueError('registered learning/controller source changed')
    state = json.loads((experiment/'colab/monitor-state.json').read_text())
    if (state['terminal'] is not True or state['cleanup_verified'] is not True
            or state['error']['type'] != 'RuntimeError' or 'Service Unavailable' not in state['error']['detail']
            or (experiment/'colab/outcomes.json').exists()):
        raise ValueError('only failed provisioning before any coordinator run is continuable')
    ledger = EpochExperimentRegistry(experiment/'budget.sqlite3', pin)
    try: records = ledger.records()
    finally: ledger.close()
    if [r['digest'] for r in records['attempts']] != [r['job_digest'] for r in protocol['runs']] or any(r['state'] != 'RESERVED' for r in records['attempts']):
        raise ValueError('a started job cannot use a pre-training continuation')
    return experiment, pin, protocol, state, records


def admission_body():
    experiment, pin, protocol, state, records = context()
    return {'schema': 'saal-explicit-pre-training-provisioning-continuation-v1',
        'protocol_digest': sha(protocol), 'previous_monitor_digest': sha(state),
        'previous_registry_digest': sha(records), 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'job_digests': [r['job_digest'] for r in protocol['runs']], 'additional_training_reservations': 0,
        'training_retries': 0, 'additional_provisioning_attempts': 1, 'budget_registry_reused': True,
        'source_parameters_changed': False, 'qualification_status': 'NOT_EVALUATED', 'external_deployment_allowed': False}


def prepare(key_path):
    body = admission_body()
    key = private_key(key_path)
    pin = (PROJECT/'experiment/controller.pub').read_text().strip()
    if key.public_key().public_bytes_raw().hex() != pin:
        raise PermissionError('controller identity')
    with (PROJECT/'experiment/signed_provisioning_continuation.json').open('xb') as out:
        out.write(canonical(sign(body, key)))


def run():
    experiment, pin, protocol, _, _ = context()
    if verify(json.loads((experiment/'signed_provisioning_continuation.json').read_text()), pin) != admission_body():
        raise ValueError('continuation/source substitution')
    root = experiment/'provisioning-continuation'
    plan = setup_plan(root, protocol['budget']['lifecycle_seconds'])
    config = {'workspace': str(root.resolve()), 'session': plan['session'], 'pin': pin,
        'jobs': [str((experiment/(r['name']+'-job.json')).resolve()) for r in protocol['runs']],
        'artifacts': str((experiment/'artifacts').resolve()), 'ledger': str((experiment/'budget.sqlite3').resolve())}
    (experiment/'artifacts').mkdir(exist_ok=True)
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
