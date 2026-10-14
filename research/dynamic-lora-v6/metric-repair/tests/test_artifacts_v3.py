"""Wire-format regression fixtures are not new experiments or admissions."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from saal_rsi.adaptive_artifacts import validate_bundle as legacy_bundle
from saal_rsi.adaptive_artifacts_v3 import validate_bundle, file_hash
from saal_rsi.adaptive_contract import worker_identity as v1_identity, validate_job as v1_job
from saal_rsi.adaptive_contract_v2 import worker_identity as v2_identity, validate_job as v2_job
from saal_rsi.adaptive_contract_v3 import SCHEMA, RANKNUM_POLICY, validate_job, worker_identity
from saal_rsi.model_contract import sign, verify
from saal_rsi.numerical_metrics_v1 import POLICY, POLICY_DIGEST, diagnostics
from saal_rsi.spec import canonical, sha

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT/'research/dynamic-lora-v6'
NATIVE = PROJECT/'native-engineering-05'


def read(path):
    return json.loads(path.read_text())


def reseal(root):
    inventory = read(root/'artifact_inventory.json')
    for name in inventory['files']:
        inventory['files'][name] = {'size': (root/name).stat().st_size, 'sha256': file_hash(root/name)}
    (root/'artifact_inventory.json').write_bytes(canonical(inventory))
    replay = read(root/'fresh_process_replay.json')
    replay['inventory_sha256'] = file_hash(root/'artifact_inventory.json')
    (root/'fresh_process_replay.json').write_bytes(canonical(replay))


@pytest.fixture
def wire_bundle(tmp_path):
    # Re-encode ONLY a temporary test copy with an ephemeral test identity.
    # This exercises schema/validator plumbing; it is NOT v3 training evidence.
    old_pin = (NATIVE/'controller.pub').read_text().strip()
    old = verify(read(NATIVE/'adalora_12_to_8-job.json'), old_pin)
    out = tmp_path/'wire_fixture'
    shutil.copytree(NATIVE/'adalora_12_to_8', out)
    key = Ed25519PrivateKey.generate()
    pin = key.public_key().public_bytes_raw().hex()
    job = copy.deepcopy(old)
    job.update(schema=SCHEMA, ranknum_policy=RANKNUM_POLICY, worker_identity=worker_identity(),
               numerical_metric_policy=copy.deepcopy(POLICY), numerical_metric_policy_digest=POLICY_DIGEST)
    # Only stored snapshots can be reconstructed without rerunning a model.
    # Give this synthetic wire fixture a terminal-only measurement schedule.
    job['checkpoint_interval'] = 4
    validate_job(job)
    (out/'signed_training_job.json').write_bytes(canonical(sign(job, key)))
    result = read(out/'result.json')
    result.update(schema='saal-adaptive-result-v3', job_digest=sha(job), worker_identity=worker_identity(),
                  ranknum_policy=RANKNUM_POLICY, ranknum_normalized=0,
                  numerical_metric_policy=copy.deepcopy(POLICY), numerical_metric_policy_digest=POLICY_DIGEST,
                  trust='TEST_FIXTURE_REENCODED_NOT_AN_EXPERIMENT')
    development = [r for r in job['dataset']['records'] if r['split'] == 'development']
    selection = read(out/'checkpoint_selection.json')
    selection['history'] = [r for r in selection['history'] if r['step'] in (0, 4)]
    by_step = {}
    for snapshot in result['snapshots'].values():
        snapshot['metrics'] = diagnostics(development, snapshot['predictions'])
        by_step[snapshot['step']] = snapshot['metrics']
    for row in selection['history']:
        # This wire fixture preserves every registered measurement checkpoint.
        assert row['step'] in by_step
        row['metrics'] = by_step[row['step']]
    (out/'checkpoint_selection.json').write_bytes(canonical(selection))
    (out/'result.json').write_bytes(canonical(result))
    sampler = read(out/'sampler_manifest.json'); sampler['job_digest'] = sha(job)
    (out/'sampler_manifest.json').write_bytes(canonical(sampler))
    inventory = read(out/'artifact_inventory.json'); inventory['schema'] = 'saal-adaptive-artifact-inventory-v3'
    (out/'artifact_inventory.json').write_bytes(canonical(inventory))
    replay = read(out/'fresh_process_replay.json')
    replay.update(schema='saal-adaptive-fresh-replay-v3', job_digest=sha(job),
                  numerical_metric_policy_digest=POLICY_DIGEST)
    (out/'fresh_process_replay.json').write_bytes(canonical(replay))
    reseal(out)
    return out, job, pin


def test_exact_v3_wire_bundle_all_three_snapshots(wire_bundle):
    out, job, pin = wire_bundle
    result = validate_bundle(out, job, pin, require_replay=True)
    assert result['trust'] == 'TEST_FIXTURE_REENCODED_NOT_AN_EXPERIMENT'
    assert all(type(s['metrics']['brier_score']) is str for s in result['snapshots'].values())
    with pytest.raises(ValueError): v1_job(job)
    with pytest.raises(ValueError): v2_job(job)


@pytest.mark.parametrize('fault', ['policy', 'digest', 'brier_last_digit', 'brier_float',
    'count_bool', 'accuracy', 'probability', 'label', 'inventory', 'worker', 'replay_policy',
    'promotion', 'frozen', 'ranknum_fixed', 'history'])
def test_v3_rejects_changes_even_with_rehashed_unsigned_inventory(wire_bundle, fault):
    out, job, pin = wire_bundle
    result = read(out/'result.json')
    terminal = result['snapshots']['terminal']
    if fault == 'policy': result['numerical_metric_policy']['rounding'] = 'UP'
    elif fault == 'digest': result['numerical_metric_policy_digest'] = 'sha256:'+'0'*64
    elif fault == 'brier_last_digit':
        value = terminal['metrics']['brier_score']; terminal['metrics']['brier_score'] = value[:-1]+str((int(value[-1])+1)%10)
    elif fault == 'brier_float': terminal['metrics']['brier_score'] = float(terminal['metrics']['brier_score'])
    elif fault == 'count_bool': terminal['metrics']['invalid_outputs'] = False
    elif fault == 'accuracy': terminal['metrics']['accuracy'] += .001
    elif fault == 'probability': terminal['predictions'][0]['probabilities'][0] += .01
    elif fault == 'label': terminal['predictions'][0]['decision'] = 'FORBIDDEN'
    elif fault == 'inventory':
        inv = read(out/'artifact_inventory.json'); inv['schema'] = 'saal-adaptive-artifact-inventory-v1'
        (out/'artifact_inventory.json').write_bytes(canonical(inv))
    elif fault == 'worker': result['worker_identity'] = 'sha256:'+'0'*64
    elif fault == 'replay_policy':
        replay = read(out/'fresh_process_replay.json'); replay['numerical_metric_policy_digest'] = 'sha256:'+'0'*64
        (out/'fresh_process_replay.json').write_bytes(canonical(replay))
    elif fault == 'promotion': result['external_deployment_allowed'] = True
    elif fault == 'frozen': result['frozen_integrity']['unchanged'] = False
    elif fault == 'ranknum_fixed':
        # An AdaLoRA test job with an unregistered adapter still fails admission.
        job['adapter']['method'] = 'fixed-r8'
    elif fault == 'history':
        selection = read(out/'checkpoint_selection.json'); selection['history'][0]['metrics']['accuracy'] += .001
        (out/'checkpoint_selection.json').write_bytes(canonical(selection))
    (out/'result.json').write_bytes(canonical(result)); reseal(out)
    with pytest.raises(ValueError): validate_bundle(out, job, pin, require_replay=True)


def test_original_interrupted_bundle_still_rejects_exactly():
    study = PROJECT/'study-v2'
    pin = (study/'controller.pub').read_text().strip()
    job = verify(read(study/'fixed_r12_157-job.json'), pin)
    bundle = study/'artifacts/run-10-6e8b0b526bfc'
    with pytest.raises(ValueError, match='snapshot predictions / metrics consistency'):
        legacy_bundle(bundle, job, pin, require_replay=True)
    with pytest.raises(ValueError): validate_bundle(bundle, job, pin, require_replay=True)


def test_frozen_bytes_workers_and_spent_states_unchanged():
    preservation = read(PROJECT/'metric-repair/PRESERVATION.json')
    for name, digest in preservation['frozen_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    assert v1_identity() == preservation['legacy_worker_identities']['v1']
    assert v2_identity() == preservation['legacy_worker_identities']['v2']
    db_path = PROJECT/'study-v2/budget.sqlite3'
    with sqlite3.connect('file:'+str(db_path)+'?mode=ro', uri=True) as db:
        assert dict(db.execute('SELECT state,COUNT(*) FROM attempts GROUP BY state')) == preservation['historical_attempt_counts']
        assert db.execute('SELECT state,provisions FROM budget').fetchall() == [('FAILED', 1)]
    foundation = read(PROJECT/'FOUNDATION.json')
    with sqlite3.connect('file:'+str(ROOT/'research/evidence-closure-v5/experiment/budget.sqlite3')+'?mode=ro', uri=True) as db:
        assert [{'digest': d, 'state': s} for d, s in db.execute('SELECT digest,state FROM attempts ORDER BY rowid')] == foundation['preserved_v5_jobs']
    for name, digest in foundation['protected_v5_files'].items():
        assert file_hash(ROOT/name) == digest
