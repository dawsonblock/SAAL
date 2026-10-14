import copy
import importlib.util
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from saal_rsi.model_contract import sign
from saal_rsi.spec import canonical


@pytest.fixture
def composite(tmp_path, monkeypatch, admitted_job):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project))
    spec = importlib.util.spec_from_file_location('composite_v2_test', project/'prepare_v2.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(module, 'PROJECT', tmp_path)
    folder = tmp_path/module.DIRECTORIES['PILOT']; folder.mkdir()
    (folder/'controller.pub').write_text(key.public_key().public_bytes_raw().hex())
    runs = {}
    for method in module.METHODS:
        job = copy.deepcopy(admitted_job)
        job['adapter']['method'] = method
        runs[method] = {'job': job, 'result': {'resource_accounting': {'sample_exposures': 16000}},
            'binding': {'method': method, 'job_digest': method, 'unadapted_model_hash': 'same-base-and-head',
                        'sample_stream_digest': 'same-stream'}}
    fixed = {m: runs[m] for m in module.METHODS[:2]}
    protocol = {'frozen_fixed_controls': [copy.deepcopy(fixed[m]['binding']) for m in module.METHODS[:2]],
                'prior_probe_digest': 'probe-bound'}
    (folder/'signed_protocol.json').write_bytes(canonical(sign(protocol, key)))
    monkeypatch.setattr(module, 'fixed_controls', lambda: fixed)
    monkeypatch.setattr(module, 'probe_accepted', lambda: 'probe-bound')
    monkeypatch.setattr(module, 'completed_runs', lambda *a, **kw: {'adalora-12-to-8': runs['adalora-12-to-8']})
    return module, runs, folder, key, protocol


def test_only_matched_completed_controls_pass_composite_engineering_gate(composite):
    module, runs, *_ = composite
    result = module.composite_pilot()
    assert result['accepted'] and result['qualification_status'] == 'NOT_EVALUATED'
    assert result['external_deployment_allowed'] is False
    assert len(result['bindings']) == 3


@pytest.mark.parametrize('fault', ['base', 'stream', 'seed', 'data', 'updates', 'exposures', 'regression_policy'])
def test_composite_cannot_hide_changed_conditions(composite, fault):
    module, runs, *_ = composite
    ada = runs['adalora-12-to-8']
    if fault == 'base': ada['binding']['unadapted_model_hash'] = 'substituted'
    elif fault == 'stream': ada['binding']['sample_stream_digest'] = 'different'
    elif fault == 'seed': ada['job']['dropout_seed'] += 1
    elif fault == 'data': ada['job']['dataset_digest'] = 'different'
    elif fault == 'updates': ada['job']['steps'] = 999
    elif fault == 'exposures': ada['result']['resource_accounting']['sample_exposures'] = 15999
    else: ada['job']['selection_policy'] = 'protected-holdout-selection'
    with pytest.raises(ValueError): module.composite_pilot()


def test_composite_must_match_controls_frozen_before_new_run(composite):
    module, runs, folder, key, protocol = composite
    protocol['frozen_fixed_controls'][0]['job_digest'] = 'replaced-after-training'
    (folder/'signed_protocol.json').write_bytes(canonical(sign(protocol, key)))
    with pytest.raises(ValueError, match='preregistration'): module.composite_pilot()


def test_probe_receipt_is_bound_to_full_pilot_admission(composite):
    module, runs, folder, key, protocol = composite
    protocol['prior_probe_digest'] = 'different-probe'
    (folder/'signed_protocol.json').write_bytes(canonical(sign(protocol, key)))
    with pytest.raises(ValueError, match='probe changed'): module.composite_pilot()
