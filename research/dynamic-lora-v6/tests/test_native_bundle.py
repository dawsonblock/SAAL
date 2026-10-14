import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from saal_rsi.adaptive_artifacts import validate_bundle, file_hash
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT/'research/dynamic-lora-v6/native-engineering-05'


@pytest.fixture
def bundle(tmp_path):
    # Actual official-checkpoint engineering outputs, not fabricated results.
    name = 'adalora_12_to_8'
    pin = (NATIVE/'controller.pub').read_text().strip()
    job = verify(json.loads((NATIVE/(name+'-job.json')).read_text()), pin)
    out = tmp_path/'bundle'; shutil.copytree(NATIVE/name, out)
    return out, job, pin


def reseal_changed_files(root):
    inventory = json.loads((root/'artifact_inventory.json').read_text())
    for name in inventory['files']:
        inventory['files'][name] = {'size': (root/name).stat().st_size, 'sha256': file_hash(root/name)}
    (root/'artifact_inventory.json').write_bytes(canonical(inventory))


def test_actual_official_adalora_bundle_and_all_snapshot_replay(bundle):
    root, job, pin = bundle
    result = validate_bundle(root, job, pin, require_replay=True)
    assert result['real_parameter_updates'] and result['same_runtime_replay'] == {r: True for r in ('initial', 'best', 'terminal')}
    assert result['allocation']['active_rank_total'] <= 176
    assert result['preflight']['optimizer_updates'] == 0
    assert result['runtime']['deterministic_algorithms'] is True


@pytest.mark.parametrize('fault', ['singular_count', 'masked_tensor', 'same_step', 'source', 'snapshot',
    'optimizer_budget', 'accumulation_budget', 'frozen_weights', 'environment', 'step_type',
    'false_metrics', 'rank_pattern', 'signature', 'missing_artifact', 'promotion'])
def test_native_negative_cases_even_when_untrusted_inventory_is_rehashed(bundle, fault):
    root, job, pin = bundle
    result = json.loads((root/'result.json').read_text())
    if fault == 'singular_count':
        checkpoint = load_file(str(root/'terminal.safetensors'))
        n = next(n for n in checkpoint if '.lora_E.' in n)
        mask = json.loads((root/'terminal_rank_pattern.json').read_text())['masks'][job['adapter']['target_modules'][0]]
        active = next(i for i, flag in enumerate(mask) if flag)
        checkpoint[n][active] = 0
        save_file(checkpoint, str(root/'terminal.safetensors'))
    elif fault == 'masked_tensor':
        checkpoint = load_file(str(root/'terminal.safetensors'))
        states = json.loads((root/'terminal_rank_pattern.json').read_text())['masks']
        target = next(t for t, mask in states.items() if not all(mask))
        checkpoint['base_model.model.'+target+'.lora_E.default'][states[target].index(False)] = 1
        save_file(checkpoint, str(root/'terminal.safetensors'))
    elif fault == 'same_step':
        pairs = [(a, b) for a in ('initial', 'terminal') for b in ('best',) if result['snapshots'][a]['step'] == result['snapshots'][b]['step']]
        assert pairs
        _, b = pairs[0]
        values = load_file(str(root/(b+'.safetensors')))
        values[next(n for n in values if 'classifier.modules_to_save.default.bias' in n)][0] += .01
        save_file(values, str(root/(b+'.safetensors')))
    elif fault == 'source': result['worker_identity'] = 'sha256:'+'0'*64
    elif fault == 'snapshot':
        upstream = json.loads((root/'snapshot_manifest.json').read_text()); upstream['revision'] = 'main'
        (root/'snapshot_manifest.json').write_bytes(canonical(upstream))
    elif fault in ('optimizer_budget', 'accumulation_budget'):
        result['resource_accounting']['completed_optimizer_updates' if fault == 'optimizer_budget' else 'microbatches'] += 1
        (root/'resource_accounting.json').write_bytes(canonical(result['resource_accounting']))
    elif fault == 'frozen_weights': result['frozen_integrity']['unchanged'] = False
    elif fault == 'environment': result['runtime']['versions']['peft'] = '0.20.0'
    elif fault == 'step_type': result['snapshots']['initial']['step'] = False
    elif fault == 'false_metrics': result['snapshots']['terminal']['metrics']['accuracy'] += .001
    elif fault == 'rank_pattern':
        state = json.loads((root/'terminal_rank_pattern.json').read_text()); state['masks'][next(iter(state['masks']))][0] = not state['masks'][next(iter(state['masks']))][0]
        (root/'terminal_rank_pattern.json').write_bytes(canonical(state))
    elif fault == 'signature':
        envelope = json.loads((root/'signed_training_job.json').read_text()); envelope['signature_b64'] = 'AAAA'
        (root/'signed_training_job.json').write_bytes(canonical(envelope))
    elif fault == 'missing_artifact':
        (root/'best.safetensors').unlink()
        with pytest.raises(ValueError): validate_bundle(root, job, pin)
        return
    else: result['external_deployment_allowed'] = True
    (root/'result.json').write_bytes(canonical(result))
    reseal_changed_files(root)
    with pytest.raises(ValueError): validate_bundle(root, job, pin)


def test_forged_replay_receipt_is_rejected(bundle):
    root, job, pin = bundle
    replay = json.loads((root/'fresh_process_replay.json').read_text())
    replay['checkpoints']['terminal'][0]['probabilities'] = [float('nan')]*4
    # Use literal invalid JSON numeric constants to exercise the parser as well.
    (root/'fresh_process_replay.json').write_text(json.dumps(replay))
    with pytest.raises(ValueError): validate_bundle(root, job, pin, require_replay=True)
