"""Exercise the v3 loop with a tiny test model, not a research campaign."""
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from saal_rsi import adaptive_training_v3 as training
from saal_rsi.pretrained_contract import DECISIONS


class TestBackend:
    __test__ = False

    def __init__(self):
        torch.manual_seed(61)
        self.model = torch.nn.Linear(4, 4)
        self.trainable = dict(self.model.named_parameters())
        self.initial_trainable_hash = self.trainable_hash()
        self.unadapted_model_hash = 'test-only-unadapted'

    def trainable_hash(self):
        return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in self.trainable.values())).hexdigest()

    def inputs(self, rows):
        return torch.eye(4)[[DECISIONS.index(r['expected_decision']) for r in rows]]

    def batch_objective(self, rows, weights):
        target = torch.tensor([DECISIONS.index(r['expected_decision']) for r in rows])
        loss = torch.nn.functional.cross_entropy(self.model(self.inputs(rows)), target, weight=weights)
        return loss, len(rows)*4, {'cross_entropy': float(loss.detach()), 'orthogonal_penalty': 0.}

    def evaluate(self, rows):
        self.model.eval()
        with torch.no_grad():
            values = torch.softmax(self.model(self.inputs(rows)), dim=-1).tolist()
        return [{'decision': DECISIONS[max(range(4), key=p.__getitem__)], 'probabilities': p} for p in values]

    def export_checkpoint(self, path):
        save_file(self.model.state_dict(), str(path))

    def restore_checkpoint(self, path, state):
        self.model.load_state_dict(load_file(str(path)))

    def verify_frozen_weights(self):
        return {'unchanged': True, 'scope': 'test fixture has no backbone'}

    def report_resources(self):
        return {'trainable_parameters': 20}


class TestRankController:
    __test__ = False

    def __init__(self, model, adapter):
        self.state = {'step': 0}

    def complete_update(self, step, **kwargs):
        assert kwargs['optimizer_updated'] is True
        self.state['step'] = step
        return {'aggregate_active_rank': 0, 'completed_optimizer_step': step}

    def require_completion(self, steps):
        assert self.state['step'] == steps
        return {'completed_optimizer_updates': steps}


def job():
    records = []
    for split in ('train', 'development'):
        for i, decision in enumerate(DECISIONS):
            records.append({'record_id': split+str(i), 'task_family_id': split+str(i//2),
                'cluster_id': split+'pair'+str(i//2), 'category': 'category',
                'expected_decision': decision, 'split': split})
    return {'dataset': {'records': records}, 'max_seconds': 30, 'steps': 4,
        'batch_size': 4, 'accumulation': 2, 'checkpoint_interval': 2,
        'sampler_seed': 61, 'adapter': {'method': 'fixed-r8'}, 'model_spec': {'precision': 'float32'},
        'head_learning_rate': .01, 'learning_rate': .01, 'class_balance': False,
        'selection_policy': 'public-guardrails-pair-family-v1', 'worker_identity': 'test-only',
        'replay_tolerance': 1e-4}


def test_v3_loop_emits_new_metrics_and_replays_every_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'RankController', TestRankController)
    monkeypatch.setattr(training, 'sample_ids', lambda records, policy, seed, count:
        [records[i % len(records)]['record_id'] for i in range(count)])
    backend = TestBackend()
    result = training.train(backend, job(), tmp_path/'output')
    assert result['schema'] == 'saal-adaptive-result-v3'
    assert result['real_parameter_updates'] is True
    assert result['same_runtime_replay'] == {r: True for r in ('initial', 'best', 'terminal')}
    assert result['resource_accounting']['completed_optimizer_updates'] == 4
    assert result['resource_accounting']['microbatches'] == 8
    assert result['resource_accounting']['sample_exposures'] == 32
    for snapshot in result['snapshots'].values():
        assert snapshot['metrics']['numerical_metric_schema'] == 'saal-classification-diagnostics-v1'
        assert type(snapshot['metrics']['brier_score']) is str
    selection = json.loads((tmp_path/'output/checkpoint_selection.json').read_text())
    assert [row['step'] for row in selection['history']] == [0, 2, 4]
    assert all(type(row['metrics']['negative_log_likelihood']) is str for row in selection['history'])


def test_v3_loop_interrupts_without_rank_advancement_on_skipped_update(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'RankController', TestRankController)
    monkeypatch.setattr(training, 'sample_ids', lambda records, policy, seed, count:
        [records[i % len(records)]['record_id'] for i in range(count)])
    monkeypatch.setattr(training, 'optimizer_update', lambda *args, **kwargs:
        (False, {'finite': False, 'norm': 0.0, 'gradient_tensors': 0}))
    with pytest.raises(ValueError, match='optimizer did not complete'):
        training.train(TestBackend(), job(), tmp_path/'output')
    assert not (tmp_path/'output/rank_history.jsonl').exists()
    assert not (tmp_path/'output/terminal.safetensors').exists()


def test_v3_loop_rejects_regression_records_before_training(tmp_path):
    value = job(); value['dataset']['records'][-1]['split'] = 'regression'
    with pytest.raises(ValueError, match='must not receive regression'):
        training.train(TestBackend(), value, tmp_path/'output')
