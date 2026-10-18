import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from saal_rsi.adaptive_backend_v2_v4 import CudaSafeAdaLoRABackend
from saal_rsi.adaptive_backend_v5 import StaticPilotBackendV5
from saal_rsi.adaptive_contract_v4 import adapter_spec
from saal_rsi.adaptive_contract_v5 import (
    build_fixed_r8_reference,
    scientific_source_digest,
)
from saal_rsi.adaptive_dataset import training_projection
from saal_rsi.adaptive_static_pilot_v5 import _ISSUER, _ExecutionPermit
from saal_rsi.adaptive_training_v5 import (
    _checkpoint_selection_key,
    _nonfinite_gradient_summary,
    _optimizer_completed,
    _train_static_pilot,
)
from saal_rsi.dynamic_lora_v5.svd_reference import orthogonal_penalty_reference
from saal_rsi.epoch_sampler import sample_ids
from saal_rsi.numerical_metrics_v2 import POLICY_DIGEST
from saal_rsi.pretrained_contract import model_spec, targets
from saal_rsi.spec import sha


def test_static_pilot_inherits_exact_cuda_ranknum_normalization():
    assert StaticPilotBackendV5.__bases__ == (CudaSafeAdaLoRABackend,)


def test_nonfinite_gradient_summary_identifies_only_bad_tensor_elements():
    good = torch.nn.Parameter(torch.tensor([1.0]))
    bad = torch.nn.Parameter(torch.tensor([float('nan'), 2.0, float('inf')]))
    good.grad = torch.tensor([0.5])
    bad.grad = torch.tensor([float('nan'), 2.0, float('inf')])
    assert _nonfinite_gradient_summary([('good', good), ('bad', bad)]) == [{
        'name': 'bad', 'nonfinite_elements': 2, 'total_elements': 3,
        'finite_max_abs': 2.0,
    }]


class TinySVDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones((1, 1)))
        self.e = torch.nn.Parameter(torch.zeros((1, 1)))
        self.b = torch.nn.Parameter(torch.tensor([[0.1], [-0.1], [0.2], [-0.2]]))
        self.bias = torch.nn.Parameter(torch.zeros(4))

    def forward(self, inputs):
        hidden = (inputs @ self.a.T) * self.e.reshape(1, 1)
        return hidden @ self.b.T + self.bias


class TinyBackend:
    """Lifecycle-only fixture; not a behavioral model or scientific result."""

    def __init__(self, config):
        self.config = config
        self.spec = SimpleNamespace(as_dict=lambda: config['identity']['model_spec'])
        self.seed = config['identity']['seed']
        self.model = TinySVDModel()
        self.adapter = adapter_spec('fixed-r8')
        self.trainable = {
            'base_model.model.target.lora_A.default': self.model.a,
            'base_model.model.target.lora_E.default': self.model.e,
            'base_model.model.target.lora_B.default': self.model.b,
            'base_model.model.classifier.modules_to_save.default.bias': self.model.bias,
        }
        self.allocation_calls = 0

    def orthogonal_penalty(self):
        return self.model.a.new_zeros(())

    @staticmethod
    def _features(records):
        return torch.tensor([
            [(sum(record['prompt'].encode('utf-8')) % 17 + 1) / 17]
            for record in records
        ], dtype=torch.float32)

    def batch_objective(self, records, class_weights=None):
        labels = torch.tensor([
            ('CHEAP', 'FAST', 'ASK', 'REFUSE').index(record['expected_decision'])
            for record in records
        ])
        cross_entropy = torch.nn.functional.cross_entropy(
            self.model(self._features(records)), labels, weight=class_weights)
        self._cross_entropy_for_diagnostics = cross_entropy
        return cross_entropy, len(records), {
            'cross_entropy': float(cross_entropy.detach()),
            'orthogonal_penalty': 0.0,
            'weighted_orthogonal_penalty': 0.0,
        }

    def evaluate(self, records):
        with torch.inference_mode():
            logits = self.model(self._features(records))
            probabilities = logits.softmax(-1).tolist()
        labels = ('CHEAP', 'FAST', 'ASK', 'REFUSE')
        return [{
            'decision': labels[max(range(4), key=lambda i: row[i])],
            'probabilities': row,
        } for row in probabilities]

    def effective_update_records(self):
        values = {
            'A': self.model.a.detach().clone(),
            'E': self.model.e.detach().clone(),
            'B': self.model.b.detach().clone(),
            'scaling': 1.0,
            'ranknum': 1.0,
        }
        return {target: copy.deepcopy(values) for target in self.adapter['target_modules']}

    def export_checkpoint(self, path):
        save_file({name: value.detach().cpu().contiguous()
                   for name, value in self.trainable.items()}, str(path))

    def restore_checkpoint(self, path, rank_state):
        assert rank_state['method'] == 'fixed-r8'
        restored = load_file(str(path), device='cpu')
        with torch.no_grad():
            for name, parameter in self.trainable.items():
                parameter.copy_(restored[name])

    def verify_frozen_weights(self):
        return {'unchanged': True, 'scope': 'fixture has no frozen weights'}


def _pilot_identity(dataset):
    spec = model_spec('answerdotai/ModernBERT-base', method='none',
                      backend='local', precision='float32',
                      budget='dynamic-lora-v5-pilot')
    lock = Path('configs/dependency-locks/pretrained-macos-arm64-py312.txt').read_bytes()
    seed, sampler_seed, batch_size, accumulation = 41, 43, 8, 2
    records = [row for row in dataset['records'] if row['split'] == 'train']
    stream = sample_ids(records, 'complete-balanced-epochs-v1', sampler_seed,
                        200 * batch_size * accumulation)
    return {
        'model_spec': spec,
        'model_digest': spec['model_files_digest'],
        'tokenizer_digest': spec['tokenizer_digest'],
        'dataset_digest': sha(dataset),
        'source_digest': scientific_source_digest(),
        'dependency_digest': 'sha256:' + hashlib.sha256(lock).hexdigest(),
        'metric_policy_digest': POLICY_DIGEST,
        'sample_order_digest': sha(stream),
        'seed': seed,
        'sampler_seed': sampler_seed,
        'dropout_seed': 47,
        'batch_size': batch_size,
        'accumulation': accumulation,
        'adapter_learning_rate': 0.0003,
        'head_learning_rate': 0.0003,
        'weight_decay': 0.01,
        'gradient_clip': 1.0,
        'target_modules': targets('modernbert-seqcls-v1'),
        'precision': 'float32',
        'class_balance': False,
    }


def test_static_pilot_loop_logs_completed_updates_and_never_selects_step_zero(population, tmp_path):
    dataset = training_projection(population)
    config = build_fixed_r8_reference(_pilot_identity(dataset))
    backend = TinyBackend(config)
    permit = _ExecutionPermit(
        issuer=_ISSUER,
        config_digest=sha(config),
        dataset_digest=sha(dataset),
        admission_digest='sha256:' + '1' * 64,
        attempt_digest='sha256:' + '2' * 64,
        deadline=time.monotonic() + 3600,
        output_limit=100_000_000,
    )
    result = _train_static_pilot(
        config, dataset, backend, tmp_path / 'run', execution_permit=permit,
        deadline_seconds=3600,
    )

    assert result['completed_optimizer_updates'] == 200
    assert result['sample_exposures'] == 3200
    assert result['sample_order_digest'] == config['identity']['sample_order_digest']
    assert result['training_seed'] == config['identity']['seed']
    assert result['dropout_seed'] == config['identity']['dropout_seed']
    assert result['qualification_status'] == 'NOT_EVALUATED'
    assert result['external_deployment_allowed'] is False
    assert result['selected_checkpoint_status'] in (
        'SELECTED_TRAINED_CHECKPOINT', 'NO_QUALIFYING_CHECKPOINT')
    if result['selected_checkpoint_step'] is not None:
        assert result['selected_checkpoint_step'] > 0
    assert result['replay']['initial']['predictions_match'] is True
    assert result['replay']['terminal']['metrics_match'] is True
    rows = [json.loads(line) for line in
            (tmp_path / 'run' / 'training_diagnostics.jsonl').read_text().splitlines()]
    assert len(rows) == 200
    assert [row['completed_optimizer_step'] for row in rows] == list(range(1, 201))
    assert all(row['optimizer_updated'] for row in rows)
    assert all(row['optimizer_parameter_update_norm'] > 0 for row in rows)
    assert all(row['gradient_contributions'] is not None for row in rows[:10])
    assert all(set(row['active_rank_by_target'].values()) == {8} for row in rows)
    assert backend.allocation_calls == 0


def test_v5_backend_passes_seed_and_freezes_rank_policy(population, tmp_path):
    dataset = training_projection(population)
    config = build_fixed_r8_reference(_pilot_identity(dataset))
    backend = StaticPilotBackendV5(config, tmp_path)

    assert backend.seed == config['identity']['seed']
    assert backend.adapter['method'] == 'fixed-r8'
    assert backend.job['dropout_seed'] == config['identity']['dropout_seed']


def test_static_backend_penalty_matches_independent_svd_reference():
    backend = object.__new__(StaticPilotBackendV5)
    backend.adapter = {'method': 'adalora-static-r12', 'target_modules': ['layer']}
    a = torch.randn(3, 5)
    b = torch.randn(7, 3)
    backend.trainable = {
        'base_model.model.layer.lora_A.default': a,
        'base_model.model.layer.lora_B.default': b,
    }

    actual = backend.orthogonal_penalty()
    expected = orthogonal_penalty_reference([(a, b)])
    torch.testing.assert_close(actual, expected)


def test_skipped_scaler_update_is_not_counted():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    class OverflowScaler:
        @staticmethod
        def step(_optimizer):
            return None

        @staticmethod
        def update():
            return None

    assert _optimizer_completed(optimizer, OverflowScaler()) is False
    assert parameter.item() == 1.0


def test_training_body_rejects_unissued_execution_permit(tmp_path):
    with pytest.raises(PermissionError, match='controller-issued execution permit'):
        _train_static_pilot({}, {}, None, tmp_path / 'denied', execution_permit=None)


def test_checkpoint_selection_prioritizes_complete_pairs_over_aggregate_accuracy():
    high_accuracy = {
        'accuracy': 0.91, 'complete_pair_accuracy': 0.41,
        'worst_family_accuracy': 0.22, 'family_macro_accuracy': 0.48,
    }
    better_pairs = {
        'accuracy': 0.84, 'complete_pair_accuracy': 0.63,
        'worst_family_accuracy': 0.51, 'family_macro_accuracy': 0.62,
    }

    assert _checkpoint_selection_key(better_pairs) > _checkpoint_selection_key(high_accuracy)
    with pytest.raises(ValueError, match='finite behavioral checkpoint metrics'):
        _checkpoint_selection_key({**better_pairs, 'complete_pair_accuracy': None})
