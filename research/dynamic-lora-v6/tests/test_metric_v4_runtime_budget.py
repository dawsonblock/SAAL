import copy

import pytest

from saal_rsi import adaptive_colab_v4
from saal_rsi.adaptive_compute_budget_v4 import assess_budget, parse_usage
from saal_rsi.adaptive_runtime_v4 import (
    GIB, colab_create_command, runtime_requirements, validate_runtime_observation,
)


def test_a100_runner_imports_signed_colab_version_verifier():
    assert callable(adaptive_colab_v4.verify_colab_cli_version)


def test_signed_job_paths_follow_protocol_run_order(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    for name in ("fixed_r8_47-job.json", "adalora_12_to_8_47-job.json"):
        (jobs / name).write_text("{}")
    protocol = {"runs": [{"name": "fixed_r8_47"}, {"name": "adalora_12_to_8_47"}]}
    assert [p.name for p in adaptive_colab_v4._ordered_job_paths(tmp_path, protocol)] == [
        "fixed_r8_47-job.json", "adalora_12_to_8_47-job.json"
    ]


def test_signed_job_path_inventory_rejects_unadmitted_file(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "fixed_r8_47-job.json").write_text("{}")
    (jobs / "unexpected-job.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        adaptive_colab_v4._ordered_job_paths(tmp_path, {"runs": [{"name": "fixed_r8_47"}]})


def test_a100_high_ram_is_an_exact_colab_request():
    requirements = runtime_requirements('A100', 'HIGH_RAM')
    assert colab_create_command('saal-0123456789abcdefabcd', requirements) == [
        'colab', 'new', '-s', 'saal-0123456789abcdefabcd', '--gpu', 'A100', '--high-mem'
    ]


def test_a100_high_ram_requires_observed_gpu_and_host_memory():
    requirements = runtime_requirements('A100', 'HIGH_RAM')
    observation = {
        'device': 'cuda:0', 'gpu': 'NVIDIA A100-SXM4-40GB', 'accelerator': 'A100',
        'gpu_memory_bytes': 40*GIB, 'free_gpu_bytes_preflight': 35*GIB,
        'host_memory_bytes': 80*GIB, 'high_memory_observed': True,
        'dependency_hash_lock_enforced': True,
    }
    assert validate_runtime_observation(requirements, observation) == observation
    for field, value in [('gpu', 'Tesla T4'), ('accelerator', 'T4'),
                         ('gpu_memory_bytes', 16*GIB), ('host_memory_bytes', 16*GIB),
                         ('high_memory_observed', False)]:
        changed = copy.deepcopy(observation)
        changed[field] = value
        with pytest.raises(ValueError):
            validate_runtime_observation(requirements, changed)


def test_colab_usage_parser_requires_all_account_fields():
    sample = 'Current balance: 174.43 compute units\nUsage rate: 12.5/hr\nActive assignments: 1\n'
    assert parse_usage(sample) == {'balance': 174.43, 'rate': 12.5, 'active_assignments': 1}
    with pytest.raises(ValueError):
        parse_usage('Current balance: 174.43 compute units\nUsage rate: 12.5/hr\n')


def test_compute_guard_accounts_actual_and_projected_spend():
    usage = {'balance': 160.0, 'rate': 12.0, 'active_assignments': 1}
    accepted = assess_budget(starting_balance=174.0, usage=usage, cap=30,
                             remaining_seconds=3600, require_single_assignment=True)
    assert accepted['reported_spend_compute_units'] == 14.0
    assert accepted['projected_spend_compute_units'] == 26.0
    with pytest.raises(ValueError):
        assess_budget(starting_balance=174.0, usage=usage, cap=20,
                      remaining_seconds=3600, require_single_assignment=True)
    with pytest.raises(ValueError):
        assess_budget(starting_balance=174.0,
                      usage={**usage, 'active_assignments': 2}, cap=30,
                      remaining_seconds=0, require_single_assignment=True)


def test_remote_colab_scripts_are_syntactically_complete():
    from saal_rsi.adaptive_colab_v4 import (
        _hardware_probe, _job_remote_script, _remote_installer,
    )
    requirements = runtime_requirements('A100','HIGH_RAM')
    compile(_hardware_probe(requirements),'hardware.py','exec')
    compile(_remote_installer('a'*64,'b'*64),'install.py','exec')
    compile(_job_remote_script('sha256:'+'c'*64,'d'*64,
        '/content/run-00','/content/run-00-job.json'),'training.py','exec')


def test_checkpoint_selector_rejects_critical_or_nonlearning_candidates():
    from saal_rsi.adaptive_training_v4 import eligible_trained_checkpoint
    initial={'accuracy':0.5,'complete_pair_accuracy':0.5,'worst_family_accuracy':0.4,
        'critical_errors':{'counts':{'authorization_violation':0,
            'unsafe_action_when_clarification_required':0,'false_refusal':1}},
        'class_accuracy':{'CHEAP':0.5,'FAST':0.5,'ASK':0.5,'REFUSE':0.5},
        'invalid_outputs':0}
    safe_gain={'accuracy':0.7,'complete_pair_accuracy':0.7,'worst_family_accuracy':0.4,
        'critical_errors':{'counts':{'authorization_violation':0,
            'unsafe_action_when_clarification_required':0,'false_refusal':1}},
        'class_accuracy':{'CHEAP':0.6,'FAST':0.6,'ASK':0.6,'REFUSE':0.6},
        'invalid_outputs':0}
    assert eligible_trained_checkpoint(safe_gain,initial)
    dangerous={**safe_gain,'critical_errors':{'counts':{
        'authorization_violation':1,'unsafe_action_when_clarification_required':0,
        'false_refusal':1}}}
    assert not eligible_trained_checkpoint(dangerous,initial)
    useless={**safe_gain,'complete_pair_accuracy':initial['complete_pair_accuracy']}
    assert not eligible_trained_checkpoint(useless,initial)
