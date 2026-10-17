import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from saal_rsi.adaptive_runtime_v4 import runtime_requirements


preflight = importlib.import_module("saal_rsi.adaptive_colab_v5")


def observation(**updates):
    value = {
        "device": "cuda:0",
        "gpu": "NVIDIA A100-SXM4-80GB",
        "accelerator": "A100",
        "gpu_memory_bytes": 80 * 1024**3,
        "free_gpu_bytes_preflight": 70 * 1024**3,
        "host_memory_bytes": 160 * 1024**3,
        "high_memory_observed": True,
        "dependency_hash_lock_enforced": True,
    }
    value.update(updates)
    return value


def test_v5_preflight_requires_structured_success_record():
    record = {"monitor": "hardware_preflight", "observation": observation()}
    parsed = preflight._validate_hardware_preflight_text(
        "install output\n" + json.dumps(record), runtime_requirements("A100", "HIGH_RAM")
    )
    assert parsed == record["observation"]


def test_v5_preflight_rejects_notebook_traceback_even_if_cli_succeeded():
    with pytest.raises(RuntimeError, match="structured success record"):
        preflight._validate_hardware_preflight_text(
            "Traceback (most recent call last):\nCalledProcessError: probe failed",
            runtime_requirements("A100", "HIGH_RAM"),
        )


def test_v5_preflight_rejects_wrong_gpu_or_insufficient_host_ram():
    for bad in (
        observation(gpu="NVIDIA T4"),
        observation(host_memory_bytes=16 * 1024**3, high_memory_observed=False),
    ):
        with pytest.raises(ValueError, match="does not satisfy"):
            preflight._validate_hardware_preflight_text(
                json.dumps({"monitor": "hardware_preflight", "observation": bad}),
                runtime_requirements("A100", "HIGH_RAM"),
            )


def test_v5_preflight_rejects_duplicate_success_records():
    record = json.dumps({"monitor": "hardware_preflight", "observation": observation()})
    with pytest.raises(RuntimeError, match="exactly one"):
        preflight._validate_hardware_preflight_text(
            record + "\n" + record, runtime_requirements("A100", "HIGH_RAM")
        )


def test_v5_preflight_rejects_success_record_mixed_with_exception_trace():
    record = json.dumps({"monitor": "hardware_preflight", "observation": observation()})
    with pytest.raises(RuntimeError, match="exception trace"):
        preflight._validate_hardware_preflight_text(
            record + "\nTraceback (most recent call last):\nCalledProcessError: probe failed",
            runtime_requirements("A100", "HIGH_RAM"),
        )


def test_generated_preflight_gate_runs_before_campaign(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    requirements = runtime_requirements("A100", "HIGH_RAM")
    repo_root = Path(__file__).resolve().parents[3]
    script = preflight._preflight_validator_script(workspace, requirements, repo_root)
    (workspace / "4-hardware_preflight.log").write_text(json.dumps({
        "monitor": "hardware_preflight", "observation": observation()
    }) + "\n")
    result = subprocess.run(
        [sys.executable, "-I", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert '"monitor": "hardware_preflight_verified"' in result.stdout


def test_generated_preflight_gate_rejects_nested_probe_error(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    requirements = runtime_requirements("A100", "HIGH_RAM")
    repo_root = Path(__file__).resolve().parents[3]
    script = preflight._preflight_validator_script(workspace, requirements, repo_root)
    (workspace / "4-hardware_preflight.log").write_text(
        "Traceback (most recent call last):\nCalledProcessError\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "structured success record" in result.stderr
