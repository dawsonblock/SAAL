import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "metric-repair" / "report_a100_pilot_v4_posthoc.py"
SPEC = importlib.util.spec_from_file_location("a100_report_posthoc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_report_jobs_follow_signed_protocol_order(tmp_path):
    jobs = tmp_path / "admission" / "jobs"
    jobs.mkdir(parents=True)
    for name in ("adalora_12_to_8_47", "fixed_r8_47", "fixed_r12_47"):
        (jobs / f"{name}-job.json").write_text("{}")
    protocol = {"runs": [
        {"name": "fixed_r8_47"},
        {"name": "fixed_r12_47"},
        {"name": "adalora_12_to_8_47"},
    ]}
    assert [path.name for path in MODULE.ordered_job_paths(tmp_path / "admission", protocol)] == [
        "fixed_r8_47-job.json",
        "fixed_r12_47-job.json",
        "adalora_12_to_8_47-job.json",
    ]


def test_report_rejects_missing_and_extra_signed_jobs(tmp_path):
    jobs = tmp_path / "admission" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "run-a-job.json").write_text("{}")
    protocol = {"runs": [{"name": "run-a"}, {"name": "run-b"}]}
    try:
        MODULE.ordered_job_paths(tmp_path / "admission", protocol)
    except ValueError as exc:
        assert "inventory differs" in str(exc)
    else:
        raise AssertionError("missing signed job was accepted")

    (jobs / "run-b-job.json").write_text("{}")
    (jobs / "unexpected-job.json").write_text("{}")
    try:
        MODULE.ordered_job_paths(tmp_path / "admission", protocol)
    except ValueError as exc:
        assert "inventory differs" in str(exc)
    else:
        raise AssertionError("extra signed job was accepted")
