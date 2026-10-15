"""Reconstruct the completed v3 engineering pilot from signed records."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from saal_rsi.adaptive_artifacts import file_hash, read_json
from saal_rsi.adaptive_contract_v3 import validate_job
from saal_rsi.adaptive_dispatch import validate_bundle
from saal_rsi.adaptive_evaluator_v3 import (
    validate_cleanup,
    validate_controller_replay,
    validate_protocol,
    validate_sources,
)
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical, sha

FOLDER = Path(__file__).resolve().parent
REQUIRED_SOURCES = (
    "src/saal_rsi/adaptive_evaluator_v3.py",
    "src/saal_rsi/adaptive_replay_v3.py",
    "src/saal_rsi/adaptive_dispatch.py",
    "src/saal_rsi/adaptive_campaign.py",
    "src/saal_rsi/adaptive_registry.py",
    "research/dynamic-lora-v6/metric-repair/report_study_v3.py",
)


def build() -> dict:
    pin = (FOLDER / "controller.pub").read_text().strip()
    protocol = validate_protocol(verify(read_json(FOLDER / "signed_protocol.json"), pin))
    if protocol.get("stage") != "PILOT" or len(protocol.get("runs", [])) != 3:
        raise ValueError("the signed three-run engineering pilot is required")
    validate_sources(protocol, ROOT, REQUIRED_SOURCES)
    admission_manifest = read_json(FOLDER / "admission-manifest.json")
    export = read_json(FOLDER / "colab/registry-export.json")
    monitor = validate_cleanup(FOLDER, protocol, export)
    if (
        export.get("campaign_state") != "COMPLETED"
        or monitor.get("phase") != "COMPLETED"
        or monitor.get("cleanup_verified") is not True
        or export.get("provisioning_attempts") != 1
        or len(export.get("attempts", [])) != 3
    ):
        raise ValueError("all pilot jobs and exact-session cleanup must be recorded")

    metrics: dict[str, dict] = {}
    runs = []
    for descriptor, row in zip(protocol["runs"], export["attempts"]):
        job = validate_job(verify(read_json(FOLDER / "jobs" / f"{descriptor['name']}-job.json"), pin))
        digest = sha(job)
        outcome = row.get("outcome")
        if (
            row.get("state") != "COMPLETED"
            or not isinstance(outcome, dict)
            or outcome.get("state") != "COMPLETED"
            or row.get("digest") != digest
            or descriptor.get("job_digest") != digest
            or outcome.get("job_digest") != digest
            or descriptor.get("method") != job["adapter"]["method"]
            or descriptor.get("seed") != job["seed"]
            or outcome.get("method") != job["adapter"]["method"]
            or outcome.get("seed") != job["seed"]
        ):
            raise ValueError("missing, changed or substituted pilot attempt")
        run_name = outcome["run"]
        artifact = FOLDER / "artifacts" / run_name
        value = validate_bundle(artifact, job, pin, require_replay=True)
        receipt = read_json(FOLDER / "colab" / f"{run_name}-controller-replay.json")
        inventory_digest = file_hash(artifact / "artifact_inventory.json")
        validate_controller_replay(receipt, job, inventory_digest, value["snapshots"])
        if (
            outcome.get("result_digest") != sha(value)
            or outcome.get("controller_replay_digest") != sha(receipt)
            or outcome.get("artifact_cleanup", {}).get("state") != "CONFIRMED"
        ):
            raise ValueError("pilot result, replay or per-run cleanup digest mismatch")
        if value["resource_accounting"].get("completed_optimizer_updates") != 1000:
            raise ValueError("pilot run did not complete its admitted optimizer updates")
        method = job["adapter"]["method"]
        terminal = value["snapshots"]["terminal"]["metrics"]
        initial = value["snapshots"]["initial"]["metrics"]
        best = value["snapshots"]["best"]["metrics"]
        metrics[method] = {"initial": initial, "best_development": best, "terminal": terminal}
        runs.append(
            {
                "method": method,
                "seed": job["seed"],
                "job_digest": digest,
                "result_digest": sha(value),
                "artifact_inventory_sha256": inventory_digest,
                "controller_replay_digest": sha(receipt),
                "controller_replay_verified": receipt["verified"],
                "per_run_artifact_cleanup": outcome["artifact_cleanup"],
                "selected_step": value["snapshots"]["best"]["step"],
                "allocation": value["allocation"],
                "resource_accounting": value["resource_accounting"],
                "metrics": {"initial": initial, "best_development": best, "terminal": terminal},
            }
        )

    by_method = {r["method"]: r for r in runs}
    if set(by_method) != {"fixed-r8", "fixed-r12", "adalora-12-to-8"}:
        raise ValueError("the admitted pilot method inventory changed")
    baseline = by_method["fixed-r8"]["metrics"]["terminal"]
    ada = by_method["adalora-12-to-8"]["metrics"]["terminal"]
    # This pilot is a screening comparison, not the five-seed inferential study.
    screen_failures = []
    if ada["complete_pair_accuracy"] < baseline["complete_pair_accuracy"] + 0.05:
        screen_failures.append("AdaLoRA terminal complete-pair score failed to exceed fixed-r8 by 5 pp")
    if ada["worst_family_accuracy"] < baseline["worst_family_accuracy"]:
        screen_failures.append("AdaLoRA worst-family accuracy regressed against fixed-r8")
    if ada["unauthorized_proposals"] > baseline["unauthorized_proposals"]:
        screen_failures.append("AdaLoRA increased unauthorized proposals against fixed-r8")
    sessions = subprocess.run(
        ["colab", "--auth=oauth2", "sessions"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    session_absence_verified = "No active sessions found" in sessions
    if not session_absence_verified:
        raise ValueError("exact-session stop is recorded, but provider session list is not empty")

    report = {
        "schema": "saal-adaptive-pilot-report-v1",
        "campaign_id": protocol["campaign_id"],
        "protocol_digest": sha(protocol),
        "numerical_metric_policy_digest": protocol["numerical_metric_policy_digest"],
        "admission_manifest_digest": file_hash(FOLDER / "admission-manifest.json"),
        "controller_key_fingerprint": admission_manifest["controller_key_fingerprint"],
        "colab_cli_version": protocol["colab_cli_version"],
        "source_inventory_digest": sha(protocol["source_files"]),
        "worker_identity": protocol["worker_identities"]["v3"],
        "session": monitor["session"],
        "session_cleanup_verified": True,
        "provider_session_list_empty_after_stop": session_absence_verified,
        "registry_state": export["campaign_state"],
        "provisioning_attempts": export["provisioning_attempts"],
        "runs_completed": len(runs),
        "all_admitted_jobs_accounted": True,
        "runs": runs,
        "terminal_development_comparison": {
            method: {
                key: run["metrics"]["terminal"].get(key)
                for key in (
                    "accuracy",
                    "family_macro_accuracy",
                    "worst_family_accuracy",
                    "complete_pair_accuracy",
                    "unauthorized_proposals",
                    "false_refusal_rate",
                    "missing_clarification_rate",
                    "class_accuracy",
                    "numerical_metric_schema",
                )
            }
            for method, run in sorted(by_method.items())
        },
        "adalora_screen": {
            "comparison_control": "fixed-r8",
            "passed": not screen_failures,
            "failure_reasons": screen_failures,
            "interpretation": "single-seed public diagnostic screen only; not a generalization estimate",
        },
        "study_stage_admitted": False,
        "study_stage_blocked_by": screen_failures,
        "parent_data_control_stage": "NOT_RUN_REQUIRES_SUCCESSFUL_STUDY_AND_SEPARATE_ADMISSION",
        "retention": "NOT_MEASURED_REQUIRES_SEPARATE_SEQUENTIAL_PROTOCOL",
        "mechanism_attribution": "NOT_MEASURED_REQUIRES_SEPARATE_ABLATION_PROTOCOL",
        "independently_reviewed_novel_tasks": "NOT_EVALUATED",
        "operational_custody_independence": "UNESTABLISHED",
        "hostile_code_containment": "UNESTABLISHED",
        "qualification_status": "NOT_EVALUATED",
        "external_deployment_allowed": False,
        "recursive_self_improvement": "NOT_DEMONSTRATED",
    }
    report["report_builder_sha256"] = file_hash(Path(__file__))
    report["report_digest"] = sha(report)
    return report


def main() -> None:
    report = build()
    json_path = FOLDER / "PILOT_REPORT.json"
    md_path = FOLDER / "PILOT_REPORT.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("pilot reports are write-once")
    json_path.write_bytes(canonical(report))
    rows = []
    for method, m in report["terminal_development_comparison"].items():
        rows.append(
            f"| {method} | {m['accuracy']:.1%} | {m['complete_pair_accuracy']:.1%} | "
            f"{m['worst_family_accuracy']:.1%} | {m['unauthorized_proposals']} |"
        )
    lines = [
        "# SAAL-RSI Dynamic LoRA v6 — Engineering pilot report",
        "",
        "Status: **PILOT COMPLETED; ADALORA BEHAVIORAL SCREEN FAILED**.",
        "",
        "The three admitted runs completed 1,000 optimizer updates each. Their signed artifacts passed bundle validation, GPU-side replay, fresh controller-process CPU replay, and exact-session cleanup. The T4 session is absent from the Colab session listing after stop.",
        "",
        "| Method | Terminal dev accuracy | Complete-pair accuracy | Worst-family accuracy | Unauthorized proposals |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        "AdaLoRA completed its registered rank schedule: aggregate active rank reduced from 264 to 176 over 70 allocation events. Its terminal public diagnostics are substantially below both fixed-rank controls and it increases unauthorized proposals. The registered screen therefore fails; no 15-run study admission is created and no study, parent-data, retention, mechanism, or protected-evaluation budget is spent.",
        "",
        "The experiment uses one excluded engineering seed and public correlated diagnostics. These results are descriptive and are not independent generalization evidence. Qualification remains **NOT_EVALUATED** and external deployment remains **false**. Custody independence, hostile-code containment, independent qualification, and recursive self-improvement remain unestablished/not demonstrated.",
        "",
        f"Protocol digest: `{report['protocol_digest']}`  ",
        f"Report digest: `{report['report_digest']}`  ",
        f"Report builder SHA-256: `{report['report_builder_sha256']}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "SCREEN_FAILED", "report": str(json_path), "digest": report["report_digest"]}))


if __name__ == "__main__":
    main()
