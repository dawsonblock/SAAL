#!/usr/bin/env python3
"""Verify and report the run05 static AdaLoRA regularization screen.

This report is deliberately scoped to public development evidence. It verifies
the signed input manifest, admissions, one-use receipts, local witness journal,
artifact inventories and in-worker checkpoint replay claims. It does not
claim independent custody, independent replay, qualification, or deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from saal_rsi.adaptive_static_pilot_v5 import (
    _verify_admission,
    load_trust_store,
    validate_attempt_receipt,
    validate_outcome_receipt,
    validate_pilot_config,
)
from saal_rsi.adaptive_dataset import validate_training_data
from saal_rsi.model_contract import verify
from saal_rsi.spec import canonical, parse_unique_json, sha


CAMPAIGN = "saal098-static-reg-20261018-05"
ATTEMPTS = (
    "saal098r05-arm-fixedr8",
    "saal098r05-arm-orth0",
    "saal098r05-arm-orth001",
    "saal098r05-arm-orth01",
    "saal098r05-arm-orth05",
)
EXPECTED_COEFFICIENTS = {
    "saal098r05-arm-fixedr8": 0.0,
    "saal098r05-arm-orth0": 0.0,
    "saal098r05-arm-orth001": 0.01,
    "saal098r05-arm-orth01": 0.1,
    "saal098r05-arm-orth05": 0.5,
}
ZERO_HASH = "sha256:" + "0" * 64


def raw_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    return parse_unique_json(path.read_bytes())


def member_json(archive: zipfile.ZipFile, name: str) -> dict:
    return parse_unique_json(archive.read(name))


def safe_zip(archive: zipfile.ZipFile, *, max_bytes: int) -> list[str]:
    infos = archive.infolist()
    names = [item.filename for item in infos]
    if not infos or len(names) != len(set(names)) or archive.testzip() is not None:
        raise ValueError("archive CRC or duplicate-member check failed")
    if sum(item.file_size for item in infos) > max_bytes:
        raise ValueError("archive exceeds its declared size limit")
    for item in infos:
        path = PurePosixPath(item.filename)
        mode = (item.external_attr >> 16) & 0o170000
        if (path.is_absolute() or ".." in path.parts or item.is_dir()
                or mode == 0o120000):
            raise ValueError("unsafe archive member")
    return names


def input_inventory(root: Path) -> dict:
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in public input evidence")
        if path.is_file() and path.name != "run_manifest.json":
            raw = path.read_bytes()
            inventory[path.relative_to(root).as_posix()] = {
                "size": len(raw), "sha256": raw_sha(raw),
            }
    return inventory


def prepare_public_inputs(run_dir: Path) -> dict:
    """Extract public inputs only; the one-use bearer transport is never copied."""
    public = run_dir / "public_inputs"
    bundle = run_dir / "inputs.zip"
    if public.exists():
        if not public.is_dir() or public.is_symlink():
            raise ValueError("public input evidence path is not a regular directory")
        return read_json(public / "public_input_manifest.json")
    if not bundle.is_file() or bundle.is_symlink():
        raise FileNotFoundError("run05 input package or published public inputs are missing")
    temporary = run_dir / ".public_inputs.tmp"
    if temporary.exists():
        raise FileExistsError("temporary extraction path already exists")
    temporary.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(bundle) as archive:
            names = safe_zip(archive, max_bytes=100_000_000)
            envelope = member_json(archive, "run_manifest.json")
            trust = member_json(archive, "trust_store.json")
            trusted, _, _ = load_trust_store_bytes(canonical(trust), run_dir)
            manifest = verify(envelope, trusted[envelope["key_sha256"]])
            if manifest.get("campaign_id") != CAMPAIGN:
                raise ValueError("signed input manifest campaign mismatch")
            if "transport.json" not in manifest["files"]:
                raise ValueError("expected one-use transport entry is not bound by the manifest")
            excluded = {"transport.json"}
            for name in names:
                if name == "run_manifest.json" or name in excluded:
                    continue
                destination = temporary / PurePosixPath(name)
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
            (temporary / "run_manifest.json").write_bytes(canonical(envelope))
            observed = input_inventory(temporary)
            expected = {name: record for name, record in manifest["files"].items()
                        if name not in excluded}
            if observed != expected:
                raise ValueError("public input bytes differ from signed input inventory")
            publication = {
                "schema": "saal-static-pilot-public-input-evidence-v1",
                "campaign_id": CAMPAIGN,
                "signed_run_manifest_sha256": raw_sha(canonical(envelope)),
                "signed_input_manifest_digest": sha(manifest),
                "included_files": observed,
                "omitted_secret_bearing_files": {
                    name: manifest["files"][name]
                    for name in sorted(excluded)
                },
                "omission_reason": (
                    "transport.json held one-use bearer credentials; the exact signed "
                    "inventory entry is retained, while credentials are excluded"
                ),
                "transport_credential_values_published": False,
            }
            (temporary / "public_input_manifest.json").write_bytes(canonical(publication))
        os.replace(temporary, public)
        return publication
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_trust_store_bytes(raw: bytes, run_dir: Path):
    path = run_dir / ".trust-store-verification.tmp"
    if path.exists():
        raise FileExistsError("temporary trust-store verification path already exists")
    path.write_bytes(raw)
    try:
        return load_trust_store(path)
    finally:
        path.unlink(missing_ok=True)


def verify_public_inputs(run_dir: Path) -> tuple[dict, dict, dict, dict, str]:
    public = run_dir / "public_inputs"
    publication = read_json(public / "public_input_manifest.json")
    if (publication.get("schema") != "saal-static-pilot-public-input-evidence-v1"
            or publication.get("campaign_id") != CAMPAIGN
            or publication.get("transport_credential_values_published") is not False):
        raise ValueError("public input evidence manifest rejected")
    inventory = input_inventory(public)
    inventory.pop("public_input_manifest.json", None)
    envelope = read_json(public / "run_manifest.json")
    trust_store = read_json(public / "trust_store.json")
    admission_keys, witness_pin, _ = load_trust_store(public / "trust_store.json")
    if envelope["key_sha256"] not in admission_keys:
        raise PermissionError("input manifest signer is not trusted")
    manifest = verify(envelope, admission_keys[envelope["key_sha256"]])
    if manifest.get("campaign_id") != CAMPAIGN:
        raise ValueError("signed manifest campaign mismatch")
    if raw_sha(canonical(envelope)) != publication["signed_run_manifest_sha256"]:
        raise ValueError("published signed manifest digest mismatch")
    if sha(manifest) != publication["signed_input_manifest_digest"]:
        raise ValueError("published signed manifest payload digest mismatch")
    omitted = publication["omitted_secret_bearing_files"]
    expected_omitted = {name: manifest["files"].get(name) for name in omitted}
    if expected_omitted != omitted or set(omitted) != {"transport.json"}:
        raise ValueError("omitted credential-file inventory does not match signed manifest")
    expected = {name: item for name, item in manifest["files"].items()
                if name not in omitted}
    if inventory != expected or publication["included_files"] != expected:
        raise ValueError("published input bytes differ from signed inventory")
    dataset = read_json(public / "dataset.json")
    validate_training_data(dataset)
    if raw_sha((public / "dataset.json").read_bytes()) != manifest["files"].get("dataset.json", {}).get("sha256"):
        raise ValueError("dataset bytes do not match the signed input inventory")
    closure = read_json(public / "source_closure.json")
    source_archive = (public / "source.tar.gz").read_bytes()
    if (raw_sha(source_archive) != manifest["source_archive_sha256"]
            or raw_sha(source_archive) != closure["archive_sha256"]):
        raise ValueError("source archive digest mismatch")
    return manifest, trust_store, dataset, closure, witness_pin


def verify_source_archive(public: Path, closure: dict) -> dict:
    expected = closure.get("files")
    if not isinstance(expected, dict) or len(expected) != closure.get("file_count"):
        raise ValueError("source closure inventory shape mismatch")
    observed = {}
    with tarfile.open(public / "source.tar.gz", "r:gz") as archive:
        members = archive.getmembers()
        names = [item.name for item in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate source archive path")
        for item in members:
            path = PurePosixPath(item.name)
            if (path.is_absolute() or ".." in path.parts
                    or (not item.isfile() and not item.isdir())
                    or item.issym() or item.islnk()):
                raise ValueError("unsafe or non-file source archive member")
            if item.isdir():
                continue
            stream = archive.extractfile(item)
            if stream is None:
                raise ValueError("source archive member unreadable")
            raw = stream.read()
            observed[item.name] = {"size": len(raw), "sha256": raw_sha(raw)}
    normalized_expected = {
        name: {"size": value["size"], "sha256": value["sha256"]}
        for name, value in expected.items()
    }
    if observed != normalized_expected:
        raise ValueError("source archive members differ from the reviewed closure")
    if sum(item["size"] for item in observed.values()) != closure["total_file_bytes"]:
        raise ValueError("source archive byte count differs from closure")
    return {"file_count": len(observed), "total_file_bytes": sum(x["size"] for x in observed.values()),
            "closure_digest": sha(closure)}


def metric_slice(metrics: dict) -> dict:
    critical = metrics["critical_errors"]["counts"]
    return {
        "accuracy": metrics["accuracy"],
        "complete_pair_accuracy": metrics["complete_pair_accuracy"],
        "worst_family_accuracy": metrics["worst_family_accuracy"],
        "family_macro_accuracy": metrics["family_macro_accuracy"],
        "false_refusal_rate": metrics["false_refusal_rate"],
        "unauthorized_proposals": metrics["unauthorized_proposals"],
        "false_refusals": critical["false_refusal"],
        "authorization_violations": critical["authorization_violation"],
        "clarification_not_returned": critical["clarification_not_returned"],
        "unsafe_action_when_asking": critical["unsafe_action_when_clarification_required"],
        "negative_log_likelihood": metrics["negative_log_likelihood"],
        "expected_calibration_error": metrics["expected_calibration_error"],
        "class_accuracy": metrics["class_accuracy"],
        "support": metrics["support"],
    }


def finite_tree(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(key) and finite_tree(item) for key, item in value.items())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def summarize_diagnostics(raw: bytes, expected_steps: int) -> dict:
    rows = [parse_unique_json(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != expected_steps:
        raise ValueError("training diagnostic row count differs from completed updates")
    if [row.get("completed_optimizer_step") for row in rows] != list(range(1, expected_steps + 1)):
        raise ValueError("training diagnostic steps are not consecutive")
    if any(row.get("optimizer_updated") is not True or row.get("precision") != "float32"
           or not finite_tree(row) for row in rows):
        raise ValueError("nonfinite, skipped, or unexpected-precision training row")
    if any(rows[i]["sample_exposures"] <= rows[i - 1]["sample_exposures"]
           for i in range(1, len(rows))):
        raise ValueError("sample exposures did not strictly advance")
    contributions = [row["gradient_contributions"] for row in rows
                     if row.get("gradient_contributions") is not None]
    ratios = [item["orthogonal_to_ce_norm_ratio"] for item in contributions]
    cosines = [item["gradient_cosine_similarity"] for item in contributions]
    clipping = [row["gradient_before_clip"] for row in rows]
    effective = []
    for row in rows:
        values = row["effective_update_by_target"].values()
        effective.append(math.sqrt(sum(value["frobenius_norm"] ** 2 for value in values)))
    return {
        "optimizer_updates": len(rows),
        "sample_exposures": rows[-1]["sample_exposures"],
        "precision": rows[-1]["precision"],
        "cross_entropy_first_step": rows[0]["cross_entropy"],
        "cross_entropy_last_step": rows[-1]["cross_entropy"],
        "orthogonal_penalty_last_step": rows[-1]["orthogonal_penalty"],
        "weighted_orthogonal_penalty_last_step": rows[-1]["weighted_orthogonal_penalty"],
        "learning_rates_first_step": rows[0]["learning_rates"],
        "learning_rates_last_step": rows[-1]["learning_rates"],
        "diagnostic_gradient_samples": len(contributions),
        "orthogonal_to_ce_gradient_norm_ratio": {
            "mean": sum(ratios) / len(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
            "first": ratios[0] if ratios else None,
            "last": ratios[-1] if ratios else None,
        },
        "orthogonal_vs_ce_gradient_cosine": {
            "mean": sum(cosines) / len(cosines) if cosines else None,
            "min": min(cosines) if cosines else None,
            "max": max(cosines) if cosines else None,
        },
        "gradient_clipping": {
            "clipped_steps": sum(item["would_clip"] for item in clipping),
            "total_steps": len(clipping),
            "minimum_predicted_scale": min(item["predicted_clip_scale"] for item in clipping),
            "maximum_pre_clip_global_norm": max(item["pre_clip_global_gradient_norm"]
                                                for item in clipping),
        },
        "effective_update_frobenius_aggregate": {
            "first_step": effective[0], "last_step": effective[-1],
        },
        "parameter_update_norm": {
            "first_step": rows[0]["optimizer_parameter_update_norm"],
            "last_step": rows[-1]["optimizer_parameter_update_norm"],
        },
        "peak_gpu_allocated_bytes": max(row["gpu_memory"]["allocated_bytes"] for row in rows),
        "peak_gpu_reserved_bytes": max(row["gpu_memory"]["reserved_bytes"] for row in rows),
        "training_elapsed_seconds": rows[-1]["elapsed_seconds"],
    }


def verify_artifact(run_dir: Path, artifact_path: Path, attempt: str,
                    config: dict, dataset: dict, admission_keys: dict,
                    witness_pin: str, identity_record: dict,
                    ledger_events: list[dict]) -> dict:
    with zipfile.ZipFile(artifact_path) as archive:
        names = safe_zip(archive, max_bytes=100_000_000)
        required = {
            "admission.json", "artifact_inventory.json", "attempt_receipt.json",
            "result.json", "runtime_preflight.json", "snapshot_manifest.json",
            "terminal_receipt.json", "training/development_metrics.jsonl",
            "training/initial.json", "training/initial.safetensors",
            "training/initial_rank_state.json", "training/result.json",
            "training/terminal.json", "training/terminal.safetensors",
            "training/terminal_rank_state.json", "training/training_diagnostics.jsonl",
        }
        if set(names) != required:
            raise ValueError(f"artifact member set mismatch for {attempt}")
        admission = member_json(archive, "admission.json")
        config_digest = sha(config)
        admission_payload = _verify_admission(config, dataset, admission, admission_keys)
        admission_digest = sha(admission)
        if (admission_payload["campaign_id"] != CAMPAIGN
                or admission_payload["attempt_id"] != attempt
                or admission_digest != identity_record["bindings"]["admission_digest"]
                or config_digest != identity_record["bindings"]["configuration_digest"]):
            raise ValueError(f"admission identity mismatch for {attempt}")
        if canonical(member_json(archive, "admission.json")) != canonical(
                parse_unique_json((run_dir / "public_inputs" / "configs" / attempt
                                   / "admission.json").read_bytes())):
            raise ValueError(f"artifact admission differs from public signed admission for {attempt}")

        inventory = member_json(archive, "artifact_inventory.json")
        if inventory.get("schema") != "saal-static-pilot-artifact-inventory-v5":
            raise ValueError(f"artifact inventory schema mismatch for {attempt}")
        actual_inventory = {}
        for name in names:
            if name == "artifact_inventory.json":
                continue
            raw = archive.read(name)
            actual_inventory[name] = {"size": len(raw), "sha256": raw_sha(raw)}
        if inventory.get("files") != actual_inventory:
            raise ValueError(f"artifact inventory digest mismatch for {attempt}")

        result = member_json(archive, "result.json")
        training_result = member_json(archive, "training/result.json")
        training_projection = dict(result)
        for name in ("admission_digest", "attempt_receipt_digest", "attempt_sequence",
                     "runtime_preflight", "campaign_status", "finalized_outcome_digest",
                     "training_artifact_inventory_digest", "terminal_receipt_digest"):
            training_projection.pop(name, None)
        accounting = dict(training_projection.get("resource_accounting", {}))
        for name in ("compute_units_budgeted", "compute_unit", "compute_units_actual",
                     "compute_units_status"):
            accounting.pop(name, None)
        training_projection["resource_accounting"] = accounting
        if training_projection != training_result:
            raise ValueError(f"outer result does not preserve training result for {attempt}")
        if (result.get("configuration_digest") != config_digest
                or result.get("dataset_digest") != sha(dataset)
                or result.get("completed_optimizer_updates") != config["steps"]
                or result.get("sample_exposures") != config["sample_exposures"]
                or result.get("qualification_status") != "NOT_EVALUATED"
                or result.get("external_deployment_allowed") is not False):
            raise ValueError(f"result binding or status mismatch for {attempt}")
        pre_final_result = dict(result)
        for name in ("finalized_outcome_digest", "training_artifact_inventory_digest",
                     "terminal_receipt_digest"):
            pre_final_result.pop(name, None)
        training_files = {}
        for name in names:
            if name in ("artifact_inventory.json", "terminal_receipt.json"):
                continue
            raw = (canonical(pre_final_result) if name == "result.json"
                   else archive.read(name))
            training_files[name] = {"size": len(raw), "sha256": raw_sha(raw)}
        training_inventory = {
            "schema": "saal-static-pilot-artifact-inventory-v5",
            "files": training_files,
        }
        training_inventory_digest = sha(training_inventory)
        if result.get("training_artifact_inventory_digest") != training_inventory_digest:
            raise ValueError(f"result does not bind the pre-finalization training inventory for {attempt}")
        if not result.get("frozen_weights", {}).get("unchanged"):
            raise ValueError(f"frozen pretrained weights changed for {attempt}")
        replay = result.get("replay", {})
        if not replay or any(not all(value.values()) for value in replay.values()):
            raise ValueError(f"within-worker checkpoint replay failed for {attempt}")
        if result.get("selected_checkpoint_status") != "NO_QUALIFYING_CHECKPOINT":
            raise ValueError(f"unexpected checkpoint selection status for {attempt}")
        if result.get("selected_checkpoint_step") is not None or result.get("best") is not None:
            raise ValueError(f"non-qualifying candidate was selected for {attempt}")

        runtime = member_json(archive, "runtime_preflight.json")
        if (runtime.get("gpu", "").find("A100") < 0
                or runtime.get("gpu_memory_bytes", 0) < 30 * 1024**3
                or runtime.get("host_memory_bytes", 0) < 32 * 1024**3
                or runtime.get("dependency_lock_status", {}).get("versions_match") is not True):
            raise ValueError(f"runtime preflight does not satisfy signed requirements for {attempt}")

        attempt_receipt = member_json(archive, "attempt_receipt.json")
        attempt_body = verify(attempt_receipt, witness_pin)
        validate_attempt_receipt(
            attempt_receipt, witness_pin, admission_payload, admission_digest,
            attempt_body["request_nonce"],
        )
        terminal_receipt = member_json(archive, "terminal_receipt.json")
        terminal_body = verify(terminal_receipt, witness_pin)
        outcome_digest = result.get("finalized_outcome_digest")
        expected_outcome_digest = sha({
            "result_digest": sha(pre_final_result),
            "training_artifact_inventory_digest": training_inventory_digest,
        })
        if outcome_digest != expected_outcome_digest:
            raise ValueError(f"final outcome digest does not bind the completed result for {attempt}")
        validate_outcome_receipt(
            terminal_receipt, witness_pin, admission_payload,
            sha(attempt_receipt), outcome_digest, admission_digest,
            attempt_body["sequence"], terminal_body["request_nonce"], "COMPLETED",
        )
        if terminal_body["status"] != "COMPLETED":
            raise ValueError(f"terminal receipt is not completed for {attempt}")

        selected = [event for event in ledger_events if event["attempt_id"] == attempt]
        kinds = [event["kind"] for event in selected]
        if kinds != ["CONSUMED", "COMPLETED", "ARTIFACT_RECEIVED"]:
            raise ValueError(f"ledger transition sequence mismatch for {attempt}")
        consume_details = {"receipt_digest": sha(attempt_receipt),
                           "bindings_digest": sha(identity_record["bindings"])}
        terminal_details = {"attempt_receipt_digest": sha(attempt_receipt),
                            "outcome_digest": outcome_digest,
                            "consumed_sequence": attempt_body["sequence"]}
        artifact_raw = artifact_path.read_bytes()
        artifact_receipt = {
            "attempt_id": attempt, "size": len(artifact_raw),
            "sha256": raw_sha(artifact_raw), "members": sorted(names),
            "status": "RECEIVED",
        }
        if (selected[0]["details_digest"] != sha(consume_details)
                or selected[1]["details_digest"] != sha(terminal_details)
                or selected[2]["details_digest"] != sha(artifact_receipt)):
            raise ValueError(f"signed ledger details do not bind receipts for {attempt}")
        if (selected[0]["sequence"] != attempt_body["sequence"]
                or selected[1]["sequence"] != terminal_body["sequence"]):
            raise ValueError(f"ledger and attempt receipt sequences differ for {attempt}")

        initial = member_json(archive, "training/initial.json")
        terminal = member_json(archive, "training/terminal.json")
        if initial.get("step") != 0 or terminal.get("step") != config["steps"]:
            raise ValueError(f"checkpoint step identity mismatch for {attempt}")
        diagnostics = summarize_diagnostics(
            archive.read("training/training_diagnostics.jsonl"), config["steps"])
        pair_total = terminal["metrics"]["support"]["counterfactual_pairs"]["counterfactual_opposite_label"]
        return {
            "attempt_id": attempt,
            "method": config["method"],
            "orthogonal_coefficient": config["orthogonal_coefficient"],
            "configuration_digest": config_digest,
            "admission_digest": admission_digest,
            "artifact_sha256": raw_sha(artifact_raw),
            "artifact_bytes": len(artifact_raw),
            "artifact_member_count": len(names),
            "updates": result["completed_optimizer_updates"],
            "sample_exposures": result["sample_exposures"],
            "initial_metrics": metric_slice(initial["metrics"]),
            "terminal_metrics": metric_slice(terminal["metrics"]),
            "terminal_counterfactual_pairs": pair_total,
            "checkpoint_selection": {
                "status": result["selected_checkpoint_status"],
                "step": result["selected_checkpoint_step"],
            },
            "within_worker_replay": replay,
            "frozen_backbone_unchanged": result["frozen_weights"]["unchanged"],
            "runtime_preflight": {
                "gpu": runtime["gpu"],
                "gpu_memory_gib": runtime["gpu_memory_bytes"] / 1024**3,
                "host_memory_gib": runtime["host_memory_bytes"] / 1024**3,
                "dependency_versions_match": runtime["dependency_lock_status"]["versions_match"],
                "wheel_hash_origin": runtime["dependency_lock_status"]["wheel_hash_origin"],
            },
            "training_diagnostics": diagnostics,
            "resource_accounting": result["resource_accounting"],
        }


def verify_ledger(run_dir: Path, witness_pin: str) -> list[dict]:
    path = run_dir / "evidence" / "attempt-ledger.jsonl"
    if not path.is_file():
        raise FileNotFoundError("signed local attempt ledger is missing")
    previous = ZERO_HASH
    events = []
    for line in path.read_bytes().splitlines():
        envelope = parse_unique_json(line)
        body = verify(envelope, witness_pin)
        if (body.get("schema") != "saal-static-pilot-local-ledger-event-v1"
                or body.get("campaign_id") != CAMPAIGN
                or body.get("sequence") != len(events) + 1
                or body.get("previous") != previous
                or body.get("attempt_id") not in ATTEMPTS):
            raise ValueError("attempt ledger chain/campaign check failed")
        previous = sha(envelope)
        events.append(body)
    if len(events) != 15:
        raise ValueError("expected exactly 15 signed ledger transitions")
    return events


def render_markdown(report: dict) -> str:
    lines = [
        "# SAAL-RSI v0.9.8 Static AdaLoRA Regularization Pilot — run05",
        "",
        "Research screening result. This is public synthetic development evidence, not independent qualification.",
        "",
        f"Campaign: `{report['campaign_id']}`",
        "",
        f"Source commit: `{report['source_commit']}`",
        "",
        f"Status: **{report['campaign_status']}**; qualification: **{report['qualification_status']}**; external deployment: **{str(report['external_deployment_allowed']).lower()}**.",
        "",
        "## What ran",
        "",
        f"Five matched arms used the same pinned ModernBERT snapshot, public training projection, seed `{report['matched_design']['training_seed']}`, sample order, float32 precision, optimizer settings, 200 completed optimizer updates, and 3,200 sample exposures. The A100/high-RAM preflight passed. The 1,152-example development split contains 12 synthetic families and 576 counterfactual pairs; it is correlated public development data, not a protected holdout.",
        "",
        "## Results",
        "",
        "| Arm | Terminal CE | Accuracy | Complete pairs | Worst family | Unauthorized proposals | False-refusal rate | Selected checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in report["arms"]:
        metrics = arm["terminal_metrics"]
        coeff = "—" if arm["orthogonal_coefficient"] is None else str(arm["orthogonal_coefficient"])
        label = "Fixed LoRA r8" if arm["method"] == "fixed-r8" else f"Static AdaLoRA λ={coeff}"
        lines.append(
            f"| {label} | {arm['training_diagnostics']['cross_entropy_last_step']:.3f} "
            f"| {metrics['accuracy']:.1%} | {metrics['complete_pair_accuracy']:.1%} "
            f"| {metrics['worst_family_accuracy']:.1%} | {metrics['unauthorized_proposals']} "
            f"| {metrics['false_refusal_rate']:.1%} "
            f"| {arm['checkpoint_selection']['status']} |"
        )
    lines += [
        "",
        "All five attempts completed, artifacts were recovered, signed ledger transitions verified, and each initial/terminal checkpoint passed the worker's prediction-and-metric replay check. **No arm produced a qualifying trained checkpoint**; the step-zero checkpoint was not accepted as a trained candidate.",
        "",
        "## Interpretation",
        "",
        "The zero-regularization AdaLoRA arm had the best terminal accuracy among the AdaLoRA settings and lower terminal cross-entropy than the nonzero-regularization arms. Increasing the orthogonal coefficient from 0 to 0.5 did not improve this one-seed, 200-update screen. This is consistent with regularization pressure contributing to the loss gap, but the pilot does not isolate a causal effect across seeds or establish a best coefficient.",
        "",
        "The diagnostics narrow the next investigation but do not identify a cause. At λ=0.5, the weighted orthogonal-gradient norm averaged about 6.2% of the cross-entropy gradient norm, with near-zero cosine similarity; at λ=0.01 and 0.1 the ratio averaged about 0.12% and 1.24%. Yet the aggregate effective-update Frobenius norm at step 200 was about 1.11 for static AdaLoRA λ=0 versus 5.23 for fixed r8. Every AdaLoRA step and 199/200 fixed-r8 steps met the configured global clipping condition. Thus the orthogonal term may contribute at λ=0.5, but it does not by itself explain the much smaller SVD updates: initialization, scaling, clipping interactions, or joint head optimization remain open hypotheses. These are worker-reported diagnostics, not independent causal measurements.",
        "",
        "The fixed-r8 arm finished with fewer unauthorized proposals but many false refusals. AdaLoRA reduced false refusals but still made many unauthorized proposals. Both tradeoffs fail the behavioral gate; these numbers do not show a safe model or an overall winning method. Every arm was marked `NO_QUALIFYING_CHECKPOINT`.",
        "",
        "The recorded gradient diagnostics contain separate cross-entropy and weighted-orthogonal gradient norms, their norm ratio and cosine, clipping observations, and effective update norms. They are worker-produced telemetry and have not been independently replayed. Per-run compute-unit usage is not independently measured; the account balance decreased by the amount recorded below.",
        "",
        "The operator recorded matching local and remote SHA-256 readbacks for the signed input bundle and runner in `UPLOAD_VERIFICATION.json`. This is an operator-observed CLI record, not a Google-signed attestation. The bundle itself is excluded from this branch because it contains one-use transport credentials; only its digest, size, signed omission record, and rotation record are retained.",
        "",
        "## Integrity and limits",
        "",
        f"The signed ledger contains {report['integrity']['ledger_events']} verified events. All five artifact ZIPs passed CRC, member-set, internal inventory, admission, receipt, and within-worker replay checks. The source archive matched the signed source closure ({report['integrity']['source_file_count']} files). The one-use transport file is omitted from this public evidence package because it held bearer credentials; its signed size and digest are retained, but its credential values are not published.",
        "",
        f"The Colab balance changed from {report['resource_accounting']['colab_balance_before']:.2f} to {report['resource_accounting']['colab_balance_after']:.2f} CU ({report['resource_accounting']['colab_balance_spent']:.2f} CU observed account delta). The supervisor's conservative rate-based upper bound was {report['resource_accounting']['supervisor_upper_bound_cu']:.3f} CU. No active assignment remained after cleanup.",
        "",
        "Limitations: one seed; short screening run; public synthetic recipes; one operator controlled admission and witness keys; no independent evaluator or hardware attestation; no independent checkpoint replay; no protected final evaluation; no retention or continual-learning study. `wheel_hash_origin` was reported as `NOT_INDEPENDENTLY_VERIFIED`. Qualification remains `NOT_EVALUATED`, and external deployment remains disabled.",
        "",
        "## Next decision",
        "",
        "Do not turn rank scheduling on yet. The static λ=0 result is a candidate for a fresh, separately admitted multi-seed diagnostic only if it is reproduced with matched initializations and improves authorization behavior without creating a false-refusal regression. Keep all current attempts and budgets spent; this report does not authorize another campaign.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--collect-state-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    publication = prepare_public_inputs(run_dir)
    public = run_dir / "public_inputs"
    if args.collect_state_dir:
        state = args.collect_state_dir.resolve()
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(mode=0o700, exist_ok=True)
        for attempt in ATTEMPTS:
            source = state / "received-artifacts" / f"{attempt}.zip"
            destination = artifacts / f"{attempt}.zip"
            if destination.exists() and raw_sha(destination.read_bytes()) != raw_sha(source.read_bytes()):
                raise FileExistsError(f"different artifact already exists for {attempt}")
            if not destination.exists():
                shutil.copyfile(source, destination)
        evidence = run_dir / "evidence"
        evidence.mkdir(mode=0o700, exist_ok=True)
        ledger_source = state / "ledger.jsonl"
        ledger_dest = evidence / "attempt-ledger.jsonl"
        if ledger_dest.exists() and ledger_dest.read_bytes() != ledger_source.read_bytes():
            raise FileExistsError("different ledger already exists")
        if not ledger_dest.exists():
            shutil.copyfile(ledger_source, ledger_dest)

    manifest, trust_store, dataset, closure, witness_pin = verify_public_inputs(run_dir)
    source_validation = verify_source_archive(public, closure)
    trusted_keys, loaded_witness_pin, _ = load_trust_store(public / "trust_store.json")
    if witness_pin != loaded_witness_pin:
        raise ValueError("witness public key changed during validation")
    events = verify_ledger(run_dir, witness_pin)
    event_groups = {attempt: [event for event in events if event["attempt_id"] == attempt]
                    for attempt in ATTEMPTS}

    supervisor = read_json(run_dir / "supervisor_result.json")
    if (supervisor.get("schema") != "saal-static-screen-supervisor-result-v2"
            or supervisor.get("campaign_id") != CAMPAIGN
            or supervisor.get("status") != "ALL_FIVE_TERMINAL_ARTIFACTS_RECEIVED"
            or supervisor.get("stop_exit_code") != 0
            or supervisor.get("status_after_stop", "").find("not found") < 0
            or "Active assignments: 0" not in supervisor.get("usage_after_stop", "")):
        raise ValueError("supervisor cleanup evidence incomplete")
    if len(supervisor.get("attempts", [])) != 5:
        raise ValueError("supervisor attempt summary incomplete")

    config_records = {}
    first_identity = None
    for attempt in ATTEMPTS:
        config = read_json(public / "configs" / attempt / "config.json")
        validate_pilot_config(config)
        if config["identity"]["dataset_digest"] != sha(dataset):
            raise ValueError("configuration is bound to another dataset")
        if config["orthogonal_coefficient"] != EXPECTED_COEFFICIENTS[attempt]:
            raise ValueError("regularization arm differs from the preregistered value")
        expected_method = "fixed-r8" if attempt == ATTEMPTS[0] else "adalora-static-r12"
        if config["method"] != expected_method:
            raise ValueError("training method differs from the preregistered arm")
        if config["steps"] != 200 or config["sample_exposures"] != 3200:
            raise ValueError("training budget differs from the registered pilot")
        if config["identity"]["precision"] != "float32":
            raise ValueError("unexpected training precision")
        if config["identity"]["model_spec"]["model_revision"] != "11aa1a9b1d9dd6e8cded5040fe217c9b6ef92e88":
            raise ValueError("unexpected pretrained model revision")
        stable = {
            key: config["identity"][key]
            for key in ("seed", "sampler_seed", "dropout_seed", "sample_order_digest",
                        "model_digest", "tokenizer_digest", "dataset_digest",
                        "dependency_digest", "metric_policy_digest", "source_digest",
                        "target_modules", "batch_size", "accumulation", "class_balance",
                        "gradient_clip", "head_learning_rate", "adapter_learning_rate",
                        "weight_decay", "precision")
        }
        if first_identity is None:
            first_identity = stable
        elif stable != first_identity:
            raise ValueError("matched training identity differs between experiment arms")
        config_records[attempt] = config

    arms = []
    for attempt in ATTEMPTS:
        artifact = run_dir / "artifacts" / f"{attempt}.zip"
        identity_record = read_json(run_dir / f"{attempt}-identity.json")
        arm = verify_artifact(
            run_dir, artifact, attempt, config_records[attempt], dataset,
            trusted_keys, witness_pin, identity_record, event_groups[attempt],
        )
        arms.append(arm)

    auth = read_json(run_dir / "OPERATOR_AUTHORIZATION.json")
    source_identity = read_json(run_dir / "SOURCE_IDENTITY.json")
    data_identity = read_json(run_dir / "dataset_identity.json")
    upload_verification = read_json(run_dir / "UPLOAD_VERIFICATION.json")
    expected_uploads = {"signed-input-bundle", "remote-runner"}
    upload_records = upload_verification.get("uploads", [])
    if (upload_verification.get("schema") != "saal-colab-upload-verification-v1"
            or upload_verification.get("campaign_id") != CAMPAIGN
            or upload_verification.get("status") != "OPERATOR_OBSERVED_MATCH"
            or upload_verification.get("independent_attestation") is not False
            or {item.get("name") for item in upload_records} != expected_uploads
            or any(item.get("remote_readback") != "MATCH"
                   or not re.fullmatch(r"[0-9a-f]{64}", item.get("local_sha256", ""))
                   or not isinstance(item.get("local_size_bytes"), int)
                   for item in upload_records)
            or upload_verification.get("credential_handling", {}).get(
                "input_bundle_in_public_results") is not False):
        raise ValueError("upload evidence is incomplete or claims unsupported attestation")
    if (auth["campaign_id"] != CAMPAIGN or auth["qualification_status"] != "NOT_EVALUATED"
            or auth["external_deployment_allowed"] is not False
            or source_identity["commit"] != manifest["source_commit"]
            or data_identity["dataset_digest"] != sha(dataset)
            or data_identity["regression_included"] is not False):
        raise ValueError("campaign identity or scope differs from the signed admission")

    balance_before = float(supervisor["colab_balance_before"])
    balance_after = float(supervisor["colab_balance_after"])
    report = {
        "schema": "saal-static-regularization-pilot-report-v1",
        "campaign_id": CAMPAIGN,
        "campaign_status": "COMPLETED_ALL_FIVE_ARTIFACTS_VERIFIED",
        "qualification_status": "NOT_EVALUATED",
        "external_deployment_allowed": False,
        "source_commit": source_identity["commit"],
        "source_digest": source_identity["source_digest"],
        "model_digest": source_identity["model_digest"],
        "model_revision": config_records[ATTEMPTS[0]]["identity"]["model_spec"]["model_revision"],
        "dataset_digest": sha(dataset),
        "metric_policy_digest": config_records[ATTEMPTS[0]]["identity"]["metric_policy_digest"],
        "matched_design": {
            "training_seed": first_identity["seed"],
            "sampler_seed": first_identity["sampler_seed"],
            "dropout_seed": first_identity["dropout_seed"],
            "precision": first_identity["precision"],
            "optimizer_updates_per_arm": 200,
            "sample_exposures_per_arm": 3200,
            "development_records": data_identity["development_count"],
            "public_families": len(arms[0]["terminal_metrics"]["support"]["by_family"]),
            "counterfactual_pairs": arms[0]["terminal_counterfactual_pairs"],
            "review_status": data_identity["review_status"],
        },
        "arms": arms,
        "integrity": {
            "input_manifest_signature_verified": True,
            "input_manifest_sha256": raw_sha((public / "run_manifest.json").read_bytes()),
            "public_input_files_verified": len(publication["included_files"]),
            "credential_bearing_transport_values_published": False,
            "upload_verification": {
                "operator_observed_remote_hash_matches": True,
                "independent_attestation": False,
                "credential_bundle_published": False,
                "record_path": "UPLOAD_VERIFICATION.json",
            },
            "source_file_count": source_validation["file_count"],
            "source_total_file_bytes": source_validation["total_file_bytes"],
            "source_closure_digest": source_validation["closure_digest"],
            "ledger_events": len(events),
            "ledger_chain_verified": True,
            "artifact_zips_verified": len(arms),
            "worker_checkpoint_replay_verified": all(
                all(flag for checkpoint in arm["within_worker_replay"].values()
                    for flag in checkpoint.values()) for arm in arms
            ),
            "independent_replay": False,
            "custody_independent": False,
        },
        "resource_accounting": {
            "colab_balance_before": balance_before,
            "colab_balance_after": balance_after,
            "colab_balance_spent": round(balance_before - balance_after, 3),
            "supervisor_upper_bound_cu": supervisor["estimated_upper_bound_cu_since_admission"],
            "signed_campaign_cap_cu": auth["aggregate_compute_unit_cap"],
            "supervisor_stop_threshold_cu": 12.5,
            "active_assignments_after_stop": 0,
            "runtime_after_stop": "NOT_FOUND",
            "per_arm_actual_compute_units": "NOT_INDEPENDENTLY_MEASURED",
        },
        "limitations": [
            "single-seed engineering screen; no inferential population claim",
            "public synthetic development families are correlated and not independently authored holdouts",
            "labels were oracle-checked, not independently human-reviewed for this experiment",
            "hardware observations and training diagnostics are worker-produced, not independent attestation",
            "checkpoint prediction/metric replay ran inside each worker; no independent replayer was used",
            "no protected final evaluation, retention study, mechanism ablation, or deployment qualification",
            "package wheel hashes were hash-locked for installation but their upstream provenance was not independently verified",
        ],
    }
    report_path = run_dir / "RUN_REPORT.json"
    markdown_path = run_dir / "REPORT.md"
    report_path.write_bytes(canonical(report) + b"\n")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "arms": [{
            "attempt_id": arm["attempt_id"],
            "method": arm["method"],
            "orthogonal_coefficient": arm["orthogonal_coefficient"],
            "selected_checkpoint": arm["checkpoint_selection"]["status"],
            "accuracy": arm["terminal_metrics"]["accuracy"],
            "complete_pair_accuracy": arm["terminal_metrics"]["complete_pair_accuracy"],
            "unauthorized_proposals": arm["terminal_metrics"]["unauthorized_proposals"],
            "false_refusals": arm["terminal_metrics"]["false_refusals"],
        } for arm in arms],
        "ledger_events": len(events),
        "source_files_verified": source_validation["file_count"],
        "account_cu_spent": report["resource_accounting"]["colab_balance_spent"],
        "qualification_status": "NOT_EVALUATED",
        "external_deployment_allowed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
