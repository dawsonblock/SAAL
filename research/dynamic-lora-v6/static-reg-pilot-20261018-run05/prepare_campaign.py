"""Create a fresh, private, one-use development admission and Colab bundle."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import uuid
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
CAMPAIGN_ID = "saal098-static-reg-20261018-05"
SOURCE_PATHS = (
    "src/saal_rsi",
    "configs/dependency-locks/pretrained-linux-x86_64-py313.txt",
    "configs/pretrained",
    "pyproject.toml",
)
ATTEMPTS = [
    ("saal098r05-arm-fixedr8", "fixed-r8", None),
    ("saal098r05-arm-orth0", "adalora-static-r12", 0.0),
    ("saal098r05-arm-orth001", "adalora-static-r12", 0.01),
    ("saal098r05-arm-orth01", "adalora-static-r12", 0.1),
    ("saal098r05-arm-orth05", "adalora-static-r12", 0.5),
]
STATE = Path.home() / ".local/state/saal-rsi/static-reg-pilot-20261018-run05"
PRIVATE_INPUTS = STATE / "private-inputs"


def raw_sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write(path: Path, raw: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def file_inventory(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in campaign bundle")
        if path.is_file() and path.name != "run_manifest.json":
            data = path.read_bytes()
            result[path.relative_to(root).as_posix()] = {
                "size": len(data), "sha256": raw_sha(data),
            }
    return result


def git_source_inventory(commit: str) -> dict:
    raw = subprocess.run(
        ["git", "ls-tree", "-rl", commit, "--", *SOURCE_PATHS], cwd=ROOT,
        check=True, capture_output=True,
    ).stdout
    files = {}
    for line in raw.splitlines():
        metadata, path = line.split(b"\t", 1)
        _mode, object_type, object_id, size = metadata.split()
        if object_type != b"blob":
            raise ValueError("unexpected non-blob in selected source closure")
        files[path.decode("utf-8")] = {
            "git_blob_oid": object_id.decode("ascii"), "size": int(size),
        }
    if not files:
        raise ValueError("empty source closure")
    return files


def create_source_archive(commit: str, target: Path) -> str:
    proc = subprocess.Popen(
        ["git", "archive", "--format=tar", commit, *SOURCE_PATHS], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        with (target.open("xb") as output,
              gzip.GzipFile(filename="", mode="wb", fileobj=output,
                            mtime=0, compresslevel=6) as zipped):
            shutil.copyfileobj(proc.stdout, zipped, length=1024 * 1024)
        stderr = proc.stderr.read()
        code = proc.wait()
        if code:
            raise RuntimeError(stderr.decode(errors="replace")[-1000:])
    except BaseException:
        proc.kill()
        proc.wait()
        target.unlink(missing_ok=True)
        raise
    return raw_sha(target.read_bytes())


def archive_source_inventory(target: Path, git_files: dict) -> dict:
    files = {}
    with tarfile.open(target, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.name not in git_files:
                raise ValueError("source archive contains an undeclared member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("source archive member unavailable")
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
            expected = git_files[member.name]
            if size != expected["size"]:
                raise ValueError("source archive member size differs from Git tree")
            files[member.name] = {**expected, "sha256": digest.hexdigest()}
    if set(files) != set(git_files):
        raise ValueError("source archive omits a declared Git-tree member")
    return files


def main() -> None:
    if STATE.exists() or STATE.is_symlink() or RUN.joinpath("inputs.zip").exists():
        raise FileExistsError("fresh campaign state required; no previous state reused")
    STATE.mkdir(mode=0o700, parents=True)
    PRIVATE_INPUTS.mkdir(mode=0o700)
    for path in (RUN, STATE, PRIVATE_INPUTS):
        os.chmod(path, 0o700)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    remote = subprocess.run(["git", "ls-remote", "origin", "refs/heads/repair/saal-v098-dynamic-lora-core"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()
    if not remote or remote[0] != head:
        raise RuntimeError("published source commit differs from this campaign input")

    from saal_rsi.adaptive_contract_v5 import (
        build_fixed_r8_reference,
        build_static_adalora_pilot,
        scientific_source_digest,
    )
    from saal_rsi.adaptive_dataset import training_projection
    from saal_rsi.adaptive_static_pilot_v5 import ADMISSION_SCHEMA
    from saal_rsi.epoch_sampler import sample_ids
    from saal_rsi.model_contract import sign
    from saal_rsi.numerical_metrics_v2 import POLICY_DIGEST
    from saal_rsi.pretrained_contract import model_spec, targets
    from saal_rsi.spec import canonical, parse_unique_json, sha

    source_root = PRIVATE_INPUTS / "source.tar.gz"
    git_files = git_source_inventory(head)
    source_archive_digest = create_source_archive(head, source_root)
    source_closure = {
        "schema": "saal-static-pilot-source-closure-v1",
        "git_commit": head,
        "scope_paths": list(SOURCE_PATHS),
        "file_count": len(git_files),
        "total_file_bytes": sum(item["size"] for item in git_files.values()),
        "archive_sha256": source_archive_digest,
        "files": archive_source_inventory(source_root, git_files),
    }
    write(PRIVATE_INPUTS / "source_closure.json", canonical(source_closure))
    population = parse_unique_json((ROOT / "research/evidence-closure-v5/dataset.json").read_bytes())
    dataset = training_projection(population)
    dataset_bytes = canonical(dataset)
    write(PRIVATE_INPUTS / "dataset.json", dataset_bytes)
    dataset_digest = sha(dataset)

    admission_key = Ed25519PrivateKey.generate()
    witness_key = Ed25519PrivateKey.generate()
    keys_dir = STATE / "keys"
    keys_dir.mkdir(mode=0o700)
    write(keys_dir / "admission.ed25519", admission_key.private_bytes_raw())
    write(keys_dir / "attempt-authority.ed25519", witness_key.private_bytes_raw())
    admission_public = admission_key.public_key().public_bytes_raw().hex()
    witness_public = witness_key.public_key().public_bytes_raw().hex()
    admission_key_id = hashlib.sha256(bytes.fromhex(admission_public)).hexdigest()
    witness_key_id = hashlib.sha256(bytes.fromhex(witness_public)).hexdigest()
    if admission_key_id == witness_key_id:
        raise RuntimeError("distinct authority keys required")
    trust_store = {
        "schema": "saal-static-pilot-trust-store-v1",
        "admission_keys": {admission_key_id: {
            "public_key": admission_public, "roles": ["static-pilot-admission"],
        }},
        "attempt_authority": {"public_key": witness_public,
                              "role": "static-pilot-attempt-authority"},
    }
    trust_bytes = canonical(trust_store)
    write(PRIVATE_INPUTS / "trust_store.json", trust_bytes)

    lock_path = ROOT / "configs/dependency-locks/pretrained-linux-x86_64-py313.txt"
    lock_digest = "sha256:" + raw_sha(lock_path.read_bytes())
    spec = model_spec("answerdotai/ModernBERT-base", method="none", backend="colab",
                      precision="float32", budget="static-reg-pilot-v5")
    seed, sampler_seed, dropout_seed = 317, 331, 337
    batch_size, accumulation = 8, 2
    exposure_count = 200 * batch_size * accumulation
    train = [row for row in dataset["records"] if row["split"] == "train"]
    order_digest = sha(sample_ids(train, "complete-balanced-epochs-v1", sampler_seed,
                                  exposure_count))
    identity = {
        "model_spec": spec,
        "model_digest": spec["model_files_digest"],
        "tokenizer_digest": spec["tokenizer_digest"],
        "dataset_digest": dataset_digest,
        "source_digest": scientific_source_digest(),
        "dependency_digest": lock_digest,
        "metric_policy_digest": POLICY_DIGEST,
        "sample_order_digest": order_digest,
        "seed": seed,
        "sampler_seed": sampler_seed,
        "dropout_seed": dropout_seed,
        "batch_size": batch_size,
        "accumulation": accumulation,
        "adapter_learning_rate": 0.0003,
        "head_learning_rate": 0.0003,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "target_modules": targets("modernbert-seqcls-v1"),
        "precision": "float32",
        "class_balance": False,
    }

    now = int(time.time())
    expires = now + 6 * 3600
    budget = {
        # At the observed A100 high-RAM rate of 6.77 CU/hour, 1,000 seconds
        # stays below the per-arm 2.0-CU cap. A local watchdog bounds the whole
        # session separately; this is an upper bound, not a billing estimate.
        "max_wall_seconds": 1000,
        "max_compute_units": 2.0,
        "compute_unit": "colab-compute-unit",
        "max_output_bytes": 100_000_000,
    }
    runtime = {
        "accelerator": "A100",
        "minimum_gpu_memory_bytes": 30 * 1024**3,
        "minimum_free_gpu_memory_bytes": 6 * 1024**3,
        "minimum_host_memory_bytes": 32 * 1024**3,
        "require_dependency_hash_lock": True,
    }
    attempts_config = []
    service_attempts = []
    authority_tokens = {}
    for attempt_id, method, coefficient in ATTEMPTS:
        config = (build_fixed_r8_reference(identity) if method == "fixed-r8"
                  else build_static_adalora_pilot(identity, coefficient))
        config_dir = PRIVATE_INPUTS / "configs" / attempt_id
        config_bytes = canonical(config)
        write(config_dir / "config.json", config_bytes)
        payload = {
            "schema": ADMISSION_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "attempt_id": attempt_id,
            "issued_at": now,
            "expires_at": expires,
            "configuration_digest": sha(config),
            "dataset_digest": dataset_digest,
            "source_digest": identity["source_digest"],
            "model_digest": identity["model_digest"],
            "tokenizer_digest": identity["tokenizer_digest"],
            "dependency_digest": identity["dependency_digest"],
            "metric_policy_digest": identity["metric_policy_digest"],
            "max_optimizer_updates": config["steps"],
            "max_sample_exposures": config["sample_exposures"],
            "budget": dict(budget),
            "runtime": dict(runtime),
        }
        envelope = sign(payload, admission_key)
        admission_bytes = canonical(envelope)
        write(config_dir / "admission.json", admission_bytes)
        admission_digest = sha(envelope)
        bindings = {
            "schema": "saal-static-pilot-attempt-bindings-v1",
            "campaign_id": CAMPAIGN_ID,
            "attempt_id": attempt_id,
            "admission_digest": admission_digest,
            "configuration_digest": sha(config),
            "dataset_digest": dataset_digest,
            "source_digest": identity["source_digest"],
            "dependency_digest": identity["dependency_digest"],
            "metric_policy_digest": identity["metric_policy_digest"],
            "model_digest": identity["model_digest"],
            "budget_digest": sha(budget),
        }
        token = uuid.uuid4().hex + uuid.uuid4().hex
        authority_tokens[attempt_id] = token
        attempts_config.append({
            "attempt_id": attempt_id,
            "admission_digest": admission_digest,
            "configuration_digest": sha(config),
            "method": config["method"],
            "orthogonal_coefficient": config["orthogonal_coefficient"],
        })
        service_attempts.append({"attempt_id": attempt_id, "bindings": bindings,
                                 "admission": payload, "token": token})
        service_config_attempt = {
            "attempt_id": attempt_id,
            "admission": payload,
            "bindings": bindings,
        }
        # Store only non-secret authority identity in the public recovery record.
        (RUN / f"{attempt_id}-identity.json").write_bytes(canonical(service_config_attempt))

    remote_url = os.environ.get("SAAL_RECEIPT_BASE_URL", "").rstrip("/")
    if not remote_url.startswith("https://") or len(remote_url) > 300:
        raise ValueError("short-lived HTTPS receipt tunnel URL required")
    transport = {"schema": "saal-static-screen-transport-v1", "attempts": {
        attempt_id: {"url": remote_url, "token": authority_tokens[attempt_id]}
        for attempt_id, _, _ in ATTEMPTS
    }}
    write(PRIVATE_INPUTS / "transport.json", canonical(transport))

    service_config = {"campaign_id": CAMPAIGN_ID, "attempts": service_attempts}
    write(STATE / "service-config.json", canonical(service_config))

    authorization = {
        "schema": "saal-single-operator-research-authorization-record-v1",
        "campaign_id": CAMPAIGN_ID,
        "authorization_source": "direct user instruction to find a way to run after the prior infrastructure failure; this is a fresh campaign, not a retry or budget reset",
        "operator_control": "ONE_OPERATOR_CONTROLS_ADMISSION_AND_ATTEMPT_AUTHORITY",
        "authority_key_ids": {"admission": admission_key_id, "attempt_authority": witness_key_id},
        "attempt_count": 5,
        "per_attempt_compute_unit_cap": 2.0,
        "aggregate_compute_unit_cap": 13.5,
        "per_attempt_wall_clock_cap_seconds": 1000,
        "aggregate_wall_clock_cap_seconds": 7178,
        "billing_note": "Fresh 13.5 CU maximum for this separately admitted campaign; prior A100 probe and run04 usage remain spent and are not reset or refunded. Session watchdog stops at 12.5 CU equivalent to reserve cleanup time.",
        "runtime": "Colab A100 high RAM, required >=30 GiB GPU and >=32 GiB host RAM",
        "protocol": "static AdaLoRA r12 orthogonal coefficients 0.0, 0.01, 0.1, 0.5 plus matched fixed-r8; 200 optimizer updates each; same ModernBERT snapshot, data projection, seed and exposure stream; float32 training to avoid the prior FP16 nonfinite-gradient failure",
        "data_scope": "public synthetic train and development projection only; regression and protected final tasks excluded",
        "qualification_status": "NOT_EVALUATED",
        "external_deployment_allowed": False,
        "independent_custody": False,
        "automatic_retries": False,
        "issued_at": now,
        "expires_at": expires,
    }
    write(RUN / "OPERATOR_AUTHORIZATION.json", canonical(authorization), mode=0o644)
    write(RUN / "trust_store.json", trust_bytes, mode=0o644)
    write(RUN / "dataset_identity.json", canonical({
        "dataset_digest": dataset_digest,
        "parent_population_digest": dataset["parent_population_digest"],
        "record_count": len(dataset["records"]),
        "train_count": len(train),
        "development_count": sum(r["split"] == "development" for r in dataset["records"]),
        "regression_included": False,
        "review_status": dataset["review_status"],
        "origin": dataset["origin"],
    }), mode=0o644)
    source_manifest = {
        "schema": "saal-static-screen-source-identity-v1",
        "commit": head,
        "remote_branch": "repair/saal-v098-dynamic-lora-core",
        "archive_sha256": source_archive_digest,
        "source_closure_sha256": sha(source_closure),
        "source_digest": identity["source_digest"],
        "dependency_lock_digest": lock_digest,
        "model_digest": identity["model_digest"],
        "tokenizer_digest": identity["tokenizer_digest"],
        "metric_policy_digest": identity["metric_policy_digest"],
        "runner_sha256": raw_sha((RUN / "remote_runner.py").read_bytes()),
        "runner_authority_server_sha256": raw_sha((RUN / "operator_authority.py").read_bytes()),
        "campaign_preparer_sha256": raw_sha(Path(__file__).read_bytes()),
    }
    write(RUN / "SOURCE_IDENTITY.json", canonical(source_manifest), mode=0o644)

    manifest_payload = {
        "schema": "saal-static-screen-input-manifest-v1",
        "campaign_id": CAMPAIGN_ID,
        "source_commit": head,
        "source_archive_sha256": source_archive_digest,
        "runner_sha256": source_manifest["runner_sha256"],
        "operator_authority_sha256": source_manifest["runner_authority_server_sha256"],
        "campaign_preparer_sha256": source_manifest["campaign_preparer_sha256"],
        "files": file_inventory(PRIVATE_INPUTS),
        "issued_at": now,
        "expires_at": expires,
        "attempts": [
            {"attempt_id": item["attempt_id"], "admission_digest": item["admission_digest"]}
            for item in attempts_config
        ],
        "authorization_scope": "SINGLE_OPERATOR_PUBLIC_DEVELOPMENT_PILOT",
        "qualification_status": "NOT_EVALUATED",
        "external_deployment_allowed": False,
    }
    write(PRIVATE_INPUTS / "run_manifest.json", canonical(sign(manifest_payload, admission_key)))
    bundle = RUN / "inputs.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PRIVATE_INPUTS.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink in campaign bundle")
            if not path.is_file():
                continue
            relative = path.relative_to(PRIVATE_INPUTS).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            archive.writestr(info, path.read_bytes())
    if bundle.stat().st_size > 100_000_000:
        raise ValueError("signed campaign input bundle exceeds 100 MB")
    print(json.dumps({
        "campaign_id": CAMPAIGN_ID, "source_commit": head,
        "source_archive_sha256": source_archive_digest,
        "dataset_digest": dataset_digest, "source_digest": identity["source_digest"],
        "attempts": attempts_config,
        "public_data_records": len(dataset["records"]),
        "aggregate_spend_cap_compute_units": 13.5,
        "qualification_status": "NOT_EVALUATED",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
