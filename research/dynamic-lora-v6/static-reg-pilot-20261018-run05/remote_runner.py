"""Pinned one-session Colab runner for the five-arm static regularizer screen."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = Path("/content/saal-static-screen")
INPUTS = BASE / "inputs"
SOURCE = BASE / "source"
RUNNER_FILE = Path(__file__).resolve()
MAX_BUNDLE = 100_000_000


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("input extraction directory already exists")
    destination.mkdir(mode=0o700, parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if (not infos or len(names) != len(set(names)) or archive.testzip() is not None
                or sum(item.file_size for item in infos) > MAX_BUNDLE
                or any(item.filename.startswith("/") or ".." in Path(item.filename).parts
                       or ((item.external_attr >> 16) & 0o170000) == 0o120000
                       for item in infos)):
            raise ValueError("input package rejected")
        archive.extractall(destination)


def verify_extracted_source(source_root: Path, inputs_root: Path, manifest: dict) -> dict:
    from saal_rsi.spec import parse_unique_json, sha

    closure = parse_unique_json((inputs_root / "source_closure.json").read_bytes())
    if (type(closure) is not dict
            or set(closure) != {"schema", "git_commit", "scope_paths", "file_count",
                                "total_file_bytes", "archive_sha256", "files"}
            or closure["schema"] != "saal-static-pilot-source-closure-v1"
            or type(closure["scope_paths"]) is not list
            or closure["git_commit"] != manifest["source_commit"]
            or closure["archive_sha256"] != manifest["source_archive_sha256"]
            or type(closure["files"]) is not dict
            or closure["file_count"] != len(closure["files"])):
        raise PermissionError("source closure identity rejected")
    observed = {}
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("source closure symlink rejected")
        if path.is_dir() and path.name == "__pycache__":
            continue
        if path.suffix == ".pyc" and "__pycache__" in path.parts:
            continue
        if path.is_file():
            raw = path.read_bytes()
            observed[path.relative_to(source_root).as_posix()] = {
                "size": len(raw), "sha256": digest(raw),
            }
    expected = {}
    for name, record in closure["files"].items():
        if (type(name) is not str or type(record) is not dict
                or set(record) != {"git_blob_oid", "size", "sha256"}):
            raise ValueError("source closure file record rejected")
        if (type(record["size"]) is not int or record["size"] < 0
                or not isinstance(record["git_blob_oid"], str)
                or not isinstance(record["sha256"], str) or len(record["sha256"]) != 64):
            raise ValueError("source closure metadata rejected")
        expected[name] = {"size": record["size"], "sha256": record["sha256"]}
    if observed != expected:
        raise PermissionError("extracted source differs from admitted Git-tree closure")
    if sum(item["size"] for item in observed.values()) != closure["total_file_bytes"]:
        raise ValueError("source closure total size mismatch")
    return {"file_count": len(observed), "total_file_bytes": sum(x["size"] for x in observed.values()),
            "closure_digest": sha(closure)}


def verify_input_files() -> dict:
    from saal_rsi.adaptive_static_pilot_v5 import load_trust_store
    from saal_rsi.model_contract import verify
    from saal_rsi.spec import parse_unique_json

    trust_path = INPUTS / "trust_store.json"
    trusted, witness_pin, _ = load_trust_store(trust_path)
    envelope = parse_unique_json((INPUTS / "run_manifest.json").read_bytes())
    pin = trusted.get(envelope.get("key_sha256"))
    if pin is None:
        raise PermissionError("run manifest signer absent from trust store")
    manifest = verify(envelope, pin)
    expected_fields = {
        "schema", "campaign_id", "source_commit", "source_archive_sha256",
        "runner_sha256", "operator_authority_sha256", "campaign_preparer_sha256",
        "files", "issued_at", "expires_at",
        "attempts",
        "authorization_scope", "qualification_status", "external_deployment_allowed",
    }
    if (type(manifest) is not dict or set(manifest) != expected_fields
            or manifest["schema"] != "saal-static-screen-input-manifest-v1"
            or manifest["authorization_scope"] != "SINGLE_OPERATOR_PUBLIC_DEVELOPMENT_PILOT"
            or manifest["qualification_status"] != "NOT_EVALUATED"
            or manifest["external_deployment_allowed"] is not False
            or not manifest["issued_at"] <= int(time.time()) < manifest["expires_at"]):
        raise PermissionError("run manifest scope or validity rejected")
    if digest(RUNNER_FILE.read_bytes()) != manifest["runner_sha256"]:
        raise PermissionError("remote runner bytes differ from local manifest")
    actual_files = {}
    for path in sorted(INPUTS.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink input rejected")
        if path.is_file() and path.name != "run_manifest.json":
            relative = path.relative_to(INPUTS).as_posix()
            actual_files[relative] = {
                "size": path.stat().st_size,
                "sha256": digest(path.read_bytes()),
            }
    if actual_files != manifest["files"]:
        raise PermissionError("input files differ from signed inventory")
    source_archive = INPUTS / "source.tar.gz"
    if digest(source_archive.read_bytes()) != manifest["source_archive_sha256"]:
        raise PermissionError("source archive digest mismatch")
    return {"manifest": manifest, "admission_keys": trusted, "witness_pin": witness_pin}


def install_locked_environment(source_root: Path) -> Path:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("admitted Colab Python 3.13 required")
    lock = source_root / "configs/dependency-locks/pretrained-linux-x86_64-py313.txt"
    env = Path("/content/saal-static-env")
    if env.exists():
        raise FileExistsError("fresh one-session environment required")
    subprocess.run(["uv", "venv", "--python", sys.executable, str(env)],
                   check=True, timeout=120, capture_output=True, text=True)
    subprocess.run([
        "uv", "pip", "install", "--python", str(env / "bin/python"),
        "--require-hashes", "--only-binary", ":all:", "-r", str(lock),
    ], check=True, timeout=1800, capture_output=True, text=True)
    return env / "bin/python"


def request_json(url: str, token: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.loads(response.read(1_000_001))
    if set(data) != {"result"}:
        raise RuntimeError("authority rejected request")
    return data["result"]


def upload_artifact(url: str, token: str, path: Path) -> dict:
    raw = path.read_bytes()
    if not 1 <= len(raw) <= 100_000_000:
        raise ValueError("attempt artifact exceeds transfer limit")
    request = urllib.request.Request(
        url, data=raw, headers={"Content-Type": "application/zip",
                                "Authorization": "Bearer " + token}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read(1_000_001))
    if set(data) != {"result"} or data["result"].get("status") != "RECEIVED":
        raise RuntimeError("artifact authority receipt rejected")
    if data["result"].get("sha256") != digest(raw):
        raise ValueError("artifact transfer receipt digest mismatch")
    return data["result"]


def archive_attempt(out: Path, attempt: str) -> Path:
    destination = BASE / "artifacts" / f"{attempt}.zip"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("attempt archive already exists")
    total = 0
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(out.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink in attempt output")
            if path.is_file():
                total += path.stat().st_size
                if total > 100_000_000:
                    raise ValueError("attempt artifact inventory exceeds admission")
                z.write(path, path.relative_to(out).as_posix())
    if destination.stat().st_size > 100_000_000:
        raise ValueError("compressed attempt artifact exceeds admission")
    return destination


def isolated_worker_main() -> None:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SOURCE / "src"))
    from saal_rsi.adaptive_static_pilot_v5 import execute_admitted_static_pilot
    from saal_rsi.spec import parse_unique_json

    verified = verify_input_files()
    manifest = verified["manifest"]
    data = parse_unique_json((INPUTS / "dataset.json").read_bytes())
    transport = parse_unique_json((INPUTS / "transport.json").read_bytes())
    cache = BASE / "model-cache"
    statuses = []
    for item in manifest["attempts"]:
        attempt = item["attempt_id"]
        config = parse_unique_json((INPUTS / "configs" / attempt / "config.json").read_bytes())
        admission = parse_unique_json((INPUTS / "configs" / attempt / "admission.json").read_bytes())
        out = BASE / "results" / attempt
        out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if out.exists():
            raise FileExistsError("attempt result already exists")
        authority_entry = transport["attempts"][attempt]

        class AuthorityClient:
            def __init__(self, selected_attempt, selected_entry):
                self.attempt = selected_attempt
                self.entry = selected_entry

            def consume_attempt(self, bindings, nonce):
                return request_json(
                    self.entry["url"] + "/v1/consume/" + self.attempt,
                    self.entry["token"], {"bindings": bindings, "challenge": nonce},
                )

            def finalize_attempt(self, receipt, outcome, outcome_digest, nonce):
                return request_json(
                    self.entry["url"] + "/v1/finalize/" + self.attempt,
                    self.entry["token"], {"attempt_receipt": receipt,
                        "outcome": outcome, "outcome_digest": outcome_digest,
                        "finish_nonce": nonce},
                )

        try:
            result = execute_admitted_static_pilot(
                config, data, admission, attempt_authority=AuthorityClient(attempt, authority_entry),
                model_cache=cache, output_dir=out,
            )
            outcome = "COMPLETED"
            detail = {"selected_checkpoint_step": result["selected_checkpoint_step"],
                      "terminal_accuracy": result["terminal"]["metrics"]["accuracy"]}
        except Exception as exc:  # noqa: BLE001 - classify and preserve every spent attempt
            outcome = "UNCONFIRMED"
            terminal_path = out / "terminal_receipt.json"
            if terminal_path.is_file():
                terminal = parse_unique_json(terminal_path.read_bytes())
                from saal_rsi.model_contract import verify
                terminal_body = verify(terminal, verified["witness_pin"])
                outcome = terminal_body["status"]
                if outcome == "COMPLETED":
                    outcome = "COMPLETION_ARTIFACT_COMMIT_UNCERTAIN"
            detail = {"failure_type": type(exc).__name__,
                      "terminal_receipt": outcome != "UNCONFIRMED"}

        # Preserve whatever the consumed attempt produced. Preflight rejection
        # does not consume the attempt and therefore stops the cohort here.
        if outcome == "UNCONFIRMED":
            statuses.append({"attempt_id": attempt, "status": outcome, **detail})
            print(json.dumps({"event": "attempt_stopped", "attempt_id": attempt,
                              "status": outcome, **detail}), flush=True)
            break
        archive = archive_attempt(out, attempt)
        receipt = upload_artifact(
            authority_entry["url"] + "/v1/artifact/" + attempt,
            authority_entry["token"], archive,
        )
        statuses.append({"attempt_id": attempt, "status": outcome,
                         "artifact_sha256": receipt["sha256"], **detail})
        print(json.dumps({"event": "attempt_terminal", "attempt_id": attempt,
                          "status": outcome, **detail}), flush=True)
        if "result" in locals():
            del result
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - CUDA cleanup is best-effort after evidence is saved
            print(json.dumps({"event": "cuda_cache_cleanup_unavailable",
                              "attempt_id": attempt}), flush=True)
        if outcome == "FAILED":
            # A fully finalized, recorded failure is preserved. The next arm
            # is a separate fresh authorization, not a retry of this attempt.
            continue
        if outcome == "COMPLETION_ARTIFACT_COMMIT_UNCERTAIN":
            break
    final = {"schema": "saal-static-screen-remote-summary-v1",
             "campaign_id": manifest["campaign_id"], "attempts": statuses,
             "qualification_status": "NOT_EVALUATED",
             "external_deployment_allowed": False}
    (BASE / "remote_summary.json").write_text(json.dumps(final, sort_keys=True, indent=2))
    print(json.dumps({"event": "cohort_summary", "attempts": statuses,
                      "qualification_status": "NOT_EVALUATED",
                      "external_deployment_allowed": False}), flush=True)
    if len(statuses) != len(manifest["attempts"]):
        raise RuntimeError("cohort stopped before all admitted arms")


def setup_main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        isolated_worker_main()
        return
    archive = Path("/content/saal-static-inputs.zip")
    if not archive.is_file() or archive.stat().st_size > MAX_BUNDLE:
        raise FileNotFoundError("uploaded pilot input archive missing")
    BASE.mkdir(mode=0o700, parents=True, exist_ok=False)
    safe_extract(archive, INPUTS)
    # Verify the admission-bound environment before starting the GPU worker.
    source_archive = INPUTS / "source.tar.gz"
    SOURCE.mkdir(mode=0o700, parents=True, exist_ok=False)
    import tarfile
    with tarfile.open(source_archive, "r:gz") as source_tar:
        members = source_tar.getmembers()
        if any(member.name.startswith("/") or ".." in Path(member.name).parts
               or member.issym() or member.islnk() for member in members):
            raise ValueError("unsafe source archive")
        if sum(member.size for member in members if member.isfile()) > MAX_BUNDLE:
            raise ValueError("source closure exceeds extraction bound")
        source_tar.extractall(SOURCE, filter="data")
    sys.path.insert(0, str(SOURCE / "src"))
    verified = verify_input_files()
    source_closure = verify_extracted_source(SOURCE, INPUTS, verified["manifest"])
    print(json.dumps({"event": "source_closure_verified", **source_closure}), flush=True)
    env_python = install_locked_environment(SOURCE)
    env = dict(os.environ)
    env["SAAL_STATIC_PILOT_TRUST_STORE"] = str(INPUTS / "trust_store.json")
    env["SAAL_HASH_LOCK_ENFORCED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    process = subprocess.run([str(env_python), "-I", str(RUNNER_FILE), "--worker"],
                             env=env, check=False)
    if process.returncode:
        raise RuntimeError("pinned pilot worker failed; inspect captured Colab evidence")


if __name__ == "__main__":
    try:
        setup_main()
    except BaseException as exc:
        print(json.dumps({"event": "runner_failure", "failure_type": type(exc).__name__,
                          "detail": str(exc)[:1000]}), flush=True)
        raise
