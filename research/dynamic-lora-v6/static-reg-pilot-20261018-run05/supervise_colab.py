"""One-session Colab supervisor driven by locally verified signed receipts."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from saal_rsi.model_contract import verify  # noqa: E402
from saal_rsi.spec import sha  # noqa: E402

SESSION = "saal-v098-static-reg-run05"
CAMPAIGN_ID = "saal098-static-reg-20261018-05"
STATE = Path.home() / ".local/state/saal-rsi/static-reg-pilot-20261018-run05"
AUTH = json.loads((RUN / "OPERATOR_AUTHORIZATION.json").read_text())
ISSUED_AT = AUTH["issued_at"]
CU_CAP = 12.5  # Keep 1.0 CU of the signed 13.5 CU limit for cleanup.
MAX_RATE = 6.77  # Conservative observed A100 high-RAM rate; tighten if higher.
POLL_SECONDS = 20


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True,
                          timeout=timeout, check=False)


def usage_snapshot() -> tuple[float, float | None, str]:
    try:
        result = run(["colab", "usage"], timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return MAX_RATE, None, "USAGE_UNAVAILABLE"
    if result.returncode:
        return MAX_RATE, None, "USAGE_UNAVAILABLE"
    rate_match = re.search(r"Usage rate:\s*([0-9.]+)/hr", result.stdout)
    balance_match = re.search(r"Current balance:\s*([0-9.]+)\s+compute units", result.stdout)
    rate = float(rate_match.group(1)) if rate_match else MAX_RATE
    balance = float(balance_match.group(1)) if balance_match else None
    return rate, balance, result.stdout.strip().replace("\n", " | ")


def verified_events() -> list[dict]:
    ledger = STATE / "ledger.jsonl"
    if not ledger.is_file():
        return []
    trust = json.loads((RUN / "trust_store.json").read_text())
    witness = trust["attempt_authority"]["public_key"]
    expected_ids = {path.stem.removesuffix("-identity")
                    for path in RUN.glob("saal098r05-arm-*-identity.json")}
    if len(expected_ids) != 5:
        raise ValueError("five fresh run05 identity records required")
    previous = "sha256:" + "0" * 64
    events = []
    for raw in ledger.read_text().splitlines():
        envelope = json.loads(raw)
        body = verify(envelope, witness)
        if (body.get("campaign_id") != CAMPAIGN_ID
                or body.get("sequence") != len(events) + 1
                or body.get("previous") != previous
                or body.get("attempt_id") not in expected_ids):
            raise ValueError("attempt ledger signature chain or campaign binding failed")
        previous = sha(envelope)
        events.append(body)
    return events


def attempt_snapshot(events: list[dict]) -> list[dict]:
    ids = sorted(path.stem.removesuffix("-identity")
                 for path in RUN.glob("saal098r05-arm-*-identity.json"))
    result = []
    for attempt in ids:
        selected = [event for event in events if event["attempt_id"] == attempt]
        kinds = [event["kind"] for event in selected]
        terminal = next((kind for kind in reversed(kinds)
                         if kind in ("COMPLETED", "FAILED")), None)
        result.append({"attempt_id": attempt, "events": kinds,
                       "terminal_status": terminal,
                       "artifact_received": "ARTIFACT_RECEIVED" in kinds})
    return result


def stop_exact_session() -> tuple[subprocess.CompletedProcess, str]:
    try:
        stop = run(["colab", "stop", "--session", SESSION], timeout=90)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return subprocess.CompletedProcess([], 124, "", type(exc).__name__), "STOP_UNCONFIRMED"
    try:
        status = run(["colab", "status", "--session", SESSION], timeout=20)
        status_text = (status.stdout + status.stderr).strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        status_text = type(exc).__name__
    return stop, status_text


def main() -> int:
    if (RUN / "supervisor_result.json").exists():
        raise FileExistsError("run05 supervisor result already exists; refusing repeat")
    if (STATE / "ledger.jsonl").exists() or not STATE.is_dir():
        raise FileExistsError("fresh run05 authority state required")
    if len(list(RUN.glob("saal098r05-arm-*-identity.json"))) != 5:
        raise ValueError("five fresh attempt identity records required before launch")
    launch_started = time.time()
    rate, balance_before, usage_before = usage_snapshot()
    global MAX_RATE
    MAX_RATE = max(MAX_RATE, rate)
    try:
        launch = run(["colab", "exec", "--session", SESSION, "--file",
                      str(RUN / "remote_start.py"), "--timeout", "60"], timeout=90)
        launch_event = {
            "exit_code": launch.returncode,
            "stdout_tail": launch.stdout[-500:],
            "stderr_tail": launch.stderr[-500:],
            "state": "START_REQUEST_RETURNED" if launch.returncode == 0 else "START_REQUEST_FAILED_NO_RETRY",
        }
    except (subprocess.TimeoutExpired, OSError) as exc:
        # A timed-out notebook request may still have launched the detached child.
        # Observe its receipts; never issue a second launch.
        launch_event = {"state": "LAUNCH_UNCERTAIN_NO_RETRY",
                        "failure_type": type(exc).__name__}
    print(json.dumps({"event": "run05_start_observed", "launch": launch_event,
                      "balance_before": balance_before, "rate_upper_bound": MAX_RATE},
                     sort_keys=True), flush=True)

    stop_reason = "ALL_FIVE_TERMINAL_ARTIFACTS_RECEIVED"
    latest: list[dict] = []
    monitor_failure = None
    while True:
        time.sleep(POLL_SECONDS)
        try:
            events = verified_events()
        except Exception as exc:  # noqa: BLE001 - integrity failure must stop exact worker
            monitor_failure = type(exc).__name__
            stop_reason = "LOCAL_RECEIPT_LEDGER_INVALID_FAIL_CLOSED"
            break
        latest = attempt_snapshot(events)
        rate_now, balance_now, usage_now = usage_snapshot()
        MAX_RATE = max(MAX_RATE, rate_now)
        elapsed = max(0.0, time.time() - ISSUED_AT)
        cu_upper = elapsed * MAX_RATE / 3600.0
        terminal_artifacts = all(
            item["terminal_status"] in ("FAILED", "COMPLETED") and item["artifact_received"]
            for item in latest
        )
        print(json.dumps({"event": "run05_monitor", "elapsed_seconds": int(elapsed),
                          "estimated_cu_upper_bound": round(cu_upper, 3),
                          "rate_upper_bound_per_hour": MAX_RATE,
                          "balance": balance_now, "attempts": latest},
                         sort_keys=True), flush=True)
        if terminal_artifacts:
            break
        if cu_upper >= CU_CAP:
            stop_reason = "SIGNED_CU_WATCHDOG_LIMIT"
            break

    # Let the detached runner flush its aggregate log after the last artifact.
    if stop_reason == "ALL_FIVE_TERMINAL_ARTIFACTS_RECEIVED":
        time.sleep(5)
        latest = attempt_snapshot(verified_events())
    stop, status_after = stop_exact_session()
    rate_after, balance_after, usage_after = usage_snapshot()
    result = {
        "schema": "saal-static-screen-supervisor-result-v2",
        "campaign_id": CAMPAIGN_ID,
        "session": SESSION,
        "status": stop_reason,
        "launch": launch_event,
        "monitor_failure_type": monitor_failure,
        "started_at_epoch": int(launch_started),
        "admission_issued_at_epoch": ISSUED_AT,
        "ended_at_epoch": int(time.time()),
        "maximum_observed_rate_cu_per_hour": MAX_RATE,
        "estimated_upper_bound_cu_since_admission": round(
            (time.time() - ISSUED_AT) * MAX_RATE / 3600.0, 3),
        "colab_balance_before": balance_before,
        "colab_balance_after": balance_after,
        "usage_before": usage_before,
        "usage_after_stop": usage_after,
        "attempts": latest,
        "verified_ledger_events": len(verified_events()),
        "remote_kernel_log_downloaded": False,
        "stop_exit_code": stop.returncode,
        "stop_output": stop.stdout.strip(),
        "status_after_stop": status_after,
        "qualification_status": "NOT_EVALUATED",
        "external_deployment_allowed": False,
    }
    (RUN / "supervisor_result.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"event": "run05_supervisor_finished", "result": result},
                     sort_keys=True), flush=True)
    return 0 if stop.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
