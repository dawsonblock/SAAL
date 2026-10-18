"""Read-only progress snapshot for the exact named Colab campaign process."""
from __future__ import annotations

import json
from pathlib import Path

base = Path("/content/saal-static-screen")
pid_path = Path("/content/saal-static-screen.pid")
log_path = Path("/content/saal-static-screen-bootstrap.log")
pid = int(pid_path.read_text(encoding="ascii")) if pid_path.is_file() else None
state = "NOT_STARTED"
if pid is not None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
        state = "EXITED" if stat[2] == "Z" else "RUNNING"
    except FileNotFoundError:
        state = "EXITED"
log_tail = []
if log_path.is_file():
    log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
attempts = []
results = base / "results"
if results.is_dir():
    for attempt_dir in sorted(results.iterdir()):
        if not attempt_dir.is_dir():
            continue
        entry = {"attempt_id": attempt_dir.name}
        for name in ("preflight_failure.json", "launch_uncertain.json", "failure.json",
                     "result.json", "terminal_receipt.json"):
            path = attempt_dir / name
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    entry[name] = data if name != "result.json" else {
                        "selected_checkpoint_status": data.get("selected_checkpoint_status"),
                        "selected_checkpoint_step": data.get("selected_checkpoint_step"),
                        "completed_optimizer_updates": data.get("completed_optimizer_updates"),
                        "qualification_status": data.get("qualification_status"),
                    }
                except json.JSONDecodeError:
                    entry[name] = "UNREADABLE"
        diagnostics = attempt_dir / "training" / "training_diagnostics.jsonl"
        if diagnostics.is_file():
            lines = diagnostics.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                try:
                    latest = json.loads(lines[-1])
                    entry["latest_step"] = latest.get("completed_optimizer_step")
                    entry["latest_ce"] = latest.get("cross_entropy")
                    entry["latest_orthogonal_penalty"] = latest.get("orthogonal_penalty")
                    entry["latest_elapsed_seconds"] = latest.get("elapsed_seconds")
                except json.JSONDecodeError:
                    entry["latest_step"] = "UNREADABLE"
        attempts.append(entry)
print(json.dumps({"state": state, "worker_pid": pid, "attempts": attempts,
                  "log_tail": log_tail}, sort_keys=True))
