"""Start the already signed runner as a tracked child in the named Colab VM."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

base = Path("/content/saal-static-screen")
pid_path = Path("/content/saal-static-screen.pid")
log_path = Path("/content/saal-static-screen-bootstrap.log")
runner_path = Path("/content/saal-static-screen-runner.py")
if pid_path.exists() or log_path.exists() or base.exists():
    raise FileExistsError("campaign runtime path already exists; refusing a duplicate start")
if not runner_path.is_file() or not Path("/content/saal-static-inputs.zip").is_file():
    raise FileNotFoundError("verified campaign inputs or runner are missing")
with log_path.open("xb") as log:
    process = subprocess.Popen(
        [sys.executable, str(runner_path)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
pid_path.write_text(str(process.pid), encoding="ascii")
time.sleep(0.5)
state = "RUNNING" if process.poll() is None else "EXITED_DURING_START"
print(json.dumps({"worker_pid": process.pid, "state": state,
                  "log_path": str(log_path)}, sort_keys=True))
if state != "RUNNING":
    raise RuntimeError("runner exited before the monitor attached; inspect its log, do not retry")
