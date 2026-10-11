#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m compileall -q src tests
python -m pytest -q
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python -m dream_rsi_governed.cli demo --worlds 40 --seed 7 --beta 1.0 --max-score-regression 0.05 --promote --out "$TMP/run" >/dev/null
for f in qualification.json canary.json promotion.json ledger.sqlite3 promotion-key.ed25519; do
  test -s "$TMP/run/$f"
done
python - <<'PY'
import tempfile
from pathlib import Path
from dream_rsi_governed.ledger import AppendOnlyLedger
p = Path(tempfile.mkdtemp()) / "ledger.sqlite3"
l = AppendOnlyLedger(p)
l.append("validation", {"ok": True})
assert l.verify()
PY
echo "validation: PASS"
