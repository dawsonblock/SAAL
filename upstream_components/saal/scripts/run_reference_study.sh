#!/usr/bin/env bash
set -euo pipefail
python -m saal train --out artifacts/intent-model.json
python -m saal evaluate --model artifacts/intent-model.json --out artifacts/evaluation-unsigned.json
printf '\nThis is an UNSIGNED local illustration, NOT independent qualification.\n'
