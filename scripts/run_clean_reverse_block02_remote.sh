#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUEUE="data/collection_batches/clean_reverse_b02_200_121.csv"
PYTHON=".venv/bin/python"
STREAMLIT=".venv/bin/streamlit"

if [[ ! -x "$PYTHON" || ! -x "$STREAMLIT" ]]; then
  echo "Missing .venv dependencies. Follow REMOTE_BLOCK02_RUNBOOK.md before launching."
  exit 1
fi

"$PYTHON" - "$QUEUE" <<'PY'
import csv
import sys
from pathlib import Path

queue = Path(sys.argv[1])
with queue.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

actual_ids = [row["clip_id"] for row in rows]
expected_ids = [f"clip_{index:03d}" for index in range(200, 120, -1)]
if actual_ids != expected_ids:
    raise SystemExit(f"Invalid block 02 queue: expected clip_200 through clip_121, got {actual_ids[:1]} ... {actual_ids[-1:]}")
if any(not row.get("preview_url", "").strip() for row in rows):
    raise SystemExit("Invalid block 02 queue: one or more songs are missing preview URLs.")

print(f"Verified block 02 queue: {len(rows)} songs, {actual_ids[0]} through {actual_ids[-1]}")
PY

if lsof -nP -iTCP:8503 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8503 is already in use. Stop the existing collector before launching block 02."
  exit 1
fi

exec env \
  BIOBEAT_CLIPS_CSV="$QUEUE" \
  BIOBEAT_CLIP_ORDER=csv \
  PYTHONPATH=src \
  "$STREAMLIT" run src/dashboard/batch_app.py \
    --server.port 8503 \
    --server.address 127.0.0.1
