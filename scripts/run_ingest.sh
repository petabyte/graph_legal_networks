#!/usr/bin/env bash
# Wrapper that auto-skips segfault cases and restarts the ingest.
# Usage: bash scripts/run_ingest.sh
# Stop it: kill the script's PID (it will cleanly stop the child process)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SENTINEL="$PROJECT_ROOT/data/ingest_last_case.txt"
SKIP_FILE="$PROJECT_ROOT/data/ingest_skip.txt"
LOG="$PROJECT_ROOT/logs/graphiti_parallel.log"
MAX_SKIPS=50  # safety limit — stop if we skip too many cases

cd "$PROJECT_ROOT"

skips_this_session=0

echo "[run_ingest] Keeping qwen3.5 loaded..."
curl -s http://localhost:11434/api/generate \
    -d '{"model": "qwen3.5:latest", "keep_alive": -1}' > /dev/null

while true; do
    # Clear sentinel before each attempt so a leftover from a previous crash
    # doesn't mask a clean exit or a new crash.
    rm -f "$SENTINEL"

    echo "[run_ingest] Starting ingest (attempt $((skips_this_session + 1)))..."
    "$HOME/.pyenv/versions/3.12.0/bin/python3.12" scripts/graphiti_parallel.py >> "$LOG" 2>&1 || true

    # Check if sentinel was written during this run (segfault)
    if [[ ! -f "$SENTINEL" ]]; then
        echo "[run_ingest] Clean exit — ingest complete."
        break
    fi

    # Parse the offending case ID
    case_id=$(grep -oP 'case_id=\K\S+' "$SENTINEL" 2>/dev/null || true)
    if [[ -z "$case_id" ]]; then
        echo "[run_ingest] Sentinel found but no case_id — stopping."
        break
    fi

    # Check if already in skip list (crashed before processing any new case)
    if grep -qx "$case_id" "$SKIP_FILE" 2>/dev/null; then
        echo "[run_ingest] Case $case_id already skipped — crash happened before processing any case. Retrying..."
        skips_this_session=$((skips_this_session + 1))
        if [[ $skips_this_session -ge $MAX_SKIPS ]]; then
            echo "[run_ingest] Reached $MAX_SKIPS retries — stopping."
            break
        fi
        sleep 2
        continue
    fi

    skips_this_session=$((skips_this_session + 1))
    if [[ $skips_this_session -ge $MAX_SKIPS ]]; then
        echo "[run_ingest] Reached $MAX_SKIPS skips this session — stopping to avoid infinite loop."
        break
    fi

    echo "[run_ingest] Segfault on case $case_id — adding to skip list and restarting ($skips_this_session/$MAX_SKIPS)..."
    echo "$case_id" >> "$SKIP_FILE"
    sleep 2
done

echo "[run_ingest] Done. Total skips this session: $skips_this_session"
echo "[run_ingest] Skip list: $(cat "$SKIP_FILE" 2>/dev/null | tr '\n' ' ')"
