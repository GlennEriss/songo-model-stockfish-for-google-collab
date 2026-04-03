#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <drive_project_dir> <job_id>"
  exit 1
fi

DRIVE_DIR="$1"
JOB_ID="$2"
JOB_DIR="$DRIVE_DIR/jobs/$JOB_ID"

echo "Watching job: $JOB_ID"
echo "Job dir: $JOB_DIR"

while true; do
  clear || true
  echo "=== run_status.json ==="
  if [[ -f "$JOB_DIR/run_status.json" ]]; then
    cat "$JOB_DIR/run_status.json"
  else
    echo "missing"
  fi
  echo
  echo "=== state.json ==="
  if [[ -f "$JOB_DIR/state.json" ]]; then
    cat "$JOB_DIR/state.json"
  else
    echo "missing"
  fi
  echo
  echo "=== recent events ==="
  if [[ -f "$JOB_DIR/events.jsonl" ]]; then
    tail -n 20 "$JOB_DIR/events.jsonl"
  else
    echo "missing"
  fi
  sleep 10
done
