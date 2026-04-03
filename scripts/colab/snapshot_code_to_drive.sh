#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <repo_dir> <drive_project_dir>"
  exit 1
fi

REPO_DIR="$1"
DRIVE_DIR="$2"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST_DIR="$DRIVE_DIR/code_snapshots/$STAMP"

mkdir -p "$DEST_DIR"

echo "Snapshot code to Drive"
echo "  from: $REPO_DIR"
echo "  to:   $DEST_DIR"

rsync -av \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'outputs/' \
  "$REPO_DIR/" \
  "$DEST_DIR/"

echo "Snapshot complete."
