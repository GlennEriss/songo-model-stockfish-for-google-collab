#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <drive_project_dir>"
  exit 1
fi

DRIVE_DIR="$1"

echo "Initialize Drive layout"
echo "  root: $DRIVE_DIR"

mkdir -p \
  "$DRIVE_DIR/code" \
  "$DRIVE_DIR/code_snapshots" \
  "$DRIVE_DIR/data/raw" \
  "$DRIVE_DIR/data/raw_colab_pro" \
  "$DRIVE_DIR/data/raw_full_matrix_colab_pro" \
  "$DRIVE_DIR/data/sampled" \
  "$DRIVE_DIR/data/sampled_colab_pro" \
  "$DRIVE_DIR/data/sampled_full_matrix_colab_pro" \
  "$DRIVE_DIR/data/datasets" \
  "$DRIVE_DIR/jobs" \
  "$DRIVE_DIR/models/checkpoints" \
  "$DRIVE_DIR/models/final" \
  "$DRIVE_DIR/logs" \
  "$DRIVE_DIR/reports/evaluations" \
  "$DRIVE_DIR/reports/benchmarks" \
  "$DRIVE_DIR/exports"

cat > "$DRIVE_DIR/README_LAYOUT.txt" <<'EOF'
songo-model-stockfish-for-google-collab - Drive layout

Recommended usage:
- code/                         -> optional synced copy of repository files
- code_snapshots/               -> timestamped code snapshots for audit/rollback
- data/raw*/                    -> raw match logs
- data/sampled*/                -> sampled positions before labeling
- data/datasets/                -> final train/validation/test datasets
- jobs/                         -> resumable job manifests and progress state
- models/checkpoints/           -> last/best checkpoints
- models/final/                 -> final exported models and model cards
- logs/                         -> long-lived textual logs if needed
- reports/evaluations/          -> evaluation JSON reports
- reports/benchmarks/           -> benchmark JSON reports
- exports/                      -> optional exported artifacts
EOF

echo "Drive layout initialized."
