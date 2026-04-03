#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <git_repo_url> <git_branch> <worktree> <drive_project_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_REPO_URL="$1"
GIT_BRANCH="$2"
WORKTREE="$3"
DRIVE_DIR="$4"

"$SCRIPT_DIR/init_drive_layout.sh" "$DRIVE_DIR"
"$SCRIPT_DIR/update_repo_from_github.sh" "$GIT_REPO_URL" "$GIT_BRANCH" "$WORKTREE"

python3 -m venv "$WORKTREE/.venv"
source "$WORKTREE/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$WORKTREE/requirements.txt"

echo "Bootstrap complete."
echo "Activate with:"
echo "  source $WORKTREE/.venv/bin/activate"
