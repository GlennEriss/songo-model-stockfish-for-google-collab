#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <git_repo_url> <git_branch> <worktree>"
  exit 1
fi

GIT_REPO_URL="$1"
GIT_BRANCH="$2"
WORKTREE="$3"

echo "Update repo from GitHub"
echo "  repo:    $GIT_REPO_URL"
echo "  branch:  $GIT_BRANCH"
echo "  worktree:$WORKTREE"

if [[ ! -d "$WORKTREE/.git" ]]; then
  rm -rf "$WORKTREE"
  git clone --branch "$GIT_BRANCH" "$GIT_REPO_URL" "$WORKTREE"
else
  git -C "$WORKTREE" fetch origin "$GIT_BRANCH"
  git -C "$WORKTREE" checkout "$GIT_BRANCH"
  git -C "$WORKTREE" pull --ff-only origin "$GIT_BRANCH"
fi

echo "Repo ready."
