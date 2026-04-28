from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _sanitize_identity(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("@", "_at_")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("._-")
    return text[:120] if text else ""


def _detect_drive_identity_key(colab_identity: str) -> str:
    local_override = _sanitize_identity(colab_identity)
    if local_override:
        return local_override

    env_override = _sanitize_identity(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", ""))
    if env_override:
        return env_override

    try:
        import google.auth
        from googleapiclient.discovery import build

        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive.metadata.readonly"]
        )
        svc = build("drive", "v3", credentials=creds, cache_discovery=False)
        about = svc.about().get(fields="user(emailAddress,permissionId)").execute()
        user = dict(about.get("user", {}) or {})
        email = _sanitize_identity(user.get("emailAddress", ""))
        if email:
            return email
        permission_id = _sanitize_identity(user.get("permissionId", ""))
        if permission_id:
            return f"perm_{permission_id}"
    except Exception:
        pass

    return "unknown_drive_identity"


def _clone_or_update_repo(*, repo_url: str, branch: str, worktree: Path) -> None:
    if not (worktree / ".git").exists():
        if worktree.exists():
            shutil.rmtree(worktree)
        subprocess.run(
            ["git", "clone", "--branch", branch, repo_url, str(worktree)],
            check=True,
        )
        return

    subprocess.run(["git", "-C", str(worktree), "fetch", "origin", branch], check=True)
    subprocess.run(["git", "-C", str(worktree), "checkout", branch], check=True)
    subprocess.run(["git", "-C", str(worktree), "pull", "--ff-only", "origin", branch], check=True)


def _install_requirements(*, worktree: Path, python_bin: str) -> None:
    subprocess.run([python_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([python_bin, "-m", "pip", "install", "-r", str(worktree / "requirements.txt")], check=True)


def bootstrap_workspace(
    *,
    git_repo_url: str,
    git_branch: str,
    worktree: Path,
    drive_project_name: str,
    colab_identity: str,
    mydrive_root: Path,
    python_bin: str,
    install_requirements: bool,
) -> dict[str, str]:
    if not mydrive_root.exists():
        raise RuntimeError(
            "MyDrive non visible. Exécute d'abord drive.mount('/content/drive')."
        )

    drive_root = mydrive_root / drive_project_name
    drive_root.mkdir(parents=True, exist_ok=True)

    drive_identity_key = _detect_drive_identity_key(colab_identity)
    drive_workspace_root = drive_root / drive_identity_key
    drive_workspace_root.mkdir(parents=True, exist_ok=True)

    os.environ["SONGO_DRIVE_ROOT"] = str(drive_root)
    os.environ["SONGO_ENFORCE_DRIVE_ROOT_WRITES"] = "1"
    os.environ["SONGO_DRIVE_IDENTITY_KEY"] = drive_identity_key
    os.environ["SONGO_DRIVE_WORKSPACE_ROOT"] = str(drive_workspace_root)

    for rel in [
        "secrets",
        "models",
        f"{drive_identity_key}/jobs",
        f"{drive_identity_key}/logs",
        f"{drive_identity_key}/reports",
        f"{drive_identity_key}/data",
        f"{drive_identity_key}/data/datasets",
        f"{drive_identity_key}/data/label_cache",
        f"{drive_identity_key}/runtime_backup/jobs",
    ]:
        (drive_root / rel).mkdir(parents=True, exist_ok=True)

    _clone_or_update_repo(repo_url=git_repo_url, branch=git_branch, worktree=worktree)
    if install_requirements:
        _install_requirements(worktree=worktree, python_bin=python_bin)

    summary = {
        "git_repo_url": git_repo_url,
        "git_branch": git_branch,
        "worktree": str(worktree),
        "drive_root": str(drive_root),
        "colab_identity": str(colab_identity or ""),
        "drive_identity_key": drive_identity_key,
        "drive_workspace_root": str(drive_workspace_root),
        "python_bin": str(python_bin),
        "install_requirements": str(bool(install_requirements)).lower(),
    }

    print("DRIVE_ROOT           =", summary["drive_root"])
    print("COLAB_IDENTITY       =", summary["colab_identity"] or "<auto>")
    print("DRIVE_IDENTITY_KEY   =", summary["drive_identity_key"])
    print("DRIVE_WORKSPACE_ROOT =", summary["drive_workspace_root"])
    print("WORKTREE             =", summary["worktree"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--git-repo-url",
        default="https://github.com/GlennEriss/songo-model-stockfish-for-google-collab.git",
    )
    parser.add_argument("--git-branch", default="main")
    parser.add_argument(
        "--worktree",
        default="/content/songo-model-stockfish-for-google-collab",
    )
    parser.add_argument("--drive-project-name", default="songo-stockfish")
    parser.add_argument("--colab-identity", default="")
    parser.add_argument("--mydrive-root", default="/content/drive/MyDrive")
    parser.add_argument("--python-bin", default=(sys.executable or "python3"))
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    summary = bootstrap_workspace(
        git_repo_url=str(args.git_repo_url),
        git_branch=str(args.git_branch),
        worktree=Path(str(args.worktree)),
        drive_project_name=str(args.drive_project_name),
        colab_identity=str(args.colab_identity),
        mydrive_root=Path(str(args.mydrive_root)),
        python_bin=str(args.python_bin),
        install_requirements=(not bool(args.skip_install)),
    )

    summary_path = str(args.summary_path or "").strip()
    if summary_path:
        Path(summary_path).write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    if bool(args.print_json):
        print(json.dumps(summary, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
