from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    repo_root: Path
    drive_root: Path
    jobs_root: Path
    logs_root: Path
    reports_root: Path
    models_root: Path
    data_root: Path


def _resolve_root(*, base_root: Path, configured: object, default_relative: str) -> Path:
    text = str(configured or "").strip()
    if not text:
        return base_root / default_relative
    path = Path(text)
    if path.is_absolute():
        return path
    return base_root / path


def build_project_paths(config: dict) -> ProjectPaths:
    storage = config.get("storage", {})
    default_repo_root = Path(__file__).resolve().parents[3]
    repo_root = Path(storage.get("repo_root") or default_repo_root)
    if not repo_root.exists():
        repo_root = default_repo_root

    configured_drive_root = storage.get("drive_root")
    drive_root = Path(str(configured_drive_root or "").strip())
    if not str(drive_root):
        raise RuntimeError(
            "Configuration invalide: `storage.drive_root` est requis. "
            "Exemple attendu: /content/drive/MyDrive/songo-stockfish"
        )
    if not drive_root.exists():
        raise RuntimeError(
            "Drive root introuvable. Monte Google Drive puis relance. "
            f"Chemin configure: {drive_root}"
        )
    jobs_root = _resolve_root(base_root=drive_root, configured=storage.get("jobs_root"), default_relative="jobs")
    logs_root = _resolve_root(base_root=drive_root, configured=storage.get("logs_root"), default_relative="logs")
    reports_root = _resolve_root(base_root=drive_root, configured=storage.get("reports_root"), default_relative="reports")
    models_root = _resolve_root(base_root=drive_root, configured=storage.get("models_root"), default_relative="models")
    data_root = _resolve_root(base_root=drive_root, configured=storage.get("data_root"), default_relative="data")

    return ProjectPaths(
        repo_root=repo_root,
        drive_root=drive_root,
        jobs_root=jobs_root,
        logs_root=logs_root,
        reports_root=reports_root,
        models_root=models_root,
        data_root=data_root,
    )
