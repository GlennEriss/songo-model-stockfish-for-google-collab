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
    return ProjectPaths(
        repo_root=repo_root,
        drive_root=drive_root,
        jobs_root=drive_root / "jobs",
        logs_root=drive_root / "logs",
        reports_root=drive_root / "reports",
        models_root=drive_root / "models",
        data_root=drive_root / "data",
    )
