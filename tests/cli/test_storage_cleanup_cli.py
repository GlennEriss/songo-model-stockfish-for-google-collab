from __future__ import annotations

import json
from pathlib import Path

from songo_model_stockfish.cli import main as cli_main
from songo_model_stockfish.ops.paths import ProjectPaths


def test_storage_cleanup_cli_smoke_all_flags(monkeypatch, capsys, tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    fake_cfg = {"storage": {"drive_root": str(drive_root)}}

    fake_paths = ProjectPaths(
        repo_root=tmp_path / "repo",
        drive_root=drive_root,
        jobs_root=drive_root / "jobs",
        jobs_backup_root=drive_root / "runtime_backup" / "jobs",
        logs_root=drive_root / "logs",
        reports_root=drive_root / "reports",
        models_root=drive_root / "models",
        data_root=drive_root / "data",
    )

    monkeypatch.setattr(cli_main, "load_yaml_config", lambda _path: fake_cfg)
    monkeypatch.setattr(cli_main, "build_project_paths", lambda _cfg: fake_paths)

    captured_kwargs: dict = {}

    def _fake_run_storage_cleanup(**kwargs):  # noqa: ANN003
        captured_kwargs.update(kwargs)
        return {"ok": True, "step_count": 4}

    import songo_model_stockfish.ops.storage_cleanup as storage_cleanup_module

    monkeypatch.setattr(storage_cleanup_module, "run_storage_cleanup", _fake_run_storage_cleanup)

    exit_code = cli_main.main(
        [
            "storage-cleanup",
            "--config",
            "config/train.full_matrix.colab_pro.scratch.yaml",
            "--all",
            "--apply",
            "--keep-model-id",
            "model_a",
            "--keep-model-id",
            "model_b",
            "--keep-model-ids",
            "model_c,model_d",
            "--keep-top-models",
            "2",
            "--keep-dataset-id",
            "dataset_x",
            "--purge-drive-raw-include-inactive-partial",
            "--purge-drive-raw-inactive-min-age-hours",
            "48",
        ]
    )

    assert int(exit_code) == 0
    assert bool(captured_kwargs.get("apply")) is True
    assert bool(captured_kwargs.get("cleanup_runtime_migration")) is True
    assert bool(captured_kwargs.get("cleanup_models")) is True
    assert bool(captured_kwargs.get("drive_raw_cleanup_include_inactive_partial")) is True
    assert float(captured_kwargs.get("drive_raw_cleanup_inactive_min_age_seconds", 0.0)) == 48.0 * 3600.0
    assert list(captured_kwargs.get("keep_model_ids", [])) == ["model_a", "model_b", "model_c", "model_d"]

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert int(payload["step_count"]) == 4
