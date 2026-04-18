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
            "--allow-model-purge",
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
            "--retention",
            "--retention-job-stream-ttl-hours",
            "96",
            "--purge-quarantine",
            "--purge-duplicate-source-metadata",
            "--purge-global-progress-mirrors",
            "--purge-pipeline-manifests",
            "--purge-completed-job-dirs",
            "--retention-global-progress-ttl-days",
            "21",
            "--retention-global-progress-keep-recent",
            "4",
            "--retention-pipeline-manifest-ttl-days",
            "10",
            "--retention-pipeline-manifest-keep-recent",
            "12",
            "--retention-completed-job-dir-ttl-days",
            "30",
            "--retention-completed-job-dir-keep-recent-per-run-type",
            "9",
            "--retention-source-metadata-raw-ttl-hours",
            "36",
        ]
    )

    assert int(exit_code) == 0
    assert bool(captured_kwargs.get("apply")) is True
    assert bool(captured_kwargs.get("cleanup_runtime_migration")) is True
    assert bool(captured_kwargs.get("cleanup_models")) is True
    assert bool(captured_kwargs.get("cleanup_retention")) is True
    assert bool(captured_kwargs.get("cleanup_quarantine_dirs")) is True
    assert bool(captured_kwargs.get("cleanup_duplicate_source_metadata")) is True
    assert bool(captured_kwargs.get("cleanup_global_progress_mirrors")) is True
    assert bool(captured_kwargs.get("cleanup_pipeline_manifests")) is True
    assert bool(captured_kwargs.get("cleanup_completed_job_dirs")) is True
    assert bool(captured_kwargs.get("drive_raw_cleanup_include_inactive_partial")) is True
    assert float(captured_kwargs.get("drive_raw_cleanup_inactive_min_age_seconds", 0.0)) == 48.0 * 3600.0
    assert float(captured_kwargs.get("retention_job_stream_ttl_seconds", 0.0)) == 96.0 * 3600.0
    assert float(captured_kwargs.get("retention_global_progress_ttl_seconds", 0.0)) == 21.0 * 86400.0
    assert int(captured_kwargs.get("retention_global_progress_keep_recent", 0)) == 4
    assert float(captured_kwargs.get("retention_pipeline_manifest_ttl_seconds", 0.0)) == 10.0 * 86400.0
    assert int(captured_kwargs.get("retention_pipeline_manifest_keep_recent", 0)) == 12
    assert float(captured_kwargs.get("retention_job_dir_ttl_seconds", 0.0)) == 30.0 * 86400.0
    assert int(captured_kwargs.get("retention_job_dir_keep_recent_per_run_type", 0)) == 9
    assert float(captured_kwargs.get("retention_source_metadata_raw_ttl_seconds", 0.0)) == 36.0 * 3600.0
    assert list(captured_kwargs.get("keep_model_ids", [])) == ["model_a", "model_b", "model_c", "model_d"]

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert int(payload["step_count"]) == 4


def test_storage_cleanup_cli_blocks_model_cleanup_without_allow_flag(monkeypatch, capsys, tmp_path: Path) -> None:
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
        return {"ok": True}

    import songo_model_stockfish.ops.storage_cleanup as storage_cleanup_module

    monkeypatch.setattr(storage_cleanup_module, "run_storage_cleanup", _fake_run_storage_cleanup)

    exit_code = cli_main.main(
        [
            "storage-cleanup",
            "--config",
            "config/train.full_matrix.colab_pro.scratch.yaml",
            "--purge-models",
        ]
    )

    assert int(exit_code) == 0
    assert bool(captured_kwargs.get("cleanup_models")) is False
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    warning_payload = json.loads(out_lines[0])
    assert "model_cleanup_blocked_by_safety" in str(warning_payload.get("warning", ""))
