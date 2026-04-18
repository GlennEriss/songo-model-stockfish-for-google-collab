from __future__ import annotations

import json
from pathlib import Path

from songo_model_stockfish.cli import main as cli_main
from songo_model_stockfish.ops.paths import ProjectPaths


def test_dataset_usage_cli_passes_flags_and_prints_json(monkeypatch, capsys, tmp_path: Path) -> None:
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

    captured_kwargs: dict[str, object] = {}

    def _fake_build_dataset_usage_report(**kwargs):  # noqa: ANN003
        captured_kwargs.update(kwargs)
        return {
            "history_entries_total": 3,
            "history_entries_from_scan": 2,
            "history_synced_from_jobs": True,
            "recommended_primary_dataset_id": "dataset_final",
            "most_used_dataset_id": "dataset_final",
            "largest_built_dataset_id": "dataset_final",
            "keep_dataset_ids": ["dataset_final"],
            "top_datasets": [{"dataset_id": "dataset_final", "completed_runs": 2}],
            "purge_candidates": [{"dataset_id": "dataset_old", "reason": "not_kept_and_stale"}],
        }

    import songo_model_stockfish.ops.dataset_usage_history as usage_module

    monkeypatch.setattr(usage_module, "build_dataset_usage_report", _fake_build_dataset_usage_report)

    exit_code = cli_main.main(
        [
            "dataset-usage",
            "--config",
            "config/train.full_matrix.colab_pro.scratch.yaml",
            "--json",
            "--keep-dataset-id",
            "dataset_manual_keep",
            "--purge-min-age-days",
            "5",
            "--max-purge-candidates",
            "12",
            "--top",
            "7",
            "--no-sync-history-from-jobs",
            "--no-include-job-scan",
        ]
    )

    assert int(exit_code) == 0
    assert bool(captured_kwargs.get("include_job_scan")) is False
    assert bool(captured_kwargs.get("sync_history_from_jobs")) is False
    assert list(captured_kwargs.get("keep_dataset_ids", [])) == ["dataset_manual_keep"]
    assert float(captured_kwargs.get("purge_min_age_seconds", 0.0)) == 5.0 * 86400.0
    assert int(captured_kwargs.get("max_purge_candidates", 0)) == 12
    assert int(captured_kwargs.get("top_n", 0)) == 7

    payload = json.loads(capsys.readouterr().out)
    assert int(payload.get("history_entries_total", 0)) == 3
    purge_candidates = payload.get("purge_candidates", [])
    assert isinstance(purge_candidates, list)
    assert str(purge_candidates[0].get("dataset_id", "")) == "dataset_old"
