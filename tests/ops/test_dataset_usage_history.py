from __future__ import annotations

import json
from pathlib import Path

from songo_model_stockfish.ops.dataset_usage_history import (
    build_dataset_usage_report,
    record_training_dataset_usage,
)
from songo_model_stockfish.ops.paths import build_project_paths


def _make_config(drive_root: Path) -> dict:
    return {
        "storage": {
            "drive_root": str(drive_root),
            "repo_root": str(drive_root / "repo"),
            "jobs_root": str(drive_root / "jobs"),
            "logs_root": str(drive_root / "logs"),
            "reports_root": str(drive_root / "reports"),
            "models_root": str(drive_root / "models"),
            "data_root": str(drive_root / "data"),
        },
        "train": {"dataset_id": "dataset_final"},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_dataset_usage_history_tracks_most_used_and_purge_candidates(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    config = _make_config(drive_root)
    paths = build_project_paths(config)

    _write_json(
        paths.data_root / "dataset_registry.json",
        {
            "dataset_sources": [],
            "built_datasets": [
                {"dataset_id": "dataset_final", "labeled_samples": 20_000_000, "build_mode": "merged_final"},
                {"dataset_id": "dataset_old", "labeled_samples": 800_000, "build_mode": "teacher_label"},
                {"dataset_id": "dataset_exp", "labeled_samples": 300_000, "build_mode": "teacher_label"},
            ],
        },
    )

    record_training_dataset_usage(
        paths=paths,
        job_id="train_job_001",
        dataset_id="dataset_final",
        dataset_selection_mode="configured",
        model_id="model_v1",
        completed_at="2026-04-17T10:00:00Z",
        completed_epochs=20,
        requested_epochs=40,
        best_validation_metric=0.90,
    )
    record_training_dataset_usage(
        paths=paths,
        job_id="train_job_002",
        dataset_id="dataset_final",
        dataset_selection_mode="configured",
        model_id="model_v2",
        completed_at="2026-04-18T10:00:00Z",
        completed_epochs=25,
        requested_epochs=40,
        best_validation_metric=0.91,
    )
    record_training_dataset_usage(
        paths=paths,
        job_id="train_job_003",
        dataset_id="dataset_old",
        dataset_selection_mode="configured",
        model_id="model_v3",
        completed_at="2026-04-10T10:00:00Z",
        completed_epochs=18,
        requested_epochs=40,
        best_validation_metric=0.84,
    )

    report = build_dataset_usage_report(
        paths=paths,
        config=config,
        include_job_scan=False,
        sync_history_from_jobs=False,
        keep_dataset_ids=[],
        purge_min_age_seconds=0.0,
        max_purge_candidates=10,
        top_n=10,
    )

    assert str(report.get("most_used_dataset_id", "")) == "dataset_final"
    assert str(report.get("largest_built_dataset_id", "")) == "dataset_final"
    assert str(report.get("recommended_primary_dataset_id", "")) == "dataset_final"

    top_rows = report.get("top_datasets", [])
    assert isinstance(top_rows, list)
    assert str(top_rows[0].get("dataset_id", "")) == "dataset_final"
    assert int(top_rows[0].get("completed_runs", 0)) == 2

    purge_rows = report.get("purge_candidates", [])
    purge_ids = {str(item.get("dataset_id", "")) for item in purge_rows if isinstance(item, dict)}
    assert "dataset_old" in purge_ids
    assert "dataset_final" not in purge_ids


def test_dataset_usage_history_can_sync_from_existing_train_jobs(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    config = _make_config(drive_root)
    paths = build_project_paths(config)

    job_a = paths.jobs_root / "train_colab_001"
    job_a.mkdir(parents=True, exist_ok=True)
    _write_json(
        job_a / "training_summary.json",
        {
            "job_id": "train_colab_001",
            "dataset_id": "dataset_scan_a",
            "dataset_selection_mode": "configured",
            "model_id": "model_scan_a",
            "completed_epochs": 10,
            "epochs": 40,
            "best_validation_metric": 0.85,
        },
    )
    _write_json(
        job_a / "run_status.json",
        {
            "run_type": "train",
            "status": "completed",
            "updated_at": "2026-04-18T11:00:00Z",
        },
    )

    job_b = paths.jobs_root / "train_colab_002"
    job_b.mkdir(parents=True, exist_ok=True)
    _write_json(
        job_b / "training_summary.json",
        {
            "job_id": "train_colab_002",
            "dataset_id": "dataset_scan_b",
            "dataset_selection_mode": "configured",
            "model_id": "model_scan_b",
            "completed_epochs": 12,
            "epochs": 40,
            "best_validation_metric": 0.87,
        },
    )
    _write_json(
        job_b / "run_status.json",
        {
            "run_type": "train",
            "status": "completed",
            "updated_at": "2026-04-18T12:00:00Z",
        },
    )

    report = build_dataset_usage_report(
        paths=paths,
        config=config,
        include_job_scan=True,
        sync_history_from_jobs=True,
        keep_dataset_ids=[],
        purge_min_age_seconds=0.0,
        max_purge_candidates=10,
        top_n=10,
    )

    assert int(report.get("history_entries_from_scan", 0)) >= 2
    assert bool(report.get("history_synced_from_jobs", False)) is True

    history_path = paths.data_root / "dataset_training_usage_history.json"
    assert history_path.exists()
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    job_ids = {str(item.get("job_id", "")) for item in entries if isinstance(item, dict)}
    assert "train_colab_001" in job_ids
    assert "train_colab_002" in job_ids
