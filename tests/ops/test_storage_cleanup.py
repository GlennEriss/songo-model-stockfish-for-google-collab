from __future__ import annotations

import json
import os
import time
from pathlib import Path

from songo_model_stockfish.ops.paths import build_project_paths
from songo_model_stockfish.ops.storage_cleanup import run_storage_cleanup


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
        }
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_model_cleanup_checkpoint_matching_ignores_prefix_collisions(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)

    (paths.models_root / "final").mkdir(parents=True, exist_ok=True)
    (paths.models_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    (paths.models_root / "final" / "songo_policy_value_colab_pro_v1.pt").write_text("x", encoding="utf-8")
    (paths.models_root / "final" / "songo_policy_value_colab_pro_v10.pt").write_text("x", encoding="utf-8")
    # Ce checkpoint ne doit jamais etre retire quand on purge v1.
    protected_checkpoint = paths.models_root / "checkpoints" / "songo_policy_value_colab_pro_v10_epoch_0001.pt"
    protected_checkpoint.write_text("x", encoding="utf-8")
    # Celui-ci peut etre purge pour v1.
    removable_checkpoint = paths.models_root / "checkpoints" / "songo_policy_value_colab_pro_v1_epoch_0001.pt"
    removable_checkpoint.write_text("x", encoding="utf-8")

    report = run_storage_cleanup(
        config=cfg,
        paths=paths,
        apply=False,
        cleanup_runtime_migration=False,
        cleanup_runtime_backup_streams=False,
        cleanup_drive_raw_dirs=False,
        cleanup_drive_label_cache=False,
        cleanup_models=True,
        keep_model_ids=["songo_policy_value_colab_pro_v10"],
        keep_top_models=0,
        keep_dataset_ids=[],
    )
    removed_models = report["steps"]["model_cleanup"]["removed_models"]
    removed_v1 = next(item for item in removed_models if item.get("model_id") == "songo_policy_value_colab_pro_v1")
    removed_paths = [str(item) for item in removed_v1.get("removed_paths", [])]

    assert any(str(removable_checkpoint) == item for item in removed_paths)
    assert all(str(protected_checkpoint) != item for item in removed_paths)


def test_drive_raw_cleanup_keeps_partial_sources_by_default(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)

    completed_raw = paths.drive_root / "data" / "raw_completed"
    completed_sampled = paths.drive_root / "data" / "sampled_completed"
    partial_raw = paths.drive_root / "data" / "raw_partial"
    partial_sampled = paths.drive_root / "data" / "sampled_partial"
    completed_raw.mkdir(parents=True, exist_ok=True)
    completed_sampled.mkdir(parents=True, exist_ok=True)
    partial_raw.mkdir(parents=True, exist_ok=True)
    partial_sampled.mkdir(parents=True, exist_ok=True)

    _write_json(
        paths.data_root / "dataset_registry.json",
        {
            "dataset_sources": [
                {
                    "dataset_source_id": "source_completed",
                    "source_status": "completed",
                    "raw_dir": str(completed_raw),
                    "sampled_dir": str(completed_sampled),
                    "updated_at": "2026-04-16T10:00:00Z",
                },
                {
                    "dataset_source_id": "source_partial",
                    "source_status": "partial",
                    "raw_dir": str(partial_raw),
                    "sampled_dir": str(partial_sampled),
                    "updated_at": "2026-04-16T10:00:00Z",
                },
            ],
            "built_datasets": [],
        },
    )

    report = run_storage_cleanup(
        config=cfg,
        paths=paths,
        apply=False,
        cleanup_runtime_migration=False,
        cleanup_runtime_backup_streams=False,
        cleanup_drive_raw_dirs=True,
        cleanup_drive_label_cache=False,
        cleanup_models=False,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
    )

    raw_step = report["steps"]["drive_raw_cleanup"]
    removed = set(str(item) for item in raw_step.get("raw_dirs_removed", []))
    skipped_reasons = raw_step.get("raw_dirs_skipped_reason", [])

    assert str(completed_raw) in removed
    assert str(partial_raw) not in removed
    assert any(item.get("raw_dir") == str(partial_raw) and item.get("reason") == "status_not_completed" for item in skipped_reasons)


def test_runtime_backup_stream_cleanup_respects_ttl(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)
    backup_root = paths.jobs_backup_root or (paths.drive_root / "runtime_backup" / "jobs")
    backup_root.mkdir(parents=True, exist_ok=True)

    old_job = backup_root / "job_old"
    old_job.mkdir(parents=True, exist_ok=True)
    _write_json(old_job / "run_status.json", {"status": "completed", "updated_at": "2026-01-01T00:00:00Z"})
    (old_job / "events.jsonl").write_text("x", encoding="utf-8")
    (old_job / "metrics.jsonl").write_text("x", encoding="utf-8")

    recent_job = backup_root / "job_recent"
    recent_job.mkdir(parents=True, exist_ok=True)
    recent_updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))
    _write_json(recent_job / "run_status.json", {"status": "completed", "updated_at": recent_updated_at})
    (recent_job / "events.jsonl").write_text("x", encoding="utf-8")
    (recent_job / "metrics.jsonl").write_text("x", encoding="utf-8")

    report = run_storage_cleanup(
        config=cfg,
        paths=paths,
        apply=False,
        cleanup_runtime_migration=False,
        cleanup_runtime_backup_streams=True,
        cleanup_drive_raw_dirs=False,
        cleanup_drive_label_cache=False,
        cleanup_models=False,
        cleanup_retention=False,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
        retention_job_stream_ttl_seconds=24.0 * 3600.0,
    )

    stream_step = report["steps"]["runtime_backup_stream_cleanup"]
    assert int(stream_step.get("events_removed", 0)) == 1
    assert int(stream_step.get("metrics_removed", 0)) == 1
    assert int(stream_step.get("jobs_skipped_recent", 0)) == 1


def test_retention_cleanup_removes_old_quarantine_dirs(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)

    quarantine_dir = paths.drive_root / "jobs" / ".quarantine_train_123"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    marker = quarantine_dir / "marker.txt"
    marker.write_text("x", encoding="utf-8")

    old_epoch = time.time() - 7.0 * 86400.0
    quarantine_dir.chmod(0o755)
    marker.chmod(0o644)
    os.utime(quarantine_dir, (old_epoch, old_epoch))
    os.utime(marker, (old_epoch, old_epoch))

    report = run_storage_cleanup(
        config=cfg,
        paths=paths,
        apply=False,
        cleanup_runtime_migration=False,
        cleanup_runtime_backup_streams=False,
        cleanup_drive_raw_dirs=False,
        cleanup_drive_label_cache=False,
        cleanup_models=False,
        cleanup_retention=True,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
        retention_quarantine_ttl_seconds=72.0 * 3600.0,
    )

    retention_step = report["steps"]["retention_cleanup"]
    removed = set(str(item) for item in retention_step.get("quarantine_removed", []))
    assert str(quarantine_dir) in removed


def test_external_artifacts_cleanup_moves_known_items_outside_drive_root(tmp_path: Path) -> None:
    mydrive_root = tmp_path / "MyDrive"
    drive_root = mydrive_root / "songo-stockfish"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)

    external_quarantine = mydrive_root / ".quarantine_benchmark_x"
    external_quarantine.mkdir(parents=True, exist_ok=True)
    external_progress = mydrive_root / "bench_models_20m_global.json"
    external_progress.write_text("{}", encoding="utf-8")

    report = run_storage_cleanup(
        config=cfg,
        paths=paths,
        apply=False,
        cleanup_runtime_migration=False,
        cleanup_runtime_backup_streams=False,
        cleanup_drive_raw_dirs=False,
        cleanup_drive_label_cache=False,
        cleanup_models=False,
        cleanup_retention=False,
        cleanup_external_artifacts=True,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
    )

    step = report["steps"]["external_artifacts_cleanup"]
    moved_sources = {str(item.get("src", "")) for item in step.get("moved", []) if isinstance(item, dict)}
    assert str(external_quarantine) in moved_sources
    assert str(external_progress) in moved_sources
