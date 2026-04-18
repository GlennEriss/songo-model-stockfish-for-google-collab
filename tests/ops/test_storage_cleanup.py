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


def test_duplicate_source_metadata_cleanup_removes_raw_copy_when_identical(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)

    raw_dir = paths.drive_root / "data" / "raw_meta_source"
    sampled_dir = paths.drive_root / "data" / "sampled_meta_source"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_source_id": "sampled_meta_source", "status": "completed"}
    _write_json(raw_dir / "_dataset_source_metadata.json", payload)
    _write_json(sampled_dir / "_dataset_source_metadata.json", payload)
    _write_json(
        paths.data_root / "dataset_registry.json",
        {
            "dataset_sources": [
                {
                    "dataset_source_id": "sampled_meta_source",
                    "raw_dir": str(raw_dir),
                    "sampled_dir": str(sampled_dir),
                }
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
        cleanup_drive_raw_dirs=False,
        cleanup_drive_label_cache=False,
        cleanup_models=False,
        cleanup_retention=False,
        cleanup_external_artifacts=False,
        cleanup_duplicate_source_metadata=True,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
    )

    step = report["steps"]["duplicate_source_metadata_cleanup"]
    removed = set(str(item) for item in step.get("removed_raw_metadata", []))
    assert str(raw_dir / "_dataset_source_metadata.json") in removed


def test_global_progress_cleanup_removes_old_unprotected_targets(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    cfg["dataset_generation"] = {"global_target_id": "bench_models_20m_global"}
    paths = build_project_paths(cfg)

    progress_root = paths.data_root / "global_generation_progress"
    progress_root.mkdir(parents=True, exist_ok=True)

    protected_progress = progress_root / "bench_models_20m_global.json"
    legacy_progress = progress_root / "bench_models_legacy_global.json"
    recent_progress = progress_root / "bench_models_recent_global.json"
    protected_progress.write_text("{}", encoding="utf-8")
    legacy_progress.write_text("{}", encoding="utf-8")
    recent_progress.write_text("{}", encoding="utf-8")

    legacy_workers = progress_root / "bench_models_legacy_global.workers"
    legacy_workers.mkdir(parents=True, exist_ok=True)
    (legacy_workers / "worker_a.json").write_text("{}", encoding="utf-8")

    old_epoch = time.time() - 30.0 * 86400.0
    os.utime(legacy_progress, (old_epoch, old_epoch))
    os.utime(legacy_workers, (old_epoch, old_epoch))
    os.utime(legacy_workers / "worker_a.json", (old_epoch, old_epoch))

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
        cleanup_external_artifacts=False,
        cleanup_global_progress_mirrors=True,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
        retention_global_progress_ttl_seconds=7.0 * 86400.0,
        retention_global_progress_keep_recent=0,
    )

    step = report["steps"]["global_progress_cleanup"]
    removed_progress_files = set(str(item) for item in step.get("removed_progress_files", []))
    removed_workers_dirs = set(str(item) for item in step.get("removed_workers_dirs", []))

    assert str(legacy_progress) in removed_progress_files
    assert str(protected_progress) not in removed_progress_files
    assert str(recent_progress) not in removed_progress_files
    assert str(legacy_workers) in removed_workers_dirs


def test_pipeline_manifest_cleanup_respects_keep_recent_and_ttl(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)

    pipeline_root = paths.drive_root / "logs" / "pipeline"
    pipeline_root.mkdir(parents=True, exist_ok=True)

    old_manifest = pipeline_root / "dataset_pipeline_old.json"
    mid_manifest = pipeline_root / "dataset_pipeline_mid.json"
    latest_manifest = pipeline_root / "latest_dataset_pipeline_w1.json"
    old_manifest.write_text("{}", encoding="utf-8")
    mid_manifest.write_text("{}", encoding="utf-8")
    latest_manifest.write_text("{}", encoding="utf-8")

    old_epoch = time.time() - 30.0 * 86400.0
    mid_epoch = time.time() - 20.0 * 86400.0
    fresh_epoch = time.time() - 1.0 * 86400.0
    os.utime(old_manifest, (old_epoch, old_epoch))
    os.utime(mid_manifest, (mid_epoch, mid_epoch))
    os.utime(latest_manifest, (fresh_epoch, fresh_epoch))

    report = run_storage_cleanup(
        config=cfg,
        paths=paths,
        apply=False,
        cleanup_runtime_migration=False,
        cleanup_runtime_backup_streams=False,
        cleanup_drive_raw_dirs=False,
        cleanup_drive_label_cache=False,
        cleanup_models=False,
        cleanup_pipeline_manifests=True,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
        retention_pipeline_manifest_ttl_seconds=7.0 * 86400.0,
        retention_pipeline_manifest_keep_recent=1,
    )

    step = report["steps"]["pipeline_manifest_cleanup"]
    removed = set(str(item) for item in step.get("removed", []))
    assert str(old_manifest) in removed
    assert str(mid_manifest) in removed
    assert str(latest_manifest) not in removed


def test_completed_job_dirs_cleanup_keeps_recent_and_protected(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    cfg["train"] = {"job_id": "train_protected_job"}
    paths = build_project_paths(cfg)

    drive_jobs_root = paths.drive_root / "jobs"
    backup_jobs_root = paths.drive_root / "runtime_backup" / "jobs"
    drive_jobs_root.mkdir(parents=True, exist_ok=True)
    backup_jobs_root.mkdir(parents=True, exist_ok=True)

    train_old = drive_jobs_root / "train_old_job"
    train_recent = drive_jobs_root / "train_recent_job"
    train_protected = backup_jobs_root / "train_protected_job"
    for job_dir in [train_old, train_recent, train_protected]:
        job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(train_old / "run_status.json", {"run_type": "train", "status": "completed", "updated_at": "2026-01-01T00:00:00Z"})
    _write_json(
        train_recent / "run_status.json",
        {"run_type": "train", "status": "completed", "updated_at": "2026-04-17T00:00:00Z"},
    )
    _write_json(
        train_protected / "run_status.json",
        {"run_type": "train", "status": "completed", "updated_at": "2026-01-01T00:00:00Z"},
    )

    report = run_storage_cleanup(
        config=cfg,
        paths=paths,
        apply=False,
        cleanup_runtime_migration=False,
        cleanup_runtime_backup_streams=False,
        cleanup_drive_raw_dirs=False,
        cleanup_drive_label_cache=False,
        cleanup_models=False,
        cleanup_completed_job_dirs=True,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
        retention_job_dir_ttl_seconds=7.0 * 86400.0,
        retention_job_dir_keep_recent_per_run_type=1,
    )

    step = report["steps"]["completed_job_dirs_cleanup"]
    removed = {str(item.get("job_dir", "")) for item in step.get("removed", []) if isinstance(item, dict)}
    assert str(train_old) in removed
    assert str(train_recent) not in removed
    assert str(train_protected) not in removed


def test_duplicate_source_metadata_cleanup_uses_ttl_for_completed_mismatch(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)

    raw_dir = paths.drive_root / "data" / "raw_meta_ttl_source"
    sampled_dir = paths.drive_root / "data" / "sampled_meta_ttl_source"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)
    _write_json(raw_dir / "_dataset_source_metadata.json", {"dataset_source_id": "sampled_meta_ttl_source", "status": "partial"})
    _write_json(sampled_dir / "_dataset_source_metadata.json", {"dataset_source_id": "sampled_meta_ttl_source", "status": "completed"})
    _write_json(
        paths.data_root / "dataset_registry.json",
        {
            "dataset_sources": [
                {
                    "dataset_source_id": "sampled_meta_ttl_source",
                    "source_status": "completed",
                    "raw_dir": str(raw_dir),
                    "sampled_dir": str(sampled_dir),
                }
            ],
            "built_datasets": [],
        },
    )
    old_epoch = time.time() - 3.0 * 86400.0
    raw_meta = raw_dir / "_dataset_source_metadata.json"
    os.utime(raw_meta, (old_epoch, old_epoch))

    report = run_storage_cleanup(
        config=cfg,
        paths=paths,
        apply=False,
        cleanup_runtime_migration=False,
        cleanup_runtime_backup_streams=False,
        cleanup_drive_raw_dirs=False,
        cleanup_drive_label_cache=False,
        cleanup_models=False,
        cleanup_duplicate_source_metadata=True,
        keep_model_ids=[],
        keep_top_models=0,
        keep_dataset_ids=[],
        retention_source_metadata_raw_ttl_seconds=24.0 * 3600.0,
    )

    step = report["steps"]["duplicate_source_metadata_cleanup"]
    removed = set(str(item) for item in step.get("removed_raw_metadata", []))
    removed_due_to_ttl = set(str(item) for item in step.get("removed_raw_metadata_due_to_ttl", []))
    assert str(raw_meta) in removed
    assert str(raw_meta) in removed_due_to_ttl


def test_retention_drive_root_artifacts_hard_max_age_overrides_keep_recent(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    cfg = _make_config(drive_root)
    paths = build_project_paths(cfg)

    state_main = paths.drive_root / "state.json"
    state_dup = paths.drive_root / "state (1).json"
    state_main.write_text("{}", encoding="utf-8")
    state_dup.write_text("{}", encoding="utf-8")

    old_epoch = time.time() - 40.0 * 86400.0
    os.utime(state_main, (old_epoch, old_epoch))
    os.utime(state_dup, (old_epoch, old_epoch))

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
        retention_drive_root_artifact_ttl_seconds=365.0 * 86400.0,
        retention_drive_root_artifact_keep_recent_per_key=1,
        retention_drive_root_artifact_hard_max_age_seconds=30.0 * 86400.0,
    )

    step = report["steps"]["retention_cleanup"]
    removed = {
        str(item.get("path", ""))
        for item in step.get("drive_root_operational_artifacts_removed", [])
        if isinstance(item, dict)
    }
    assert str(state_main) in removed
    assert str(state_dup) in removed
    assert int(step.get("drive_root_operational_removed_hard_max_age", 0) or 0) >= 2
