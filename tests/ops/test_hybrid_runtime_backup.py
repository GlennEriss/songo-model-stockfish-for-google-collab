from __future__ import annotations

import json
import time
from pathlib import Path

from songo_model_stockfish.ops.job import create_job_context


def _base_config(*, drive_root: Path, jobs_root: Path, jobs_backup_root: Path, job_id: str) -> dict:
    drive_root.mkdir(parents=True, exist_ok=True)
    return {
        "storage": {
            "drive_root": str(drive_root),
            "jobs_root": str(jobs_root),
            "logs_root": str(drive_root / "logs"),
            "reports_root": str(drive_root / "reports"),
            "models_root": str(drive_root / "models"),
            "data_root": str(drive_root / "data"),
            "runtime_state_backup_enabled": True,
            "jobs_backup_root": str(jobs_backup_root),
            "runtime_state_backup_min_interval_seconds": 3600.0,
            "runtime_state_backup_force_interval_seconds": 1.0,
        },
        "job": {
            "run_type": "dataset_build",
            "job_id": job_id,
            "auto_rollover_completed_job": True,
        },
    }


def test_events_metrics_are_mirrored_to_backup(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    jobs_root = tmp_path / "runtime" / "jobs"
    backup_root = drive_root / "runtime_backup" / "jobs"
    cfg = _base_config(
        drive_root=drive_root,
        jobs_root=jobs_root,
        jobs_backup_root=backup_root,
        job_id="job_events_metrics_001",
    )
    ctx = create_job_context(cfg, override_job_id="job_events_metrics_001")
    ctx.write_event("event_one", value=1)
    ctx.write_metric({"metric_type": "metric_one", "value": 2})

    backup_events = backup_root / ctx.job_id / "events.jsonl"
    backup_metrics = backup_root / ctx.job_id / "metrics.jsonl"
    assert backup_events.exists()
    assert backup_metrics.exists()
    assert "event_one" in backup_events.read_text(encoding="utf-8")
    assert "metric_one" in backup_metrics.read_text(encoding="utf-8")


def test_state_backup_force_interval_flushes_dirty_state(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    jobs_root = tmp_path / "runtime" / "jobs"
    backup_root = drive_root / "runtime_backup" / "jobs"
    cfg = _base_config(
        drive_root=drive_root,
        jobs_root=jobs_root,
        jobs_backup_root=backup_root,
        job_id="job_force_flush_001",
    )
    ctx = create_job_context(cfg, override_job_id="job_force_flush_001")

    ctx._last_runtime_backup_state_write_ts = time.time() - 5.0
    ctx._last_runtime_backup_state_signature = json.dumps(
        {"job_id": ctx.job_id, "run_type": ctx.run_type, "counter": 0},
        sort_keys=True,
        ensure_ascii=True,
    )
    ctx.write_state({"counter": 1})

    backup_state = backup_root / ctx.job_id / "state.json"
    payload = json.loads(backup_state.read_text(encoding="utf-8"))
    assert int(payload["counter"]) == 1


def test_read_state_restores_from_backup_and_emits_restore_signal(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    jobs_root = tmp_path / "runtime" / "jobs"
    backup_root = drive_root / "runtime_backup" / "jobs"
    job_id = "job_restore_state_001"
    cfg = _base_config(
        drive_root=drive_root,
        jobs_root=jobs_root,
        jobs_backup_root=backup_root,
        job_id=job_id,
    )
    ctx = create_job_context(cfg, override_job_id=job_id)
    ctx.write_state({"counter": 7})

    local_state = jobs_root / ctx.job_id / "state.json"
    if local_state.exists():
        local_state.unlink()

    restored = ctx.read_state()
    assert int(restored.get("counter", -1)) == 7
    assert local_state.exists()

    metrics_text = (jobs_root / ctx.job_id / "metrics.jsonl").read_text(encoding="utf-8")
    assert "runtime_backup_restored" in metrics_text


def test_create_job_context_restores_events_and_metrics_from_backup(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    jobs_root = tmp_path / "runtime" / "jobs"
    backup_root = drive_root / "runtime_backup" / "jobs"
    job_id = "job_restore_startup_001"
    backup_job_dir = backup_root / job_id
    backup_job_dir.mkdir(parents=True, exist_ok=True)
    (backup_job_dir / "config.yaml").write_text("job:\n  run_type: dataset_build\n", encoding="utf-8")
    (backup_job_dir / "run_status.json").write_text(
        json.dumps({"status": "running", "phase": "dataset_build"}, ensure_ascii=True),
        encoding="utf-8",
    )
    (backup_job_dir / "state.json").write_text(json.dumps({"counter": 3}, ensure_ascii=True), encoding="utf-8")
    (backup_job_dir / "events.jsonl").write_text(
        json.dumps({"message": "old_event"}, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (backup_job_dir / "metrics.jsonl").write_text(
        json.dumps({"metric_type": "old_metric"}, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    cfg = _base_config(
        drive_root=drive_root,
        jobs_root=jobs_root,
        jobs_backup_root=backup_root,
        job_id=job_id,
    )
    ctx = create_job_context(cfg, override_job_id=job_id)

    local_job_dir = jobs_root / ctx.job_id
    assert (local_job_dir / "events.jsonl").exists()
    assert (local_job_dir / "metrics.jsonl").exists()
    events_text = (local_job_dir / "events.jsonl").read_text(encoding="utf-8")
    metrics_text = (local_job_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert "old_event" in events_text
    assert "old_metric" in metrics_text
    assert "runtime_backup_restored_startup" in metrics_text
