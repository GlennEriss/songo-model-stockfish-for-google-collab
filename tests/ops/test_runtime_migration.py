from __future__ import annotations

import json
from pathlib import Path

from songo_model_stockfish.ops.runtime_migration import (
    run_drive_to_local_runtime_migration,
    sync_tree_with_hash_verify,
    verify_tree_hash,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def test_verify_tree_hash_detects_same_size_hash_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir(parents=True, exist_ok=True)
    dst.mkdir(parents=True, exist_ok=True)

    (src / "a.txt").write_text("abcdef", encoding="utf-8")
    (dst / "a.txt").write_text("abcdeg", encoding="utf-8")

    result = verify_tree_hash(src, dst)
    assert result["verified"] is False
    assert int(result["mismatch_hash_count"]) == 1
    assert int(result["mismatch_size_count"]) == 0


def test_run_migration_skips_active_job_when_manifest_pid_alive(tmp_path: Path) -> None:
    drive_jobs_root = tmp_path / "drive_jobs"
    drive_logs_root = tmp_path / "drive_logs_pipeline"
    local_jobs_root = tmp_path / "local_jobs"
    local_logs_root = tmp_path / "local_logs_pipeline"
    active_job_id = "dataset_job_active_001"
    active_job_dir = drive_jobs_root / active_job_id
    active_job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        active_job_dir / "run_status.json",
        {"status": "running", "updated_at": "2026-04-13T12:00:00Z"},
    )
    (active_job_dir / "state.json").write_text("{}", encoding="utf-8")

    manifest = {"generate_job_id": active_job_id, "generate_pid": 4242}

    summary = run_drive_to_local_runtime_migration(
        drive_jobs_root=drive_jobs_root,
        drive_pipeline_logs_root=drive_logs_root,
        local_jobs_root=local_jobs_root,
        local_pipeline_logs_root=local_logs_root,
        manifest=manifest,
        purge_after_verify=True,
        skip_active_job_dirs=True,
        pid_check_fn=lambda pid: int(pid) == 4242,
    )

    assert int(summary["jobs"]["skipped_active"]) == 1
    assert (drive_jobs_root / active_job_id).exists()
    assert not (local_jobs_root / active_job_id).exists()


def test_run_migration_copies_and_purges_inactive_job(tmp_path: Path) -> None:
    drive_jobs_root = tmp_path / "drive_jobs"
    drive_logs_root = tmp_path / "drive_logs_pipeline"
    local_jobs_root = tmp_path / "local_jobs"
    local_logs_root = tmp_path / "local_logs_pipeline"
    inactive_job_id = "dataset_job_done_001"
    inactive_job_dir = drive_jobs_root / inactive_job_id
    inactive_job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        inactive_job_dir / "run_status.json",
        {"status": "completed", "updated_at": "2026-04-13T12:00:00Z"},
    )
    (inactive_job_dir / "state.json").write_text('{"samples": 42}', encoding="utf-8")
    (drive_logs_root / "a.log").parent.mkdir(parents=True, exist_ok=True)
    (drive_logs_root / "a.log").write_text("ok", encoding="utf-8")

    summary = run_drive_to_local_runtime_migration(
        drive_jobs_root=drive_jobs_root,
        drive_pipeline_logs_root=drive_logs_root,
        local_jobs_root=local_jobs_root,
        local_pipeline_logs_root=local_logs_root,
        manifest={},
        purge_after_verify=True,
        skip_active_job_dirs=True,
        pid_check_fn=lambda _pid: False,
    )

    assert int(summary["jobs"]["migrated"]) == 1
    assert int(summary["jobs"]["purged"]) == 1
    assert (local_jobs_root / inactive_job_id / "state.json").exists()
    assert not (drive_jobs_root / inactive_job_id).exists()
    assert (local_logs_root / "a.log").exists()

    verify = sync_tree_with_hash_verify(local_jobs_root / inactive_job_id, local_jobs_root / inactive_job_id)
    assert verify["verified"] is True
