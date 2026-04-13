from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from songo_model_stockfish.data import jobs
from songo_model_stockfish.ops.job import create_job_context


def _base_config(*, tmp_path: Path, job_id: str, async_merge: bool) -> dict:
    runtime_root = tmp_path / "runtime"
    drive_root = tmp_path / "drive"
    return {
        "storage": {
            "drive_root": str(drive_root),
            "jobs_root": str(runtime_root / "jobs"),
            "logs_root": str(runtime_root / "logs"),
            "reports_root": str(runtime_root / "reports"),
            "models_root": str(runtime_root / "models"),
            "data_root": str(runtime_root / "data"),
        },
        "job": {
            "run_type": "dataset_build",
            "job_id": job_id,
            "auto_rollover_completed_job": False,
        },
        "dataset_build": {
            "dataset_id": "dataset_local_worker",
            "source_dataset_id": "source_local_worker",
            "build_mode": "teacher_label",
            "num_workers": 1,
            "max_pending_futures": 1,
            "follow_source_updates": False,
            "adaptive_source_polling": False,
            "stop_when_global_target_reached": False,
            "target_labeled_samples": 0,
            "export_partial_every_n_files": 1,
            "progressive_global_merge_enabled": True,
            "progressive_global_merge_dataset_id": "dataset_global",
            "progressive_global_merge_source_dataset_id_prefix": "dataset_local_",
            "progressive_global_merge_include_partial": True,
            "progressive_global_merge_every_n_partial_exports": 1,
            "progressive_global_merge_min_interval_seconds": 0.0,
            "progressive_global_merge_min_sources": 2,
            "progressive_global_merge_on_completion": True,
            "progressive_global_merge_dedupe_sample_ids": True,
            "progressive_global_merge_lock_wait_seconds": 5.0,
            "progressive_global_merge_lock_ttl_seconds": 120.0,
            "progressive_global_merge_async": bool(async_merge),
            "progressive_global_merge_candidates_cache_ttl_seconds": 600.0,
            "progressive_global_merge_require_data_delta": True,
            "progressive_global_merge_completion_wait_seconds": 10.0,
        },
    }


def _prepare_sampled_files(sampled_dir: Path) -> None:
    sampled_dir.mkdir(parents=True, exist_ok=True)
    (sampled_dir / "game_000001.jsonl").write_text("{}\n", encoding="utf-8")
    (sampled_dir / "game_000002.jsonl").write_text("{}\n", encoding="utf-8")


def _prepare_merge_candidate_npz(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    arrays = {"x": np.array([1], dtype=np.int8)}
    np.savez_compressed(root / "train.npz", **arrays)
    np.savez_compressed(root / "validation.npz", **arrays)
    np.savez_compressed(root / "test.npz", **arrays)


def _stub_export_snapshot(*_args, **kwargs):
    labeled_samples_total = int(kwargs.get("labeled_samples_total", 0))
    return (
        {
            "train": {"games": 0, "samples": labeled_samples_total},
            "validation": {"games": 0, "samples": 0},
            "test": {"games": 0, "samples": 0},
        },
        175,
        0,
    )


def _stub_label_samples_from_file(*_args, **_kwargs):
    # Return 1 labeled sample per file.
    return 1, [{"sample_id": "s"}], 0, 0


def test_progressive_merge_is_idempotent_with_signature_guard_and_registry_cache(tmp_path: Path, monkeypatch) -> None:
    cfg = _base_config(tmp_path=tmp_path, job_id="job_progressive_sync_001", async_merge=False)
    ctx = create_job_context(cfg, override_job_id="job_progressive_sync_001")

    sampled_dir = tmp_path / "sampled"
    _prepare_sampled_files(sampled_dir)
    _prepare_merge_candidate_npz(tmp_path / "dataset_local_shard_a")
    _prepare_merge_candidate_npz(tmp_path / "dataset_local_shard_b")

    merge_calls: list[dict] = []
    registry_reads = {"count": 0}

    def _stub_resolve_dataset_source(_job, _dataset_source_id: str) -> dict:
        return {
            "dataset_source_id": "source_local_worker",
            "source_mode": "benchmatch",
            "sampled_dir": str(sampled_dir),
            "source_status": "completed",
        }

    def _stub_read_dataset_registry(_job):
        registry_reads["count"] += 1
        return {
            "built_datasets": [
                {
                    "dataset_id": "dataset_global",
                    "build_status": "partial",
                    "labeled_samples": 100,
                    "output_dir": str(tmp_path / "dataset_global"),
                },
                {
                    "dataset_id": "dataset_local_shard_a",
                    "build_status": "partial",
                    "labeled_samples": 25,
                    "output_dir": str(tmp_path / "dataset_local_shard_a"),
                },
                {
                    "dataset_id": "dataset_local_shard_b",
                    "build_status": "partial",
                    "labeled_samples": 35,
                    "output_dir": str(tmp_path / "dataset_local_shard_b"),
                },
            ]
        }

    def _stub_run_dataset_merge_final(_job, *, cfg_override=None):
        merge_calls.append(dict(cfg_override or {}))
        return {"labeled_samples": 999}

    monkeypatch.setattr(jobs, "_resolve_dataset_source", _stub_resolve_dataset_source)
    monkeypatch.setattr(jobs, "_label_samples_from_file", _stub_label_samples_from_file)
    monkeypatch.setattr(jobs, "_export_built_dataset_snapshot", _stub_export_snapshot)
    monkeypatch.setattr(jobs, "_read_dataset_registry", _stub_read_dataset_registry)
    monkeypatch.setattr(jobs, "run_dataset_merge_final", _stub_run_dataset_merge_final)

    summary = jobs.run_dataset_build(ctx)
    assert int(summary["labeled_samples"]) == 2

    # First partial snapshot triggers merge; subsequent attempts are skipped
    # because source signature has not changed.
    assert len(merge_calls) == 1
    # Candidate selection should be cached after first scan.
    assert int(registry_reads["count"]) == 1


def test_progressive_merge_async_skips_duplicate_inflight_runs(tmp_path: Path, monkeypatch) -> None:
    cfg = _base_config(tmp_path=tmp_path, job_id="job_progressive_async_001", async_merge=True)
    ctx = create_job_context(cfg, override_job_id="job_progressive_async_001")

    sampled_dir = tmp_path / "sampled_async"
    _prepare_sampled_files(sampled_dir)
    _prepare_merge_candidate_npz(tmp_path / "dataset_local_shard_async_a")
    _prepare_merge_candidate_npz(tmp_path / "dataset_local_shard_async_b")

    merge_calls = {"count": 0}

    def _stub_resolve_dataset_source(_job, _dataset_source_id: str) -> dict:
        return {
            "dataset_source_id": "source_local_worker",
            "source_mode": "benchmatch",
            "sampled_dir": str(sampled_dir),
            "source_status": "completed",
        }

    def _stub_read_dataset_registry(_job):
        return {
            "built_datasets": [
                {
                    "dataset_id": "dataset_global",
                    "build_status": "partial",
                    "labeled_samples": 100,
                    "output_dir": str(tmp_path / "dataset_global"),
                },
                {
                    "dataset_id": "dataset_local_shard_async_a",
                    "build_status": "partial",
                    "labeled_samples": 25,
                    "output_dir": str(tmp_path / "dataset_local_shard_async_a"),
                },
                {
                    "dataset_id": "dataset_local_shard_async_b",
                    "build_status": "partial",
                    "labeled_samples": 35,
                    "output_dir": str(tmp_path / "dataset_local_shard_async_b"),
                },
            ]
        }

    def _stub_run_dataset_merge_final(_job, *, cfg_override=None):
        _ = cfg_override
        merge_calls["count"] += 1
        # Keep merge inflight long enough so second export sees merge_inflight.
        time.sleep(0.25)
        return {"labeled_samples": 1200}

    monkeypatch.setattr(jobs, "_resolve_dataset_source", _stub_resolve_dataset_source)
    monkeypatch.setattr(jobs, "_label_samples_from_file", _stub_label_samples_from_file)
    monkeypatch.setattr(jobs, "_export_built_dataset_snapshot", _stub_export_snapshot)
    monkeypatch.setattr(jobs, "_read_dataset_registry", _stub_read_dataset_registry)
    monkeypatch.setattr(jobs, "run_dataset_merge_final", _stub_run_dataset_merge_final)

    summary = jobs.run_dataset_build(ctx)
    assert int(summary["labeled_samples"]) == 2
    assert int(merge_calls["count"]) == 1

    events_path = ctx.job_dir / "events.jsonl"
    events_text = events_path.read_text(encoding="utf-8")
    assert "dataset_build_progressive_global_merge_queued" in events_text
    assert "dataset_build_progressive_global_merge_completed" in events_text
