from __future__ import annotations

import sys
import types
from pathlib import Path

from songo_model_stockfish.ops import job as job_ops


def _base_config(tmp_path: Path, *, run_type: str, job_id: str) -> dict:
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    return {
        "storage": {
            "drive_root": str(drive_root),
            "jobs_root": str(tmp_path / "runtime" / "jobs"),
            "logs_root": str(tmp_path / "runtime" / "logs"),
            "reports_root": str(tmp_path / "runtime" / "reports"),
            "models_root": str(tmp_path / "runtime" / "models"),
            "data_root": str(tmp_path / "runtime" / "data"),
            "runtime_state_backup_enabled": False,
        },
        "job": {
            "run_type": run_type,
            "job_id": job_id,
            "auto_rollover_completed_job": False,
        },
    }


def test_firestore_strict_default_is_false_for_train_eval_benchmark() -> None:
    base_cfg = {"firestore": {"job_firestore_backend": "firestore"}}
    for run_type in ("train", "evaluation", "benchmark"):
        resolved = job_ops._resolve_firestore_sync_config(base_cfg, run_type=run_type)
        assert resolved["enabled"] is True
        assert resolved["strict"] is False
        assert int(resolved["retry_attempts"]) == 3
        assert float(resolved["retry_backoff_seconds"]) == 1.0

    dataset_resolved = job_ops._resolve_firestore_sync_config(base_cfg, run_type="dataset_build")
    assert dataset_resolved["strict"] is True
    assert int(dataset_resolved["retry_attempts"]) == 1


def test_firestore_sync_retries_with_backoff_when_non_strict(tmp_path: Path, monkeypatch) -> None:
    cfg = _base_config(tmp_path, run_type="train", job_id="job_firestore_retry_001")
    ctx = job_ops.create_job_context(cfg, override_job_id="job_firestore_retry_001")

    # Active Firestore sync only for this explicit check.
    ctx.firestore_sync = {
        "enabled": True,
        "strict": False,
        "project_id": "test-project",
        "collection": "worker_checkpoints",
        "credentials_path": "fake.json",
        "api_key": "",
        "checkpoint_min_interval_seconds": 0.0,
        "checkpoint_state_only_on_change": False,
        "retry_attempts": 3,
        "retry_backoff_seconds": 0.5,
    }

    attempts = {"count": 0}

    class _FakeDocRef:
        def set(self, payload, merge=True):  # noqa: ANN001
            _ = payload
            _ = merge
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("transient failure")

    class _FakeCollection:
        def document(self, job_id: str) -> _FakeDocRef:
            _ = job_id
            return _FakeDocRef()

    class _FakeClient:
        def collection(self, name: str) -> _FakeCollection:
            _ = name
            return _FakeCollection()

    fake_firestore_mod = types.SimpleNamespace(DELETE_FIELD="__DELETE_FIELD__")
    fake_google_mod = types.ModuleType("google")
    fake_google_cloud_mod = types.ModuleType("google.cloud")
    fake_google_cloud_mod.firestore = fake_firestore_mod
    monkeypatch.setitem(sys.modules, "google", fake_google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_google_cloud_mod)
    monkeypatch.setattr(job_ops, "_build_firestore_job_client", lambda **_kwargs: _FakeClient())

    sleeps: list[float] = []
    monkeypatch.setattr(job_ops.time, "sleep", lambda seconds: sleeps.append(float(seconds)))

    ctx._write_worker_checkpoint_firestore(
        status_payload={"phase": "checkpoint"},
        state_payload={"counter": 1},
        force=True,
    )

    assert int(attempts["count"]) == 3
    assert sleeps == [0.5, 1.0]
