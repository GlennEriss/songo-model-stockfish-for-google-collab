from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from songo_model_stockfish.ops.io_utils import (
    acquire_lock_dir,
    read_json_dict,
    release_lock_dir,
    write_json_atomic,
)
from songo_model_stockfish.ops.logging import utc_now_iso
from songo_model_stockfish.ops.paths import ProjectPaths


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_iso_epoch(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return 0.0


def _history_path(paths: ProjectPaths) -> Path:
    return paths.data_root / "dataset_training_usage_history.json"


def _history_lock_dir(paths: ProjectPaths) -> Path:
    return paths.data_root / "dataset_training_usage_history.lock"


def _history_default_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": "",
        "entries": [],
    }


def _normalize_entry(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    source = payload if isinstance(payload, dict) else {}
    dataset_id = str(source.get("dataset_id", "")).strip()
    if not dataset_id:
        return None
    return {
        "job_id": str(source.get("job_id", "")).strip(),
        "dataset_id": dataset_id,
        "dataset_selection_mode": str(source.get("dataset_selection_mode", "")).strip(),
        "model_id": str(source.get("model_id", "")).strip(),
        "status": str(source.get("status", "completed")).strip().lower() or "completed",
        "started_at": str(source.get("started_at", "")).strip(),
        "completed_at": str(source.get("completed_at", "")).strip(),
        "recorded_at": str(source.get("recorded_at", "")).strip(),
        "completed_epochs": max(0, _as_int(source.get("completed_epochs", 0), 0)),
        "requested_epochs": max(0, _as_int(source.get("requested_epochs", 0), 0)),
        "best_validation_metric": _as_float(source.get("best_validation_metric", 0.0), 0.0),
        "job_dir": str(source.get("job_dir", "")).strip(),
        "training_summary_path": str(source.get("training_summary_path", "")).strip(),
        "source": str(source.get("source", "train_runtime")).strip() or "train_runtime",
    }


def _entry_timestamp_epoch(entry: dict[str, Any]) -> float:
    for key in ("completed_at", "started_at", "recorded_at"):
        epoch = _parse_iso_epoch(entry.get(key))
        if epoch > 0.0:
            return epoch
    return 0.0


def _entry_key(entry: dict[str, Any]) -> str:
    job_id = str(entry.get("job_id", "")).strip()
    if job_id:
        return f"job:{job_id}"
    path_hint = str(entry.get("training_summary_path", "")).strip()
    if path_hint:
        return f"path:{path_hint}"
    dataset_id = str(entry.get("dataset_id", "")).strip()
    model_id = str(entry.get("model_id", "")).strip()
    completed_at = str(entry.get("completed_at", "")).strip()
    return f"fallback:{dataset_id}:{model_id}:{completed_at}"


def _load_history_entries(paths: ProjectPaths) -> list[dict[str, Any]]:
    payload = read_json_dict(_history_path(paths), default=_history_default_payload())
    entries: list[dict[str, Any]] = []
    for raw in payload.get("entries", []):
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_entry(raw)
        if normalized is None:
            continue
        entries.append(normalized)
    return entries


def _save_history_entries(
    *,
    paths: ProjectPaths,
    entries: list[dict[str, Any]],
    max_entries: int,
    timeout_seconds: float = 30.0,
) -> None:
    lock_dir = _history_lock_dir(paths)
    lock_acquired = acquire_lock_dir(lock_dir, timeout_seconds=max(3.0, float(timeout_seconds)))
    if not lock_acquired:
        raise TimeoutError(f"Impossible d'acquerir le lock dataset history: {lock_dir}")
    try:
        dedup: dict[str, dict[str, Any]] = {}
        for item in entries:
            normalized = _normalize_entry(item)
            if normalized is None:
                continue
            key = _entry_key(normalized)
            previous = dedup.get(key)
            if previous is None or _entry_timestamp_epoch(normalized) >= _entry_timestamp_epoch(previous):
                dedup[key] = normalized
        sorted_entries = sorted(
            dedup.values(),
            key=lambda item: (_entry_timestamp_epoch(item), str(item.get("job_id", ""))),
            reverse=True,
        )
        kept_entries = sorted_entries[: max(1, int(max_entries))]
        payload = _history_default_payload()
        payload["schema_version"] = 1
        payload["updated_at"] = utc_now_iso()
        payload["entries"] = kept_entries
        write_json_atomic(_history_path(paths), payload, ensure_ascii=True, indent=2)
    finally:
        release_lock_dir(lock_dir)


def _scan_training_usage_entries(paths: ProjectPaths) -> list[dict[str, Any]]:
    roots: list[Path] = []
    candidates = [
        paths.jobs_root,
        paths.jobs_backup_root,
        paths.drive_root / "jobs",
        paths.drive_root / "runtime_backup" / "jobs",
        Path("/content/songo-stockfish-runtime/jobs"),
    ]
    env_jobs_root = str(os.environ.get("SONGO_JOBS_ROOT", "")).strip()
    if env_jobs_root:
        candidates.append(Path(env_jobs_root))
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate in roots:
            continue
        roots.append(candidate)

    scanned: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            job_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
        except Exception:
            continue
        for job_dir in job_dirs:
            summary_path = job_dir / "training_summary.json"
            if not summary_path.exists():
                continue
            summary = read_json_dict(summary_path, default={})
            if not summary:
                continue
            run_status = read_json_dict(job_dir / "run_status.json", default={})
            run_type = str(run_status.get("run_type", "")).strip().lower()
            if run_type and run_type != "train":
                continue
            dataset_id = str(summary.get("dataset_id", "")).strip()
            if not dataset_id:
                dataset_id = str(run_status.get("dataset_id", "")).strip()
            if not dataset_id:
                continue
            completed_at = str(run_status.get("updated_at", "")).strip() or str(summary.get("updated_at", "")).strip()
            entry = _normalize_entry(
                {
                    "job_id": str(summary.get("job_id", "")).strip() or str(job_dir.name).strip(),
                    "dataset_id": dataset_id,
                    "dataset_selection_mode": str(summary.get("dataset_selection_mode", "")).strip(),
                    "model_id": str(summary.get("model_id", "")).strip() or str(run_status.get("model_id", "")).strip(),
                    "status": str(run_status.get("status", "completed")).strip().lower() or "completed",
                    "started_at": str(run_status.get("created_at", "")).strip(),
                    "completed_at": completed_at,
                    "recorded_at": str(run_status.get("updated_at", "")).strip() or utc_now_iso(),
                    "completed_epochs": _as_int(
                        summary.get("completed_epochs", summary.get("epochs_completed", 0)),
                        0,
                    ),
                    "requested_epochs": _as_int(summary.get("epochs", summary.get("max_epochs", 0)), 0),
                    "best_validation_metric": _as_float(
                        summary.get("best_validation_metric", summary.get("best_val_metric", 0.0)),
                        0.0,
                    ),
                    "job_dir": str(job_dir),
                    "training_summary_path": str(summary_path),
                    "source": "job_scan",
                }
            )
            if entry is not None:
                scanned.append(entry)
    return scanned


def _merge_entries(base_entries: list[dict[str, Any]], incoming_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in base_entries + incoming_entries:
        normalized = _normalize_entry(item)
        if normalized is None:
            continue
        key = _entry_key(normalized)
        previous = merged.get(key)
        if previous is None:
            merged[key] = normalized
            continue
        prev_score = (_entry_timestamp_epoch(previous), 1 if previous.get("source") == "train_runtime" else 0)
        new_score = (_entry_timestamp_epoch(normalized), 1 if normalized.get("source") == "train_runtime" else 0)
        if new_score >= prev_score:
            merged[key] = normalized
    return sorted(
        merged.values(),
        key=lambda item: (_entry_timestamp_epoch(item), str(item.get("job_id", ""))),
        reverse=True,
    )


def _load_built_dataset_map(data_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json_dict(data_root / "dataset_registry.json", default={"built_datasets": []})
    built = payload.get("built_datasets", [])
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(built, list):
        return result
    for entry in built:
        if not isinstance(entry, dict):
            continue
        dataset_id = str(entry.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        result[dataset_id] = dict(entry)
    return result


def _collect_config_keep_dataset_ids(config: dict[str, Any] | None) -> set[str]:
    cfg = config if isinstance(config, dict) else {}
    keep: set[str] = set()
    for block_name in ("train", "evaluation", "dataset_build", "dataset_merge_final"):
        block = cfg.get(block_name, {})
        if not isinstance(block, dict):
            continue
        dataset_id = str(block.get("dataset_id", "")).strip()
        if dataset_id and dataset_id.lower() not in {"auto", "<auto>"}:
            keep.add(dataset_id)
        source_dataset_id = str(block.get("source_dataset_id", "")).strip()
        if source_dataset_id and source_dataset_id.lower() not in {"auto", "<auto>"}:
            keep.add(source_dataset_id)
        source_dataset_ids = block.get("source_dataset_ids", [])
        if isinstance(source_dataset_ids, list):
            for value in source_dataset_ids:
                candidate = str(value or "").strip()
                if candidate and candidate.lower() not in {"auto", "<auto>"}:
                    keep.add(candidate)
    return keep


def _aggregate_dataset_usage(
    *,
    entries: list[dict[str, Any]],
    built_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for entry in entries:
        dataset_id = str(entry.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        aggregate = by_dataset.get(dataset_id)
        if aggregate is None:
            built_meta = built_map.get(dataset_id, {})
            aggregate = {
                "dataset_id": dataset_id,
                "train_runs": 0,
                "completed_runs": 0,
                "first_used_at": "",
                "last_used_at": "",
                "first_used_epoch": 0.0,
                "last_used_epoch": 0.0,
                "total_completed_epochs": 0,
                "max_best_validation_metric": 0.0,
                "unique_models": set(),
                "labeled_samples": _as_int(built_meta.get("labeled_samples", 0), 0),
                "build_mode": str(built_meta.get("build_mode", "")).strip(),
                "output_dir": str(built_meta.get("output_dir", "")).strip(),
                "updated_at_registry": str(built_meta.get("updated_at", "")).strip(),
            }
            by_dataset[dataset_id] = aggregate

        aggregate["train_runs"] = int(aggregate["train_runs"]) + 1
        if str(entry.get("status", "")).strip().lower() == "completed":
            aggregate["completed_runs"] = int(aggregate["completed_runs"]) + 1
        aggregate["total_completed_epochs"] = int(aggregate["total_completed_epochs"]) + max(
            0, _as_int(entry.get("completed_epochs", 0), 0)
        )
        aggregate["max_best_validation_metric"] = max(
            float(aggregate["max_best_validation_metric"]),
            _as_float(entry.get("best_validation_metric", 0.0), 0.0),
        )
        model_id = str(entry.get("model_id", "")).strip()
        if model_id:
            casted = aggregate.get("unique_models")
            if isinstance(casted, set):
                casted.add(model_id)

        timestamp_epoch = _entry_timestamp_epoch(entry)
        if timestamp_epoch > 0.0:
            last_used_epoch = float(aggregate.get("last_used_epoch", 0.0))
            if timestamp_epoch >= last_used_epoch:
                aggregate["last_used_epoch"] = timestamp_epoch
                aggregate["last_used_at"] = (
                    str(entry.get("completed_at", "")).strip()
                    or str(entry.get("started_at", "")).strip()
                    or str(entry.get("recorded_at", "")).strip()
                )
            first_used_epoch = float(aggregate.get("first_used_epoch", 0.0))
            if first_used_epoch <= 0.0 or timestamp_epoch <= first_used_epoch:
                aggregate["first_used_epoch"] = timestamp_epoch
                aggregate["first_used_at"] = (
                    str(entry.get("completed_at", "")).strip()
                    or str(entry.get("started_at", "")).strip()
                    or str(entry.get("recorded_at", "")).strip()
                )

    aggregated_rows: list[dict[str, Any]] = []
    for aggregate in by_dataset.values():
        models_raw = aggregate.get("unique_models")
        models = sorted(str(item) for item in models_raw) if isinstance(models_raw, set) else []
        row = dict(aggregate)
        row["unique_models"] = models
        row["unique_models_count"] = len(models)
        aggregated_rows.append(row)

    return sorted(
        aggregated_rows,
        key=lambda item: (
            _as_int(item.get("completed_runs", 0), 0),
            _as_int(item.get("train_runs", 0), 0),
            _as_float(item.get("last_used_epoch", 0.0), 0.0),
            _as_int(item.get("labeled_samples", 0), 0),
        ),
        reverse=True,
    )


def record_training_dataset_usage(
    *,
    paths: ProjectPaths,
    job_id: str,
    dataset_id: str,
    dataset_selection_mode: str,
    model_id: str,
    status: str = "completed",
    started_at: str = "",
    completed_at: str = "",
    completed_epochs: int = 0,
    requested_epochs: int = 0,
    best_validation_metric: float = 0.0,
    job_dir: str = "",
    training_summary_path: str = "",
    keep_entries: int = 4000,
) -> dict[str, Any]:
    entry = _normalize_entry(
        {
            "job_id": str(job_id).strip(),
            "dataset_id": str(dataset_id).strip(),
            "dataset_selection_mode": str(dataset_selection_mode).strip(),
            "model_id": str(model_id).strip(),
            "status": str(status).strip().lower() or "completed",
            "started_at": str(started_at).strip(),
            "completed_at": str(completed_at).strip() or utc_now_iso(),
            "recorded_at": utc_now_iso(),
            "completed_epochs": int(max(0, int(completed_epochs))),
            "requested_epochs": int(max(0, int(requested_epochs))),
            "best_validation_metric": float(best_validation_metric),
            "job_dir": str(job_dir).strip(),
            "training_summary_path": str(training_summary_path).strip(),
            "source": "train_runtime",
        }
    )
    if entry is None:
        raise ValueError("dataset_id manquant pour enregistrer l'historique dataset train.")

    existing_entries = _load_history_entries(paths)
    merged_entries = _merge_entries(existing_entries, [entry])
    _save_history_entries(paths=paths, entries=merged_entries, max_entries=max(1, int(keep_entries)))

    dataset_runs = 0
    dataset_completed_runs = 0
    for row in merged_entries:
        if str(row.get("dataset_id", "")).strip() != str(entry.get("dataset_id", "")).strip():
            continue
        dataset_runs += 1
        if str(row.get("status", "")).strip().lower() == "completed":
            dataset_completed_runs += 1

    return {
        "history_path": str(_history_path(paths)),
        "entries_total": len(merged_entries),
        "dataset_id": str(entry.get("dataset_id", "")).strip(),
        "dataset_train_runs": int(dataset_runs),
        "dataset_completed_runs": int(dataset_completed_runs),
    }


def build_dataset_usage_report(
    *,
    paths: ProjectPaths,
    config: dict[str, Any] | None = None,
    include_job_scan: bool = True,
    sync_history_from_jobs: bool = False,
    keep_dataset_ids: list[str] | None = None,
    purge_min_age_seconds: float = 7.0 * 86400.0,
    max_purge_candidates: int = 25,
    top_n: int = 10,
    history_max_entries: int = 4000,
) -> dict[str, Any]:
    explicit_keep = {str(item).strip() for item in (keep_dataset_ids or []) if str(item).strip()}
    persisted_entries = _load_history_entries(paths)
    scanned_entries = _scan_training_usage_entries(paths) if bool(include_job_scan) else []
    merged_entries = _merge_entries(persisted_entries, scanned_entries)

    synced = False
    if bool(sync_history_from_jobs):
        if len(merged_entries) != len(persisted_entries):
            _save_history_entries(paths=paths, entries=merged_entries, max_entries=max(1, int(history_max_entries)))
            synced = True
        else:
            persisted_keys = [_entry_key(item) for item in persisted_entries]
            merged_keys = [_entry_key(item) for item in merged_entries]
            if persisted_keys != merged_keys:
                _save_history_entries(paths=paths, entries=merged_entries, max_entries=max(1, int(history_max_entries)))
                synced = True

    built_map = _load_built_dataset_map(paths.data_root)
    ranked = _aggregate_dataset_usage(entries=merged_entries, built_map=built_map)
    now_epoch = time.time()
    total_completed_runs = sum(_as_int(item.get("completed_runs", 0), 0) for item in ranked)

    largest_built_dataset_id = ""
    if built_map:
        largest_built_dataset_id = max(
            built_map.items(),
            key=lambda item: (
                _as_int(item[1].get("labeled_samples", 0), 0),
                str(item[1].get("updated_at", "")),
            ),
        )[0]
    most_used_dataset_id = str(ranked[0].get("dataset_id", "")).strip() if ranked else ""
    recommended_primary_dataset_id = largest_built_dataset_id or most_used_dataset_id

    keep_set = set(explicit_keep)
    keep_set.update(_collect_config_keep_dataset_ids(config))
    if recommended_primary_dataset_id:
        keep_set.add(recommended_primary_dataset_id)

    datasets_ranked: list[dict[str, Any]] = []
    for item in ranked:
        row = dict(item)
        completed_runs = _as_int(row.get("completed_runs", 0), 0)
        row["completed_runs_share"] = (
            float(completed_runs) / float(total_completed_runs) if total_completed_runs > 0 else 0.0
        )
        last_used_epoch = _as_float(row.get("last_used_epoch", 0.0), 0.0)
        row["last_used_age_seconds"] = max(0.0, now_epoch - last_used_epoch) if last_used_epoch > 0.0 else None
        row["is_keep"] = str(row.get("dataset_id", "")).strip() in keep_set
        datasets_ranked.append(row)

    purge_age_threshold = max(0.0, float(purge_min_age_seconds))
    purge_candidates: list[dict[str, Any]] = []
    for row in datasets_ranked:
        dataset_id = str(row.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        if bool(row.get("is_keep", False)):
            continue
        age_raw = row.get("last_used_age_seconds")
        age_seconds = _as_float(age_raw, 10_000_000_000.0) if age_raw is not None else 10_000_000_000.0
        if age_seconds < purge_age_threshold:
            continue
        reason = "not_kept_and_stale"
        if _as_int(row.get("completed_runs", 0), 0) <= 0:
            reason = "never_completed_train"
        purge_candidates.append(
            {
                "dataset_id": dataset_id,
                "reason": reason,
                "completed_runs": _as_int(row.get("completed_runs", 0), 0),
                "train_runs": _as_int(row.get("train_runs", 0), 0),
                "last_used_at": str(row.get("last_used_at", "")).strip(),
                "last_used_age_seconds": age_seconds,
                "labeled_samples": _as_int(row.get("labeled_samples", 0), 0),
                "build_mode": str(row.get("build_mode", "")).strip(),
                "output_dir": str(row.get("output_dir", "")).strip(),
            }
        )
    purge_candidates = sorted(
        purge_candidates,
        key=lambda item: (
            _as_int(item.get("completed_runs", 0), 0),
            -_as_float(item.get("last_used_age_seconds", 0.0), 0.0),
            _as_int(item.get("labeled_samples", 0), 0),
        ),
    )[: max(0, int(max_purge_candidates))]

    return {
        "generated_at": utc_now_iso(),
        "history_path": str(_history_path(paths)),
        "history_entries_total": len(merged_entries),
        "history_entries_from_scan": len(scanned_entries),
        "history_synced_from_jobs": bool(synced),
        "most_used_dataset_id": most_used_dataset_id,
        "largest_built_dataset_id": largest_built_dataset_id,
        "recommended_primary_dataset_id": recommended_primary_dataset_id,
        "keep_dataset_ids": sorted(keep_set),
        "purge_min_age_seconds": purge_age_threshold,
        "total_completed_runs": int(total_completed_runs),
        "datasets_ranked": datasets_ranked,
        "top_datasets": datasets_ranked[: max(0, int(top_n))],
        "purge_candidates": purge_candidates,
    }
