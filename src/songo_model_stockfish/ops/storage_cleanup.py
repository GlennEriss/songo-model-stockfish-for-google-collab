from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from songo_model_stockfish.ops.model_registry import (
    load_registry,
    promote_best_model,
    promoted_best_metadata,
    save_registry,
)
from songo_model_stockfish.ops.paths import ProjectPaths
from songo_model_stockfish.ops.runtime_migration import (
    is_job_active,
    load_manifest_prefer_local,
    run_drive_to_local_runtime_migration,
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _path_within(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _parse_iso_to_epoch(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.astimezone(timezone.utc).timestamp())
    except Exception:
        return 0.0


def _is_safe_model_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if "/" in text or "\\" in text:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", text))


def _model_checkpoint_candidates_for_cleanup(checkpoints_dir: Path, model_id: str) -> list[Path]:
    model_text = str(model_id or "").strip()
    if not model_text or not checkpoints_dir.exists():
        return []
    allowed_prefixes = (f"{model_text}_", f"{model_text}-")
    candidates: list[Path] = []
    for path in sorted(checkpoints_dir.glob("*.pt")):
        stem = str(path.stem or "").strip()
        if stem == model_text or stem.startswith(allowed_prefixes):
            candidates.append(path)
    return candidates


def _load_json_dict(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = dict(default or {})
    if not path.exists():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    return payload if isinstance(payload, dict) else fallback


def _remove_file(path: Path, *, apply: bool) -> bool:
    if not path.exists():
        return False
    if not apply:
        return True
    try:
        path.unlink()
        return True
    except Exception:
        return False


def _remove_tree(path: Path, *, apply: bool) -> bool:
    if not path.exists():
        return False
    if not apply:
        return True
    try:
        shutil.rmtree(path, ignore_errors=False)
        return True
    except Exception:
        return False


def _sort_model_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (
            float(item.get("benchmark_score", -1.0)),
            float(item.get("evaluation_top1", -1.0)),
            float(item.get("sort_ts", 0.0)),
        ),
        reverse=True,
    )
    for idx, rec in enumerate(ranked, start=1):
        rec["rank"] = idx
    return ranked


def _load_latest_pipeline_manifest(drive_root: Path) -> tuple[dict[str, Any], str]:
    pipeline_root = drive_root / "logs" / "pipeline"
    candidates: list[Path] = []
    if pipeline_root.exists():
        candidates.extend(sorted(pipeline_root.glob("latest_dataset_pipeline_*.json")))
        candidates.extend(sorted(pipeline_root.glob("dataset_pipeline_*.json")))
    if not candidates:
        return {}, "none"
    latest = sorted(candidates, key=lambda p: float(p.stat().st_mtime), reverse=True)[0]
    return _load_json_dict(latest, default={}), str(latest)


def _collect_keep_model_ids(
    *,
    paths: ProjectPaths,
    explicit_keep: list[str],
    keep_top_models: int,
) -> set[str]:
    keep_ids = {str(item).strip() for item in explicit_keep if str(item).strip()}
    promoted = promoted_best_metadata(paths.models_root)
    if isinstance(promoted, dict):
        promoted_model_id = str(promoted.get("model_id", "")).strip()
        if promoted_model_id:
            keep_ids.add(promoted_model_id)
    registry = load_registry(paths.models_root)
    ranked = list(registry.get("models", [])) if isinstance(registry, dict) else []
    keep_top = max(0, int(keep_top_models))
    for rec in ranked[:keep_top]:
        if not isinstance(rec, dict):
            continue
        model_id = str(rec.get("model_id", "")).strip()
        if model_id:
            keep_ids.add(model_id)
    return keep_ids


def _resync_model_registry(paths: ProjectPaths, *, apply: bool) -> dict[str, Any]:
    models_root = paths.models_root
    final_dir = models_root / "final"
    existing_registry = load_registry(models_root)
    existing_records = list(existing_registry.get("models", [])) if isinstance(existing_registry, dict) else []
    record_map: dict[str, dict[str, Any]] = {}
    for item in existing_records:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id", "")).strip()
        if model_id:
            record_map[model_id] = dict(item)

    disk_model_ids = {
        path.stem.strip()
        for path in final_dir.glob("*.pt")
        if path.is_file() and path.stem.strip()
    }

    synced_records: list[dict[str, Any]] = []
    for model_id in sorted(disk_model_ids):
        checkpoint_path = final_dir / f"{model_id}.pt"
        rec = dict(record_map.get(model_id, {}))
        rec["model_id"] = model_id
        rec["checkpoint_path"] = str(checkpoint_path)
        rec.setdefault("sort_ts", float(checkpoint_path.stat().st_mtime))
        rec.setdefault("best_validation_metric", -1.0)
        rec.setdefault("evaluation_top1", -1.0)
        rec.setdefault("benchmark_score", -1.0)
        model_card_path = final_dir / f"{model_id}.model_card.json"
        if model_card_path.exists():
            rec["model_card_path"] = str(model_card_path)
        synced_records.append(rec)

    synced_records = _sort_model_records(synced_records)
    if apply:
        save_registry(models_root, {"models": synced_records})
        promoted_meta = promote_best_model(models_root)
    else:
        promoted_meta = synced_records[0] if synced_records else None

    return {
        "models_count": len(synced_records),
        "model_ids": [str(item.get("model_id", "")) for item in synced_records],
        "promoted": promoted_meta if isinstance(promoted_meta, dict) else {},
    }


def run_storage_cleanup(
    *,
    config: dict[str, Any],
    paths: ProjectPaths,
    apply: bool,
    cleanup_runtime_migration: bool,
    cleanup_runtime_backup_streams: bool,
    cleanup_drive_raw_dirs: bool,
    cleanup_drive_label_cache: bool,
    cleanup_models: bool,
    keep_model_ids: list[str] | None = None,
    keep_top_models: int = 1,
    keep_dataset_ids: list[str] | None = None,
    allow_purge_without_manifest: bool = False,
    drive_raw_cleanup_include_inactive_partial: bool = False,
    drive_raw_cleanup_inactive_min_age_seconds: float = 0.0,
) -> dict[str, Any]:
    keep_model_ids = list(keep_model_ids or [])
    keep_dataset_ids = [str(item).strip() for item in (keep_dataset_ids or []) if str(item).strip()]
    report: dict[str, Any] = {
        "apply": bool(apply),
        "cleanup_runtime_migration": bool(cleanup_runtime_migration),
        "cleanup_runtime_backup_streams": bool(cleanup_runtime_backup_streams),
        "cleanup_drive_raw_dirs": bool(cleanup_drive_raw_dirs),
        "cleanup_drive_label_cache": bool(cleanup_drive_label_cache),
        "cleanup_models": bool(cleanup_models),
        "drive_raw_cleanup_include_inactive_partial": bool(drive_raw_cleanup_include_inactive_partial),
        "drive_raw_cleanup_inactive_min_age_seconds": float(max(0.0, drive_raw_cleanup_inactive_min_age_seconds)),
        "steps": {},
    }

    drive_root = paths.drive_root
    jobs_root = paths.jobs_root
    logs_root = paths.logs_root
    data_root = paths.data_root
    models_root = paths.models_root

    if cleanup_runtime_migration:
        drive_jobs_root = drive_root / "jobs"
        drive_pipeline_logs_root = drive_root / "logs" / "pipeline"
        local_jobs_root = jobs_root
        local_pipeline_logs_root = logs_root / "pipeline"
        local_manifest_path = drive_root / "logs" / "pipeline" / "latest_dataset_pipeline_local.json"
        manifest_payload, manifest_source = load_manifest_prefer_local(
            local_manifest_path,
            firestore_manifest=_load_latest_pipeline_manifest(drive_root)[0],
        )
        allow_purge = bool(apply) and (bool(manifest_payload) or bool(allow_purge_without_manifest))
        migration_summary = run_drive_to_local_runtime_migration(
            drive_jobs_root=drive_jobs_root,
            drive_pipeline_logs_root=drive_pipeline_logs_root,
            local_jobs_root=local_jobs_root,
            local_pipeline_logs_root=local_pipeline_logs_root,
            manifest=manifest_payload,
            purge_after_verify=allow_purge,
            skip_active_job_dirs=True,
            active_updated_max_age_seconds=300.0,
            verbose=True,
            lock_dir=drive_root / "runtime_migration" / "locks" / "drive_to_local",
        )
        migration_summary["manifest_source"] = str(manifest_source)
        migration_summary["purge_requested"] = bool(apply)
        migration_summary["purge_effective"] = bool(allow_purge)
        if bool(apply) and not bool(allow_purge):
            migration_summary["purge_skipped_reason"] = "missing_manifest"
        report["steps"]["runtime_migration"] = migration_summary

    if cleanup_runtime_backup_streams:
        backup_root = paths.jobs_backup_root or (drive_root / "runtime_backup" / "jobs")
        step = {
            "backup_root": str(backup_root),
            "jobs_scanned": 0,
            "jobs_skipped_active": 0,
            "events_removed": 0,
            "metrics_removed": 0,
            "errors": [],
        }
        if backup_root.exists():
            for job_dir in sorted([p for p in backup_root.iterdir() if p.is_dir()]):
                step["jobs_scanned"] = int(step["jobs_scanned"]) + 1
                active, _ = is_job_active(job_dir, active_updated_max_age_seconds=600.0)
                if active:
                    step["jobs_skipped_active"] = int(step["jobs_skipped_active"]) + 1
                    continue
                events_path = job_dir / "events.jsonl"
                metrics_path = job_dir / "metrics.jsonl"
                if _remove_file(events_path, apply=apply):
                    step["events_removed"] = int(step["events_removed"]) + 1
                if _remove_file(metrics_path, apply=apply):
                    step["metrics_removed"] = int(step["metrics_removed"]) + 1
        report["steps"]["runtime_backup_stream_cleanup"] = step

    dataset_registry = _load_json_dict(data_root / "dataset_registry.json", default={"dataset_sources": [], "built_datasets": []})

    if cleanup_drive_raw_dirs:
        raw_partial_min_age_seconds = max(0.0, _as_float(drive_raw_cleanup_inactive_min_age_seconds, 0.0))
        step = {
            "raw_dirs_removed": [],
            "raw_dirs_skipped": [],
            "raw_dirs_skipped_reason": [],
            "include_inactive_partial": bool(drive_raw_cleanup_include_inactive_partial),
            "inactive_min_age_seconds": float(raw_partial_min_age_seconds),
        }
        candidates: set[Path] = set()
        now_epoch = time.time()
        for entry in dataset_registry.get("dataset_sources", []):
            if not isinstance(entry, dict):
                continue
            status_text = str(entry.get("source_status", "")).strip().lower()
            raw_dir_text = str(entry.get("raw_dir", "")).strip()
            sampled_dir_text = str(entry.get("sampled_dir", "")).strip()
            dataset_source_id = str(entry.get("dataset_source_id", "")).strip()
            if not raw_dir_text:
                continue
            raw_dir = Path(raw_dir_text)
            sampled_dir = Path(sampled_dir_text) if sampled_dir_text else None
            if not raw_dir.exists() or not raw_dir.is_dir():
                continue
            if not _path_within(raw_dir, drive_root):
                continue
            if sampled_dir is not None and sampled_dir.exists():
                if status_text == "completed":
                    candidates.add(raw_dir)
                    continue
                if bool(drive_raw_cleanup_include_inactive_partial):
                    updated_epoch = max(
                        _parse_iso_to_epoch(entry.get("updated_at")),
                        _parse_iso_to_epoch(entry.get("source_updated_at")),
                        _parse_iso_to_epoch(entry.get("last_updated_at")),
                    )
                    if updated_epoch <= 0.0:
                        step["raw_dirs_skipped_reason"].append(
                            {
                                "dataset_source_id": dataset_source_id,
                                "raw_dir": str(raw_dir),
                                "reason": "status_not_completed_and_no_timestamp",
                            }
                        )
                        continue
                    age_seconds = max(0.0, now_epoch - updated_epoch)
                    if age_seconds >= raw_partial_min_age_seconds:
                        candidates.add(raw_dir)
                        continue
                    step["raw_dirs_skipped_reason"].append(
                        {
                            "dataset_source_id": dataset_source_id,
                            "raw_dir": str(raw_dir),
                            "reason": "status_not_completed_recent_activity",
                            "age_seconds": float(age_seconds),
                        }
                    )
                    continue
                step["raw_dirs_skipped_reason"].append(
                    {
                        "dataset_source_id": dataset_source_id,
                        "raw_dir": str(raw_dir),
                        "reason": "status_not_completed",
                        "status": status_text or "<none>",
                    }
                )
        for raw_dir in sorted(candidates):
            if _remove_tree(raw_dir, apply=apply):
                step["raw_dirs_removed"].append(str(raw_dir))
            else:
                step["raw_dirs_skipped"].append(str(raw_dir))
        report["steps"]["drive_raw_cleanup"] = step

    if cleanup_drive_label_cache:
        keep_dataset_set = set(keep_dataset_ids)
        train_cfg = config.get("train", {}) if isinstance(config.get("train"), dict) else {}
        eval_cfg = config.get("evaluation", {}) if isinstance(config.get("evaluation"), dict) else {}
        build_cfg = config.get("dataset_build", {}) if isinstance(config.get("dataset_build"), dict) else {}
        for source_cfg in [train_cfg, eval_cfg, build_cfg]:
            dataset_id = str(source_cfg.get("dataset_id", "")).strip()
            if dataset_id and dataset_id not in {"auto", "<auto>"}:
                keep_dataset_set.add(dataset_id)
        built_entries = [entry for entry in dataset_registry.get("built_datasets", []) if isinstance(entry, dict)]
        if built_entries:
            largest = max(
                built_entries,
                key=lambda entry: (
                    int(entry.get("labeled_samples", 0) or 0),
                    str(entry.get("updated_at", "")),
                ),
            )
            largest_id = str(largest.get("dataset_id", "")).strip()
            if largest_id:
                keep_dataset_set.add(largest_id)
        step = {
            "label_cache_root": str(data_root / "label_cache"),
            "keep_dataset_ids": sorted(keep_dataset_set),
            "removed": [],
            "skipped": [],
        }
        label_cache_root = data_root / "label_cache"
        if label_cache_root.exists() and _path_within(label_cache_root, drive_root):
            for dataset_cache_dir in sorted([p for p in label_cache_root.iterdir() if p.is_dir()]):
                dataset_id = str(dataset_cache_dir.name).strip()
                if dataset_id in keep_dataset_set:
                    continue
                if _remove_tree(dataset_cache_dir, apply=apply):
                    step["removed"].append(str(dataset_cache_dir))
                else:
                    step["skipped"].append(str(dataset_cache_dir))
        report["steps"]["label_cache_cleanup"] = step

    if cleanup_models:
        keep_ids = _collect_keep_model_ids(
            paths=paths,
            explicit_keep=list(keep_model_ids),
            keep_top_models=max(0, int(keep_top_models)),
        )
        final_dir = models_root / "final"
        checkpoints_dir = models_root / "checkpoints"
        lineage_dir = models_root / "lineage"
        step = {
            "keep_model_ids": sorted(keep_ids),
            "removed_models": [],
            "skipped_unsafe_model_ids": [],
        }
        registry = load_registry(models_root)
        ranked = [item for item in registry.get("models", []) if isinstance(item, dict)]
        candidate_ids = {
            str(item.get("model_id", "")).strip()
            for item in ranked
            if str(item.get("model_id", "")).strip()
        }
        candidate_ids.update(
            {
                path.stem.strip()
                for path in final_dir.glob("*.pt")
                if path.is_file() and path.stem.strip()
            }
        )
        for model_id in sorted(candidate_ids):
            if not _is_safe_model_id(model_id):
                step["skipped_unsafe_model_ids"].append(str(model_id))
                continue
            if model_id in keep_ids:
                continue
            removed_paths: list[str] = []
            candidates = [
                final_dir / f"{model_id}.pt",
                final_dir / f"{model_id}.model_card.json",
                lineage_dir / f"{model_id}_parent_snapshot.pt",
            ]
            if checkpoints_dir.exists():
                candidates.extend(_model_checkpoint_candidates_for_cleanup(checkpoints_dir, model_id))
            for path in candidates:
                if not _path_within(path, models_root):
                    continue
                if _remove_file(path, apply=apply):
                    removed_paths.append(str(path))
            step["removed_models"].append(
                {
                    "model_id": model_id,
                    "removed_paths": removed_paths,
                    "removed_count": len(removed_paths),
                }
            )
        step["registry_sync"] = _resync_model_registry(paths, apply=apply)
        report["steps"]["model_cleanup"] = step

    report["created_at_epoch"] = time.time()
    return report
