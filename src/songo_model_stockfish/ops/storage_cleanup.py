from __future__ import annotations

import os
import json
import re
import shutil
import sys
import time
from collections import deque
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


def _path_age_seconds(path: Path, *, now_epoch: float | None = None) -> float:
    now_ts = float(now_epoch) if now_epoch is not None else time.time()
    try:
        modified_ts = float(path.stat().st_mtime)
    except Exception:
        return 0.0
    return max(0.0, now_ts - modified_ts)


def _job_dir_age_seconds(job_dir: Path, *, now_epoch: float | None = None) -> float:
    status_payload = _load_json_dict(job_dir / "run_status.json", default={})
    updated_epoch = _parse_iso_to_epoch(status_payload.get("updated_at"))
    if updated_epoch > 0.0:
        now_ts = float(now_epoch) if now_epoch is not None else time.time()
        return max(0.0, now_ts - updated_epoch)
    return _path_age_seconds(job_dir, now_epoch=now_epoch)


def _resolve_retention_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cleanup_cfg = config.get("storage_cleanup", {})
    if not isinstance(cleanup_cfg, dict):
        return {}
    retention_cfg = cleanup_cfg.get("retention", {})
    if not isinstance(retention_cfg, dict):
        return {}
    return dict(retention_cfg)


def _infer_checkpoint_model_id(stem: str) -> str:
    text = str(stem or "").strip()
    if not text:
        return ""
    if "_epoch_" in text:
        return text.split("_epoch_", 1)[0].strip()
    if text.endswith("_best") or text.endswith("_last"):
        return text.rsplit("_", 1)[0].strip()
    return text


def _cleanup_external_drive_artifacts(*, drive_root: Path, apply: bool, now_epoch: float | None = None) -> dict[str, Any]:
    # Nettoie des artefacts connus ecrits par erreur hors de drive_root,
    # mais strictement sous /content/drive/MyDrive pour eviter tout effet de bord.
    now_ts = float(now_epoch) if now_epoch is not None else time.time()
    mydrive_root = drive_root.parent
    recovered_root = drive_root / "runtime_migration" / "recovered_external"
    step: dict[str, Any] = {
        "mydrive_root": str(mydrive_root),
        "drive_root": str(drive_root),
        "recovered_root": str(recovered_root),
        "moved": [],
        "skipped": [],
        "errors": [],
        "scan_entries": 0,
        "scan_candidates": 0,
        "scan_truncated": False,
        "move_truncated": False,
    }
    if not mydrive_root.exists():
        return step
    session_root = recovered_root / str(int(now_ts))
    known_file_names = {
        "_dataset_source_metadata.json",
        "bench_models_20m_global.json",
        "dataset_registry.json",
        "dataset_generation_summary.json",
        "dataset_build_summary.json",
        "training_summary.json",
        "evaluation_summary.json",
        "benchmark_summary.json",
        "latest_dataset_pipeline",
        "latest_dataset_pipeline.json",
        "tournament_progress.latest.json",
        "config.yaml",
        "run_status.json",
        "state.json",
    }
    scan_max_depth = max(1, _as_int(os.environ.get("SONGO_EXTERNAL_ARTIFACT_SCAN_MAX_DEPTH", "6"), 6))
    scan_max_seconds = max(30.0, _as_float(os.environ.get("SONGO_EXTERNAL_ARTIFACT_SCAN_MAX_SECONDS", "300"), 300.0))
    force_full_scan = _as_bool(os.environ.get("SONGO_EXTERNAL_ARTIFACT_FORCE_FULL_SCAN", "0"), default=False)
    scan_progress_every = max(
        500,
        _as_int(os.environ.get("SONGO_EXTERNAL_ARTIFACT_SCAN_PROGRESS_EVERY", "20000"), 20000),
    )
    move_progress_every = max(
        10,
        _as_int(os.environ.get("SONGO_EXTERNAL_ARTIFACT_MOVE_PROGRESS_EVERY", "100"), 100),
    )
    move_max_items = max(0, _as_int(os.environ.get("SONGO_EXTERNAL_ARTIFACT_MOVE_MAX_ITEMS", "0"), 0))
    step["scan_max_depth"] = int(scan_max_depth)
    step["scan_max_seconds"] = float(scan_max_seconds)
    step["scan_force_full"] = bool(force_full_scan)
    step["scan_progress_every"] = int(scan_progress_every)
    step["move_progress_every"] = int(move_progress_every)
    step["move_max_items"] = int(move_max_items)

    def _is_suspicious_path(path: Path) -> bool:
        name = path.name
        lower_name = str(name or "").lower()
        if name.startswith(".quarantine"):
            return True
        if name.startswith("quarantine"):
            return True
        if name.startswith("model_songo_policy"):
            return True
        if name.startswith("model_songo"):
            return True
        if name.startswith("dataset_"):
            return True
        if name.startswith("dataset"):
            return True
        if name.startswith("labeled_positions"):
            return True
        if name.startswith("songo_policy_value") and name.endswith(".pt"):
            return True
        if name.startswith("songo_policy_value") and name.endswith(".model_card.json"):
            return True
        if name.endswith(".model_card.json"):
            return True
        if name.startswith("events") and name.endswith(".jsonl"):
            return True
        if name.startswith("metrics") and name.endswith(".jsonl"):
            return True
        if name.startswith("latest_dataset_pipeline"):
            return True
        if name.startswith("tournament_progress") and (name.endswith(".json") or name.endswith(".jsonl")):
            return True
        if name.startswith("metadata") and name.endswith(".json"):
            return True
        if name.startswith("mcts"):
            return True
        if name.startswith("minimax"):
            return True
        if name.startswith("_dataset"):
            return True
        if name.startswith(".model") or name.startswith("._model"):
            return True
        if name.startswith(".dataset") or name.startswith("._dataset"):
            return True
        if name in known_file_names:
            return True
        if lower_name.startswith("config") and lower_name.endswith(".yaml"):
            return True
        if lower_name.startswith("run_status") and lower_name.endswith(".json"):
            return True
        if lower_name.startswith("state") and lower_name.endswith(".json"):
            return True
        if lower_name.startswith("dataset_registry") and lower_name.endswith(".json"):
            return True
        if lower_name.endswith("_evaluation_summary.json"):
            return True
        if (
            (
                "dataset_generation_summary" in lower_name
                or "dataset_build_summary" in lower_name
                or "training_summary" in lower_name
                or "evaluation_summary" in lower_name
                or "benchmark_summary" in lower_name
            )
            and lower_name.endswith(".json")
        ):
            return True
        if lower_name.startswith("dataset_benchmatch") and (
            lower_name.endswith(".log")
            or lower_name.endswith(".json")
            or lower_name.endswith(".jsonl")
        ):
            return True
        if lower_name.endswith("_summary.json") and (
            "dataset" in lower_name
            or "train" in lower_name
            or "evaluation" in lower_name
            or "benchmark" in lower_name
        ):
            return True
        if lower_name.endswith(".json") and "summary" in lower_name and (
            "dataset" in lower_name
            or "train" in lower_name
            or "evaluation" in lower_name
            or "benchmark" in lower_name
        ):
            return True
        if name.startswith("bench_models_") and name.endswith(".json"):
            return True
        if lower_name.startswith("build_dataset") and lower_name.endswith(".log"):
            return True
        if "dataset" in lower_name and "metadata" in lower_name and lower_name.endswith(".json"):
            return True
        if ".tmp." in lower_name:
            if (
                lower_name.startswith(".dataset")
                or lower_name.startswith("._dataset")
                or lower_name.startswith(".model")
                or lower_name.startswith("._model")
                or lower_name.startswith(".bench_models")
                or lower_name.startswith(".events")
                or lower_name.startswith(".metrics")
                or lower_name.startswith(".run_status")
                or lower_name.startswith(".state")
                or lower_name.startswith(".tournament_progress")
                or lower_name.startswith(".metadata")
                or lower_name.startswith(".mcts")
                or lower_name.startswith(".minimax")
                or lower_name.startswith(".songo_policy_value")
                or "config.yaml.tmp." in lower_name
                or (
                    ("config" in lower_name or lower_name.startswith(".config") or lower_name.startswith("._config"))
                    and ".yaml.tmp." in lower_name
                )
                or "run_status.json.tmp." in lower_name
                or "state.json.tmp." in lower_name
                or "tournament_progress.latest.json.tmp." in lower_name
                or "dataset_registry.json.tmp." in lower_name
                or "dataset_generation_summary.json.tmp." in lower_name
                or "dataset_build_summary.json.tmp." in lower_name
                or "training_summary.json.tmp." in lower_name
                or "evaluation_summary.json.tmp." in lower_name
                or "benchmark_summary.json.tmp." in lower_name
                or "dataset_benchmatch" in lower_name
                or (
                    ".tmp." in lower_name
                    and (
                        "dataset_generation_summary" in lower_name
                        or "dataset_build_summary" in lower_name
                        or "training_summary" in lower_name
                        or "evaluation_summary" in lower_name
                        or "benchmark_summary" in lower_name
                    )
                )
                or "_dataset_source_metadata.json.tmp" in lower_name
            ):
                return True
        return False

    def _is_interesting_dir_name(name: str) -> bool:
        text = str(name or "").strip().lower()
        if not text:
            return False
        if text.startswith(".quarantine_"):
            return True
        interesting_tokens = (
            "songo",
            "stockfish",
            "runtime",
            "benchmark",
            "dataset",
            "model",
            "train",
            "eval",
            "jobs",
            "logs",
        )
        return any(token in text for token in interesting_tokens)

    def _resolve_available_target(path: Path) -> Path:
        if not path.exists():
            return path
        parent = path.parent
        suffix = path.suffix
        if suffix:
            base_name = path.name[: -len(suffix)]
        else:
            base_name = path.name
        for idx in range(1, 10000):
            if suffix:
                candidate = parent / f"{base_name}__dup_{idx:03d}{suffix}"
            else:
                candidate = parent / f"{base_name}__dup_{idx:03d}"
            if not candidate.exists():
                return candidate
        # Fallback theoretique si collisions massives.
        if suffix:
            return parent / f"{base_name}__dup_{int(now_ts)}{suffix}"
        return parent / f"{base_name}__dup_{int(now_ts)}"

    raw_candidates: list[Path] = []
    scan_started = time.time()
    queue: deque[tuple[Path, int, bool]] = deque()
    queue.append((mydrive_root, 0, False))
    try:
        while queue:
            if (time.time() - scan_started) > scan_max_seconds:
                step["scan_truncated"] = True
                step["errors"].append(
                    {
                        "scan": str(mydrive_root),
                        "warning": (
                            "scan_timeout_reached; "
                            f"max_seconds={scan_max_seconds}. "
                            "Relance avec SONGO_EXTERNAL_ARTIFACT_FORCE_FULL_SCAN=1 "
                            "et/ou un timeout plus eleve si necessaire."
                        ),
                    }
                )
                break
            current_dir, depth, branch_interesting = queue.popleft()
            try:
                iterator = os.scandir(current_dir)
            except Exception as exc:
                step["errors"].append(
                    {"scan": str(current_dir), "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            with iterator as it:
                for entry in it:
                    step["scan_entries"] = int(step["scan_entries"]) + 1
                    scan_entries = int(step["scan_entries"])
                    if scan_entries > 0 and (scan_entries % int(scan_progress_every) == 0):
                        print(
                            (
                                "[storage-cleanup][external-scan] "
                                f"scanned={scan_entries} | candidates={len(raw_candidates)} "
                                f"| queue={len(queue)} | elapsed={int(time.time() - scan_started)}s"
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                    path = Path(entry.path)
                    if _path_within(path, drive_root):
                        continue
                    if _is_suspicious_path(path):
                        raw_candidates.append(path)
                        step["scan_candidates"] = len(raw_candidates)
                        continue
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    if depth >= scan_max_depth:
                        continue
                    name_interesting = _is_interesting_dir_name(entry.name)
                    if force_full_scan:
                        queue.append((path, depth + 1, True))
                        continue
                    if depth == 0:
                        if name_interesting:
                            queue.append((path, depth + 1, True))
                        continue
                    if branch_interesting or name_interesting:
                        queue.append((path, depth + 1, bool(branch_interesting or name_interesting)))
    except Exception as exc:
        step["errors"].append({"scan": str(mydrive_root), "error": f"{type(exc).__name__}: {exc}"})
        return step

    # Si un dossier parent est deja candidat, ne pas traiter ses enfants.
    candidates_sorted = sorted(raw_candidates, key=lambda p: (len(p.parts), str(p)))
    selected: list[Path] = []
    for candidate in candidates_sorted:
        if any(_path_within(candidate, parent) for parent in selected):
            continue
        selected.append(candidate)
    step["selected_candidates"] = len(selected)

    if selected:
        print(
            (
                "[storage-cleanup][external-scan] "
                f"scan_done | scanned={int(step['scan_entries'])} "
                f"| selected={len(selected)} | scan_truncated={bool(step['scan_truncated'])}"
            ),
            file=sys.stderr,
            flush=True,
        )

    processed_count = 0
    for entry in selected:
        if move_max_items > 0 and processed_count >= move_max_items:
            remaining = max(0, len(selected) - processed_count)
            step["move_truncated"] = True
            step["skipped"].append(
                {
                    "src": "<multiple>",
                    "reason": "move_limit_reached",
                    "limit": int(move_max_items),
                    "remaining": int(remaining),
                }
            )
            print(
                (
                    "[storage-cleanup][external-move] "
                    f"limit_reached | limit={move_max_items} | remaining={remaining}"
                ),
                file=sys.stderr,
                flush=True,
            )
            break
        rel_hint = str(entry.relative_to(mydrive_root))
        target = _resolve_available_target(session_root / rel_hint)
        if not apply:
            step["moved"].append({"src": str(entry), "dst": str(target), "mode": "dry_run"})
            processed_count += 1
            if processed_count % int(move_progress_every) == 0:
                print(
                    (
                        "[storage-cleanup][external-move] "
                        f"planned={processed_count}/{len(selected)}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(target))
            step["moved"].append({"src": str(entry), "dst": str(target), "mode": "moved"})
            processed_count += 1
            if processed_count % int(move_progress_every) == 0:
                print(
                    (
                        "[storage-cleanup][external-move] "
                        f"moved={processed_count}/{len(selected)}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            step["errors"].append({"src": str(entry), "error": f"{type(exc).__name__}: {exc}"})
    step["processed_candidates"] = int(processed_count)
    return step


def _cleanup_quarantine_dirs(
    *,
    drive_root: Path,
    apply: bool,
    quarantine_ttl_seconds: float,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "quarantine_ttl_seconds": float(max(0.0, quarantine_ttl_seconds)),
        "removed": [],
        "skipped_recent": 0,
        "errors": [],
    }
    roots = [
        drive_root / "jobs",
        drive_root / "logs",
        drive_root / "runtime_backup",
        drive_root / "runtime_migration",
    ]
    ttl_seconds = float(max(0.0, quarantine_ttl_seconds))
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob(".quarantine_*"):
                if not path.exists():
                    continue
                if not _path_within(path, drive_root):
                    continue
                age_seconds = _path_age_seconds(path, now_epoch=now_epoch)
                if age_seconds < ttl_seconds:
                    step["skipped_recent"] = int(step["skipped_recent"]) + 1
                    continue
                removed = _remove_tree(path, apply=apply) if path.is_dir() else _remove_file(path, apply=apply)
                if removed:
                    step["removed"].append(str(path))
        except Exception as exc:
            step["errors"].append(f"quarantine_scan_failed:{type(exc).__name__}:{exc}")
    return step


def _cleanup_duplicate_source_metadata(
    *,
    dataset_registry: dict[str, Any],
    drive_root: Path,
    apply: bool,
    raw_metadata_ttl_seconds: float,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    ttl_seconds = float(max(0.0, raw_metadata_ttl_seconds))
    step: dict[str, Any] = {
        "raw_metadata_ttl_seconds": ttl_seconds,
        "removed_raw_metadata": [],
        "removed_raw_metadata_due_to_ttl": [],
        "skipped_missing": 0,
        "skipped_recent_ttl": [],
        "skipped_mismatch": [],
        "errors": [],
    }
    dataset_sources = dataset_registry.get("dataset_sources", [])
    if not isinstance(dataset_sources, list):
        return step

    for entry in dataset_sources:
        if not isinstance(entry, dict):
            continue
        dataset_source_id = str(entry.get("dataset_source_id", "")).strip()
        raw_dir_text = str(entry.get("raw_dir", "")).strip()
        sampled_dir_text = str(entry.get("sampled_dir", "")).strip()
        if not raw_dir_text or not sampled_dir_text:
            continue
        source_status = str(entry.get("source_status", "")).strip().lower()
        raw_dir = Path(raw_dir_text)
        sampled_dir = Path(sampled_dir_text)
        raw_meta = raw_dir / "_dataset_source_metadata.json"
        sampled_meta = sampled_dir / "_dataset_source_metadata.json"

        if not raw_meta.exists() or not sampled_meta.exists():
            step["skipped_missing"] = int(step["skipped_missing"]) + 1
            continue
        if not _path_within(raw_meta, drive_root) or not _path_within(sampled_meta, drive_root):
            continue

        same_content = False
        try:
            raw_payload = _load_json_dict(raw_meta, default={})
            sampled_payload = _load_json_dict(sampled_meta, default={})
            if raw_payload and sampled_payload:
                same_content = raw_payload == sampled_payload
            else:
                same_content = raw_meta.read_text(encoding="utf-8") == sampled_meta.read_text(encoding="utf-8")
        except Exception as exc:
            step["errors"].append(f"metadata_compare_failed:{dataset_source_id}:{type(exc).__name__}:{exc}")
            continue

        if not same_content:
            if source_status == "completed":
                raw_age_seconds = _path_age_seconds(raw_meta, now_epoch=now_epoch)
                if raw_age_seconds >= ttl_seconds:
                    if _remove_file(raw_meta, apply=apply):
                        step["removed_raw_metadata"].append(str(raw_meta))
                        step["removed_raw_metadata_due_to_ttl"].append(str(raw_meta))
                    continue
                step["skipped_recent_ttl"].append(
                    {
                        "dataset_source_id": dataset_source_id,
                        "raw_metadata": str(raw_meta),
                        "sampled_metadata": str(sampled_meta),
                        "age_seconds": float(raw_age_seconds),
                    }
                )
                continue
            step["skipped_mismatch"].append(
                {
                    "dataset_source_id": dataset_source_id,
                    "raw_metadata": str(raw_meta),
                    "sampled_metadata": str(sampled_meta),
                }
            )
            continue

        if _remove_file(raw_meta, apply=apply):
            step["removed_raw_metadata"].append(str(raw_meta))
    return step


def _collect_protected_job_ids(config: dict[str, Any], latest_manifest_payload: dict[str, Any] | None = None) -> set[str]:
    protected: set[str] = set()
    spaces = [
        config.get("job", {}),
        config.get("dataset_generation", {}),
        config.get("dataset_build", {}),
        config.get("dataset_merge_final", {}),
        config.get("train", {}),
        config.get("evaluation", {}),
        config.get("benchmark", {}),
    ]
    for space in spaces:
        if not isinstance(space, dict):
            continue
        for key, value in space.items():
            if "job_id" not in str(key):
                continue
            job_id = str(value or "").strip()
            if job_id and job_id.lower() not in {"auto", "<auto>"}:
                protected.add(job_id)
    manifest = latest_manifest_payload if isinstance(latest_manifest_payload, dict) else {}
    for key, value in manifest.items():
        if "job_id" not in str(key):
            continue
        if isinstance(value, list):
            for item in value:
                job_id = str(item or "").strip()
                if job_id:
                    protected.add(job_id)
            continue
        job_id = str(value or "").strip()
        if job_id:
            protected.add(job_id)
    return protected


def _cleanup_pipeline_manifests(
    *,
    pipeline_root: Path,
    apply: bool,
    pipeline_manifest_ttl_seconds: float,
    pipeline_manifest_keep_recent: int,
    pipeline_manifest_hard_max_age_seconds: float = 0.0,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    ttl_seconds = float(max(0.0, pipeline_manifest_ttl_seconds))
    keep_recent = max(0, int(pipeline_manifest_keep_recent))
    hard_max_seconds = float(max(0.0, pipeline_manifest_hard_max_age_seconds))
    step: dict[str, Any] = {
        "pipeline_root": str(pipeline_root),
        "pipeline_manifest_ttl_seconds": ttl_seconds,
        "pipeline_manifest_keep_recent": keep_recent,
        "pipeline_manifest_hard_max_age_seconds": hard_max_seconds,
        "removed": [],
        "removed_hard_max_age": 0,
        "skipped_recent_ttl": 0,
        "skipped_keep_recent": 0,
        "errors": [],
    }
    if not pipeline_root.exists():
        return step
    files: list[Path] = []
    try:
        files.extend(sorted(pipeline_root.glob("dataset_pipeline_*.json")))
        files.extend(sorted(pipeline_root.glob("latest_dataset_pipeline_*.json")))
        files = [item for item in files if item.is_file()]
    except Exception as exc:
        step["errors"].append(f"pipeline_manifest_scan_failed:{type(exc).__name__}:{exc}")
        return step

    files_sorted = sorted(files, key=lambda item: float(item.stat().st_mtime), reverse=True)
    kept = 0
    for file_path in files_sorted:
        age_seconds = _path_age_seconds(file_path, now_epoch=now_epoch)
        if kept < keep_recent:
            if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                if _remove_file(file_path, apply=apply):
                    step["removed"].append(str(file_path))
                    step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                continue
            kept += 1
            step["skipped_keep_recent"] = int(step["skipped_keep_recent"]) + 1
            continue
        if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
            if _remove_file(file_path, apply=apply):
                step["removed"].append(str(file_path))
                step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
            continue
        if age_seconds < ttl_seconds:
            step["skipped_recent_ttl"] = int(step["skipped_recent_ttl"]) + 1
            continue
        if _remove_file(file_path, apply=apply):
            step["removed"].append(str(file_path))
    return step


def _canonicalize_duplicate_basename(file_name: str) -> str:
    path = Path(str(file_name or "").strip())
    stem = str(path.stem or "").strip()
    suffix = str(path.suffix or "").strip().lower()
    duplicate_match = re.fullmatch(r"(.+)\s\((\d+)\)", stem)
    if duplicate_match:
        stem = str(duplicate_match.group(1) or "").strip()
    return (stem + suffix).strip().lower()


def _is_drive_root_operational_artifact_name(file_name: str) -> bool:
    name = _canonicalize_duplicate_basename(file_name)
    if not name:
        return False
    if name in {
        "config.yaml",
        "run_status.json",
        "state.json",
        "dataset_registry.json",
        "dataset_generation_summary.json",
        "dataset_build_summary.json",
        "training_summary.json",
        "evaluation_summary.json",
        "benchmark_summary.json",
        "latest_dataset_pipeline.json",
        "latest_dataset_pipeline",
        "tournament_progress.latest.json",
        "_dataset_source_metadata.json",
    }:
        return True
    if name.startswith("state") and name.endswith(".json"):
        return True
    if name.startswith("run_status") and name.endswith(".json"):
        return True
    if name.startswith("config") and name.endswith(".yaml"):
        return True
    if name.startswith("dataset_registry") and name.endswith(".json"):
        return True
    if name.startswith("dataset_benchmatch") and (
        name.endswith(".log") or name.endswith(".json") or name.endswith(".jsonl")
    ):
        return True
    if name.startswith("bench_models") and name.endswith(".json"):
        return True
    if name.startswith("latest_dataset_pipeline"):
        return True
    if name.startswith("tournament_progress") and (
        name.endswith(".json") or name.endswith(".jsonl")
    ):
        return True
    if name.startswith("events") and name.endswith(".jsonl"):
        return True
    if name.startswith("metrics") and name.endswith(".jsonl"):
        return True
    if name.startswith("mcts") and (name.endswith(".json") or name.endswith(".jsonl")):
        return True
    if name.startswith("minimax") and (name.endswith(".json") or name.endswith(".jsonl")):
        return True
    if name.startswith("metadata") and name.endswith(".json"):
        return True
    if name.endswith("_evaluation_summary.json"):
        return True
    if (
        name.endswith(".json")
        and (
            "dataset_generation_summary" in name
            or "dataset_build_summary" in name
            or "training_summary" in name
            or "evaluation_summary" in name
            or "benchmark_summary" in name
        )
    ):
        return True
    if ".tmp." in str(file_name or "").lower():
        return True
    return False


def _drive_root_operational_artifact_group_key(file_name: str) -> str:
    name = _canonicalize_duplicate_basename(file_name)
    if not name:
        return ""
    if name.startswith("state") and name.endswith(".json"):
        return "state.json"
    if name.startswith("run_status") and name.endswith(".json"):
        return "run_status.json"
    if name.startswith("config") and name.endswith(".yaml"):
        return "config.yaml"
    if name.startswith("dataset_registry") and name.endswith(".json"):
        return "dataset_registry.json"
    if name.startswith("dataset_benchmatch"):
        return "dataset_benchmatch"
    if name.startswith("bench_models") and name.endswith(".json"):
        return "bench_models.json"
    if name.startswith("latest_dataset_pipeline"):
        return "latest_dataset_pipeline.json"
    if name.startswith("tournament_progress"):
        return "tournament_progress.latest.json"
    if name.startswith("events") and name.endswith(".jsonl"):
        return "events.jsonl"
    if name.startswith("metrics") and name.endswith(".jsonl"):
        return "metrics.jsonl"
    if name.startswith("mcts"):
        return "mcts.json_or_jsonl"
    if name.startswith("minimax"):
        return "minimax.json_or_jsonl"
    if name.startswith("metadata") and name.endswith(".json"):
        return "metadata.json"
    if name.endswith("_evaluation_summary.json"):
        return "evaluation_summary.json"
    if "dataset_generation_summary" in name and name.endswith(".json"):
        return "dataset_generation_summary.json"
    if "dataset_build_summary" in name and name.endswith(".json"):
        return "dataset_build_summary.json"
    if "training_summary" in name and name.endswith(".json"):
        return "training_summary.json"
    if "evaluation_summary" in name and name.endswith(".json"):
        return "evaluation_summary.json"
    if "benchmark_summary" in name and name.endswith(".json"):
        return "benchmark_summary.json"
    if ".tmp." in name:
        return f"tmp::{name.split('.tmp.', 1)[0]}"
    return name


def _cleanup_drive_root_operational_artifacts(
    *,
    drive_root: Path,
    apply: bool,
    artifact_ttl_seconds: float,
    artifact_keep_recent_per_key: int,
    artifact_hard_max_age_seconds: float = 0.0,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    ttl_seconds = float(max(0.0, artifact_ttl_seconds))
    keep_recent = max(0, int(artifact_keep_recent_per_key))
    hard_max_seconds = float(max(0.0, artifact_hard_max_age_seconds))
    step: dict[str, Any] = {
        "drive_root": str(drive_root),
        "artifact_ttl_seconds": ttl_seconds,
        "artifact_keep_recent_per_key": keep_recent,
        "artifact_hard_max_age_seconds": hard_max_seconds,
        "scanned_files": 0,
        "candidate_files": 0,
        "groups_count": 0,
        "removed": [],
        "removed_hard_max_age": 0,
        "skipped_keep_recent": 0,
        "skipped_recent_ttl": 0,
        "errors": [],
    }
    if not drive_root.exists():
        return step
    groups: dict[str, list[Path]] = {}
    try:
        entries = sorted(drive_root.iterdir())
    except Exception as exc:
        step["errors"].append(f"drive_root_scan_failed:{type(exc).__name__}:{exc}")
        return step

    for path in entries:
        if not path.is_file():
            continue
        step["scanned_files"] = int(step["scanned_files"]) + 1
        if not _is_drive_root_operational_artifact_name(path.name):
            continue
        group_key = _drive_root_operational_artifact_group_key(path.name)
        if not group_key:
            continue
        groups.setdefault(group_key, []).append(path)
        step["candidate_files"] = int(step["candidate_files"]) + 1

    step["groups_count"] = int(len(groups))
    for group_key, paths_for_group in groups.items():
        def _path_sort_key(path: Path) -> tuple[int, float]:
            stem = str(path.stem or "").strip()
            duplicate_named_copy = bool(re.fullmatch(r"(.+)\s\((\d+)\)", stem))
            try:
                mtime = float(path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            # Priorite au nom canonique (sans "(n)"), puis date.
            return (1 if duplicate_named_copy else 0, -mtime)

        sorted_paths = sorted(paths_for_group, key=_path_sort_key)
        for idx, file_path in enumerate(sorted_paths):
            file_name_lower = str(file_path.name or "").lower()
            group_keep_recent = 0 if ".tmp." in file_name_lower else keep_recent
            age_seconds = _path_age_seconds(file_path, now_epoch=now_epoch)
            if idx < group_keep_recent:
                if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                    if _remove_file(file_path, apply=apply):
                        step["removed"].append(
                            {
                                "path": str(file_path),
                                "group_key": str(group_key),
                                "age_seconds": float(age_seconds),
                                "reason": "hard_max_age",
                            }
                        )
                        step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                    continue
                step["skipped_keep_recent"] = int(step["skipped_keep_recent"]) + 1
                continue
            if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                if _remove_file(file_path, apply=apply):
                    step["removed"].append(
                        {
                            "path": str(file_path),
                            "group_key": str(group_key),
                            "age_seconds": float(age_seconds),
                            "reason": "hard_max_age",
                        }
                    )
                    step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                continue
            if age_seconds < ttl_seconds:
                step["skipped_recent_ttl"] = int(step["skipped_recent_ttl"]) + 1
                continue
            if _remove_file(file_path, apply=apply):
                step["removed"].append(
                    {
                        "path": str(file_path),
                        "group_key": str(group_key),
                        "age_seconds": float(age_seconds),
                    }
                )
    return step


def _cleanup_recovered_external_sessions(
    *,
    drive_root: Path,
    apply: bool,
    recovered_external_ttl_seconds: float,
    recovered_external_keep_recent_sessions: int,
    recovered_external_hard_max_age_seconds: float = 0.0,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    ttl_seconds = float(max(0.0, recovered_external_ttl_seconds))
    keep_recent = max(0, int(recovered_external_keep_recent_sessions))
    hard_max_seconds = float(max(0.0, recovered_external_hard_max_age_seconds))
    recovered_root = drive_root / "runtime_migration" / "recovered_external"
    step: dict[str, Any] = {
        "recovered_root": str(recovered_root),
        "recovered_external_ttl_seconds": ttl_seconds,
        "recovered_external_keep_recent_sessions": keep_recent,
        "recovered_external_hard_max_age_seconds": hard_max_seconds,
        "sessions_scanned": 0,
        "removed_sessions": [],
        "removed_hard_max_age": 0,
        "skipped_keep_recent": 0,
        "skipped_recent_ttl": 0,
        "removed_empty_dirs": [],
        "errors": [],
    }
    if not recovered_root.exists():
        return step

    session_dirs: list[Path] = []
    try:
        for path in recovered_root.rglob("*"):
            if not path.is_dir():
                continue
            if re.fullmatch(r"\d{9,}", str(path.name or "").strip()):
                session_dirs.append(path)
    except Exception as exc:
        step["errors"].append(f"recovered_external_scan_failed:{type(exc).__name__}:{exc}")
        return step

    step["sessions_scanned"] = int(len(session_dirs))
    sorted_sessions = sorted(session_dirs, key=lambda item: float(item.stat().st_mtime), reverse=True)
    for idx, session_dir in enumerate(sorted_sessions):
        age_seconds = _path_age_seconds(session_dir, now_epoch=now_epoch)
        if idx < keep_recent:
            if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                if _remove_tree(session_dir, apply=apply):
                    step["removed_sessions"].append(
                        {
                            "path": str(session_dir),
                            "age_seconds": float(age_seconds),
                            "reason": "hard_max_age",
                        }
                    )
                    step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                continue
            step["skipped_keep_recent"] = int(step["skipped_keep_recent"]) + 1
            continue
        if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
            if _remove_tree(session_dir, apply=apply):
                step["removed_sessions"].append(
                    {
                        "path": str(session_dir),
                        "age_seconds": float(age_seconds),
                        "reason": "hard_max_age",
                    }
                )
                step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
            continue
        if age_seconds < ttl_seconds:
            step["skipped_recent_ttl"] = int(step["skipped_recent_ttl"]) + 1
            continue
        if _remove_tree(session_dir, apply=apply):
            step["removed_sessions"].append(
                {
                    "path": str(session_dir),
                    "age_seconds": float(age_seconds),
                }
            )

    try:
        empty_dirs = sorted(
            [path for path in recovered_root.rglob("*") if path.is_dir()],
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for path in empty_dirs:
            try:
                if any(path.iterdir()):
                    continue
                if path == recovered_root:
                    continue
                if _remove_tree(path, apply=apply):
                    step["removed_empty_dirs"].append(str(path))
            except Exception:
                continue
    except Exception as exc:
        step["errors"].append(f"recovered_external_empty_cleanup_failed:{type(exc).__name__}:{exc}")
    return step


def _cleanup_completed_job_dirs(
    *,
    job_roots: list[Path],
    apply: bool,
    protected_job_ids: set[str],
    job_dir_ttl_seconds: float,
    job_dir_keep_recent_per_run_type: int,
    job_dir_hard_max_age_seconds: float = 0.0,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    ttl_seconds = float(max(0.0, job_dir_ttl_seconds))
    keep_recent = max(0, int(job_dir_keep_recent_per_run_type))
    hard_max_seconds = float(max(0.0, job_dir_hard_max_age_seconds))
    terminal_statuses = {"completed", "failed", "cancelled"}
    step: dict[str, Any] = {
        "job_roots": [str(root) for root in job_roots],
        "job_dir_ttl_seconds": ttl_seconds,
        "job_dir_keep_recent_per_run_type": keep_recent,
        "job_dir_hard_max_age_seconds": hard_max_seconds,
        "protected_job_ids": sorted(str(item) for item in protected_job_ids if str(item).strip()),
        "removed": [],
        "removed_hard_max_age": 0,
        "skipped_protected": 0,
        "skipped_active": 0,
        "skipped_recent_ttl": 0,
        "skipped_keep_recent": 0,
        "scanned": 0,
        "errors": [],
    }
    protected = {str(item).strip() for item in protected_job_ids if str(item).strip()}
    candidates_by_run_type: dict[str, list[dict[str, Any]]] = {}
    for root in job_roots:
        if not root.exists():
            continue
        try:
            job_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
        except Exception as exc:
            step["errors"].append(f"job_root_scan_failed:{root}:{type(exc).__name__}:{exc}")
            continue
        for job_dir in job_dirs:
            step["scanned"] = int(step["scanned"]) + 1
            job_id = str(job_dir.name).strip()
            if job_id in protected:
                step["skipped_protected"] = int(step["skipped_protected"]) + 1
                continue
            active, _ = is_job_active(job_dir, active_updated_max_age_seconds=600.0)
            if active:
                step["skipped_active"] = int(step["skipped_active"]) + 1
                continue
            status_payload = _load_json_dict(job_dir / "run_status.json", default={})
            status = str(status_payload.get("status", "")).strip().lower()
            if status not in terminal_statuses:
                continue
            run_type = str(status_payload.get("run_type", "")).strip().lower() or "unknown"
            updated_epoch = _parse_iso_to_epoch(status_payload.get("updated_at"))
            if updated_epoch <= 0.0:
                try:
                    updated_epoch = float(job_dir.stat().st_mtime)
                except Exception:
                    updated_epoch = 0.0
            age_seconds = _job_dir_age_seconds(job_dir, now_epoch=now_epoch)
            candidates_by_run_type.setdefault(run_type, []).append(
                {
                    "job_id": job_id,
                    "job_dir": job_dir,
                    "updated_epoch": float(updated_epoch),
                    "age_seconds": float(age_seconds),
                    "root": str(root),
                    "run_type": run_type,
                }
            )

    for run_type, items in candidates_by_run_type.items():
        sorted_items = sorted(items, key=lambda item: float(item.get("updated_epoch", 0.0)), reverse=True)
        for idx, item in enumerate(sorted_items):
            age_seconds = float(item.get("age_seconds", 0.0))
            job_dir = Path(str(item.get("job_dir")))
            if idx < keep_recent:
                if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                    if _remove_tree(job_dir, apply=apply):
                        step["removed"].append(
                            {
                                "job_id": str(item.get("job_id", "")),
                                "run_type": str(run_type),
                                "job_dir": str(job_dir),
                                "age_seconds": float(age_seconds),
                                "reason": "hard_max_age",
                            }
                        )
                        step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                    continue
                step["skipped_keep_recent"] = int(step["skipped_keep_recent"]) + 1
                continue
            if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                if _remove_tree(job_dir, apply=apply):
                    step["removed"].append(
                        {
                            "job_id": str(item.get("job_id", "")),
                            "run_type": str(run_type),
                            "job_dir": str(job_dir),
                            "age_seconds": float(age_seconds),
                            "reason": "hard_max_age",
                        }
                    )
                    step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                continue
            if age_seconds < ttl_seconds:
                step["skipped_recent_ttl"] = int(step["skipped_recent_ttl"]) + 1
                continue
            if _remove_tree(job_dir, apply=apply):
                step["removed"].append(
                    {
                        "job_id": str(item.get("job_id", "")),
                        "run_type": str(run_type),
                        "job_dir": str(job_dir),
                        "age_seconds": float(age_seconds),
                    }
                )
    return step


def _collect_protected_global_target_ids(config: dict[str, Any]) -> set[str]:
    protected: set[str] = set()
    dataset_generation_cfg = config.get("dataset_generation", {})
    dataset_build_cfg = config.get("dataset_build", {})
    if not isinstance(dataset_generation_cfg, dict):
        dataset_generation_cfg = {}
    if not isinstance(dataset_build_cfg, dict):
        dataset_build_cfg = {}

    def _add_target_id(value: Any) -> None:
        target_id = str(value or "").strip()
        if target_id:
            protected.add(target_id)

    _add_target_id(dataset_generation_cfg.get("global_target_id"))
    _add_target_id(dataset_build_cfg.get("global_target_id"))
    generate_progress_path = str(dataset_generation_cfg.get("global_progress_path", "")).strip()
    if generate_progress_path:
        _add_target_id(Path(generate_progress_path).stem)
    build_progress_path = str(dataset_build_cfg.get("global_target_progress_path", "")).strip()
    if build_progress_path:
        _add_target_id(Path(build_progress_path).stem)

    # Fallback utile si global_target_id n'est pas explicitement configure.
    if not protected and _as_bool(dataset_generation_cfg.get("global_target_enabled", False), default=False):
        _add_target_id(dataset_generation_cfg.get("dataset_source_id"))
    return protected


def _cleanup_global_progress_mirrors(
    *,
    data_root: Path,
    apply: bool,
    protected_target_ids: set[str],
    global_progress_ttl_seconds: float,
    global_progress_keep_recent: int,
    global_progress_hard_max_age_seconds: float = 0.0,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    ttl_seconds = float(max(0.0, global_progress_ttl_seconds))
    keep_recent = max(0, int(global_progress_keep_recent))
    hard_max_seconds = float(max(0.0, global_progress_hard_max_age_seconds))
    progress_root = data_root / "global_generation_progress"
    protected = {str(item).strip() for item in protected_target_ids if str(item).strip()}
    step: dict[str, Any] = {
        "progress_root": str(progress_root),
        "global_progress_ttl_seconds": ttl_seconds,
        "global_progress_keep_recent": keep_recent,
        "global_progress_hard_max_age_seconds": hard_max_seconds,
        "protected_target_ids": sorted(protected),
        "removed_progress_files": [],
        "removed_workers_dirs": [],
        "removed_worker_snapshots": [],
        "removed_hard_max_age": 0,
        "skipped_protected": 0,
        "skipped_recent_ttl": 0,
        "skipped_keep_recent": 0,
        "workers_dirs_scanned": 0,
        "errors": [],
    }
    if not progress_root.exists():
        return step

    try:
        progress_files = sorted(
            [path for path in progress_root.glob("*.json") if path.is_file()],
            key=lambda path: float(path.stat().st_mtime),
            reverse=True,
        )
    except Exception as exc:
        step["errors"].append(f"progress_scan_failed:{type(exc).__name__}:{exc}")
        progress_files = []

    kept_recent = 0
    removed_target_ids: set[str] = set()
    for progress_file in progress_files:
        target_id = str(progress_file.stem).strip()
        if target_id in protected:
            step["skipped_protected"] = int(step["skipped_protected"]) + 1
            continue
        age_seconds = _path_age_seconds(progress_file, now_epoch=now_epoch)
        if kept_recent < keep_recent:
            if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                if _remove_file(progress_file, apply=apply):
                    step["removed_progress_files"].append(str(progress_file))
                    step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                    removed_target_ids.add(target_id)
                continue
            kept_recent += 1
            step["skipped_keep_recent"] = int(step["skipped_keep_recent"]) + 1
            continue
        if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
            if _remove_file(progress_file, apply=apply):
                step["removed_progress_files"].append(str(progress_file))
                step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                removed_target_ids.add(target_id)
            continue
        if age_seconds < ttl_seconds:
            step["skipped_recent_ttl"] = int(step["skipped_recent_ttl"]) + 1
            continue
        if _remove_file(progress_file, apply=apply):
            step["removed_progress_files"].append(str(progress_file))
            removed_target_ids.add(target_id)

    workers_dirs = sorted([path for path in progress_root.glob("*.workers") if path.is_dir()])
    for workers_dir in workers_dirs:
        step["workers_dirs_scanned"] = int(step["workers_dirs_scanned"]) + 1
        name_text = str(workers_dir.name or "").strip()
        if name_text.endswith(".workers"):
            target_id = name_text[: -len(".workers")]
        else:
            target_id = name_text

        progress_file = progress_root / f"{target_id}.json"
        if target_id in protected:
            # Pour les cibles protegees, on purge seulement les snapshots devenus vieux.
            for snapshot_path in workers_dir.glob("*.json"):
                age_seconds = _path_age_seconds(snapshot_path, now_epoch=now_epoch)
                if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                    if _remove_file(snapshot_path, apply=apply):
                        step["removed_worker_snapshots"].append(str(snapshot_path))
                        step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                    continue
                if age_seconds < ttl_seconds:
                    continue
                if _remove_file(snapshot_path, apply=apply):
                    step["removed_worker_snapshots"].append(str(snapshot_path))
            continue

        if target_id in removed_target_ids:
            if _remove_tree(workers_dir, apply=apply):
                step["removed_workers_dirs"].append(str(workers_dir))
            continue

        for snapshot_path in workers_dir.glob("*.json"):
            age_seconds = _path_age_seconds(snapshot_path, now_epoch=now_epoch)
            if hard_max_seconds > 0.0 and age_seconds >= hard_max_seconds:
                if _remove_file(snapshot_path, apply=apply):
                    step["removed_worker_snapshots"].append(str(snapshot_path))
                    step["removed_hard_max_age"] = int(step["removed_hard_max_age"]) + 1
                continue
            if age_seconds < ttl_seconds:
                continue
            if _remove_file(snapshot_path, apply=apply):
                step["removed_worker_snapshots"].append(str(snapshot_path))

        try:
            dir_has_entries = any(workers_dir.iterdir())
        except Exception:
            dir_has_entries = True
        if dir_has_entries:
            continue
        if progress_file.exists():
            continue
        dir_age_seconds = _path_age_seconds(workers_dir, now_epoch=now_epoch)
        if dir_age_seconds < ttl_seconds:
            continue
        if _remove_tree(workers_dir, apply=apply):
            step["removed_workers_dirs"].append(str(workers_dir))

    return step


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
    cleanup_retention: bool = False,
    cleanup_external_artifacts: bool = False,
    cleanup_quarantine_dirs: bool = False,
    cleanup_duplicate_source_metadata: bool = False,
    cleanup_global_progress_mirrors: bool = False,
    cleanup_pipeline_manifests: bool = False,
    cleanup_completed_job_dirs: bool = False,
    keep_model_ids: list[str] | None = None,
    keep_top_models: int = 1,
    keep_dataset_ids: list[str] | None = None,
    allow_purge_without_manifest: bool = False,
    drive_raw_cleanup_include_inactive_partial: bool = False,
    drive_raw_cleanup_inactive_min_age_seconds: float = 0.0,
    retention_job_stream_ttl_seconds: float = 72.0 * 3600.0,
    retention_quarantine_ttl_seconds: float = 72.0 * 3600.0,
    retention_benchmark_report_ttl_seconds: float = 45.0 * 86400.0,
    retention_benchmark_keep_recent: int = 60,
    retention_checkpoint_ttl_seconds: float = 14.0 * 86400.0,
    retention_checkpoint_keep_recent_per_model: int = 2,
    retention_global_progress_ttl_seconds: float = 14.0 * 86400.0,
    retention_global_progress_keep_recent: int = 3,
    retention_global_progress_hard_max_age_seconds: float = 30.0 * 86400.0,
    retention_pipeline_manifest_ttl_seconds: float = 14.0 * 86400.0,
    retention_pipeline_manifest_keep_recent: int = 60,
    retention_pipeline_manifest_hard_max_age_seconds: float = 30.0 * 86400.0,
    retention_job_dir_ttl_seconds: float = 14.0 * 86400.0,
    retention_job_dir_keep_recent_per_run_type: int = 8,
    retention_job_dir_hard_max_age_seconds: float = 30.0 * 86400.0,
    retention_source_metadata_raw_ttl_seconds: float = 24.0 * 3600.0,
    retention_drive_root_artifact_ttl_seconds: float = 24.0 * 3600.0,
    retention_drive_root_artifact_keep_recent_per_key: int = 1,
    retention_drive_root_artifact_hard_max_age_seconds: float = 30.0 * 86400.0,
    retention_recovered_external_ttl_seconds: float = 7.0 * 86400.0,
    retention_recovered_external_keep_recent_sessions: int = 2,
    retention_recovered_external_hard_max_age_seconds: float = 30.0 * 86400.0,
    retention_benchmark_report_hard_max_age_seconds: float = 60.0 * 86400.0,
    retention_checkpoint_hard_max_age_seconds: float = 30.0 * 86400.0,
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
        "cleanup_retention": bool(cleanup_retention),
        "cleanup_external_artifacts": bool(cleanup_external_artifacts),
        "cleanup_quarantine_dirs": bool(cleanup_quarantine_dirs),
        "cleanup_duplicate_source_metadata": bool(cleanup_duplicate_source_metadata),
        "cleanup_global_progress_mirrors": bool(cleanup_global_progress_mirrors),
        "cleanup_pipeline_manifests": bool(cleanup_pipeline_manifests),
        "cleanup_completed_job_dirs": bool(cleanup_completed_job_dirs),
        "drive_raw_cleanup_include_inactive_partial": bool(drive_raw_cleanup_include_inactive_partial),
        "drive_raw_cleanup_inactive_min_age_seconds": float(max(0.0, drive_raw_cleanup_inactive_min_age_seconds)),
        "steps": {},
    }

    drive_root = paths.drive_root
    jobs_root = paths.jobs_root
    logs_root = paths.logs_root
    data_root = paths.data_root
    models_root = paths.models_root
    retention_cfg = _resolve_retention_cfg(config)
    now_epoch = time.time()
    job_stream_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("job_stream_ttl_seconds", retention_job_stream_ttl_seconds),
            retention_job_stream_ttl_seconds,
        ),
    )
    quarantine_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("quarantine_ttl_seconds", retention_quarantine_ttl_seconds),
            retention_quarantine_ttl_seconds,
        ),
    )
    benchmark_report_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("benchmark_report_ttl_seconds", retention_benchmark_report_ttl_seconds),
            retention_benchmark_report_ttl_seconds,
        ),
    )
    benchmark_keep_recent = max(
        0,
        _as_int(retention_cfg.get("benchmark_keep_recent", retention_benchmark_keep_recent), retention_benchmark_keep_recent),
    )
    checkpoint_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("checkpoint_ttl_seconds", retention_checkpoint_ttl_seconds),
            retention_checkpoint_ttl_seconds,
        ),
    )
    checkpoint_keep_recent_per_model = max(
        0,
        _as_int(
            retention_cfg.get("checkpoint_keep_recent_per_model", retention_checkpoint_keep_recent_per_model),
            retention_checkpoint_keep_recent_per_model,
        ),
    )
    global_progress_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("global_progress_ttl_seconds", retention_global_progress_ttl_seconds),
            retention_global_progress_ttl_seconds,
        ),
    )
    global_progress_keep_recent = max(
        0,
        _as_int(
            retention_cfg.get("global_progress_keep_recent", retention_global_progress_keep_recent),
            retention_global_progress_keep_recent,
        ),
    )
    global_progress_hard_max_age_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("global_progress_hard_max_age_seconds", retention_global_progress_hard_max_age_seconds),
            retention_global_progress_hard_max_age_seconds,
        ),
    )
    pipeline_manifest_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("pipeline_manifest_ttl_seconds", retention_pipeline_manifest_ttl_seconds),
            retention_pipeline_manifest_ttl_seconds,
        ),
    )
    pipeline_manifest_keep_recent = max(
        0,
        _as_int(
            retention_cfg.get("pipeline_manifest_keep_recent", retention_pipeline_manifest_keep_recent),
            retention_pipeline_manifest_keep_recent,
        ),
    )
    pipeline_manifest_hard_max_age_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get(
                "pipeline_manifest_hard_max_age_seconds", retention_pipeline_manifest_hard_max_age_seconds
            ),
            retention_pipeline_manifest_hard_max_age_seconds,
        ),
    )
    job_dir_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("job_dir_ttl_seconds", retention_job_dir_ttl_seconds),
            retention_job_dir_ttl_seconds,
        ),
    )
    job_dir_keep_recent_per_run_type = max(
        0,
        _as_int(
            retention_cfg.get("job_dir_keep_recent_per_run_type", retention_job_dir_keep_recent_per_run_type),
            retention_job_dir_keep_recent_per_run_type,
        ),
    )
    job_dir_hard_max_age_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("job_dir_hard_max_age_seconds", retention_job_dir_hard_max_age_seconds),
            retention_job_dir_hard_max_age_seconds,
        ),
    )
    source_metadata_raw_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("source_metadata_raw_ttl_seconds", retention_source_metadata_raw_ttl_seconds),
            retention_source_metadata_raw_ttl_seconds,
        ),
    )
    drive_root_artifact_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("drive_root_artifact_ttl_seconds", retention_drive_root_artifact_ttl_seconds),
            retention_drive_root_artifact_ttl_seconds,
        ),
    )
    drive_root_artifact_keep_recent_per_key = max(
        0,
        _as_int(
            retention_cfg.get(
                "drive_root_artifact_keep_recent_per_key",
                retention_drive_root_artifact_keep_recent_per_key,
            ),
            retention_drive_root_artifact_keep_recent_per_key,
        ),
    )
    drive_root_artifact_hard_max_age_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get(
                "drive_root_artifact_hard_max_age_seconds",
                retention_drive_root_artifact_hard_max_age_seconds,
            ),
            retention_drive_root_artifact_hard_max_age_seconds,
        ),
    )
    recovered_external_ttl_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("recovered_external_ttl_seconds", retention_recovered_external_ttl_seconds),
            retention_recovered_external_ttl_seconds,
        ),
    )
    recovered_external_keep_recent_sessions = max(
        0,
        _as_int(
            retention_cfg.get(
                "recovered_external_keep_recent_sessions",
                retention_recovered_external_keep_recent_sessions,
            ),
            retention_recovered_external_keep_recent_sessions,
        ),
    )
    recovered_external_hard_max_age_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get(
                "recovered_external_hard_max_age_seconds",
                retention_recovered_external_hard_max_age_seconds,
            ),
            retention_recovered_external_hard_max_age_seconds,
        ),
    )
    benchmark_report_hard_max_age_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("benchmark_report_hard_max_age_seconds", retention_benchmark_report_hard_max_age_seconds),
            retention_benchmark_report_hard_max_age_seconds,
        ),
    )
    checkpoint_hard_max_age_seconds = max(
        0.0,
        _as_float(
            retention_cfg.get("checkpoint_hard_max_age_seconds", retention_checkpoint_hard_max_age_seconds),
            retention_checkpoint_hard_max_age_seconds,
        ),
    )

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
            quarantine_root=drive_root / "runtime_migration" / "quarantine",
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
            "jobs_skipped_recent": 0,
            "job_stream_ttl_seconds": float(job_stream_ttl_seconds),
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
                age_seconds = _job_dir_age_seconds(job_dir, now_epoch=now_epoch)
                if age_seconds < float(job_stream_ttl_seconds):
                    step["jobs_skipped_recent"] = int(step["jobs_skipped_recent"]) + 1
                    continue
                events_path = job_dir / "events.jsonl"
                metrics_path = job_dir / "metrics.jsonl"
                if _remove_file(events_path, apply=apply):
                    step["events_removed"] = int(step["events_removed"]) + 1
                if _remove_file(metrics_path, apply=apply):
                    step["metrics_removed"] = int(step["metrics_removed"]) + 1
        report["steps"]["runtime_backup_stream_cleanup"] = step

    dataset_registry = _load_json_dict(data_root / "dataset_registry.json", default={"dataset_sources": [], "built_datasets": []})
    protected_global_target_ids = _collect_protected_global_target_ids(config)
    latest_manifest_payload, _ = _load_latest_pipeline_manifest(drive_root)
    if isinstance(latest_manifest_payload, dict):
        latest_manifest_target = str(latest_manifest_payload.get("global_target_id", "")).strip()
        if latest_manifest_target:
            protected_global_target_ids.add(latest_manifest_target)
    protected_job_ids = _collect_protected_job_ids(config, latest_manifest_payload)

    global_progress_step: dict[str, Any] | None = None
    if cleanup_global_progress_mirrors or cleanup_retention:
        global_progress_step = _cleanup_global_progress_mirrors(
            data_root=data_root,
            apply=bool(apply),
            protected_target_ids=protected_global_target_ids,
            global_progress_ttl_seconds=float(global_progress_ttl_seconds),
            global_progress_keep_recent=int(global_progress_keep_recent),
            global_progress_hard_max_age_seconds=float(global_progress_hard_max_age_seconds),
            now_epoch=now_epoch,
        )
    if cleanup_global_progress_mirrors and isinstance(global_progress_step, dict):
        report["steps"]["global_progress_cleanup"] = global_progress_step

    pipeline_manifest_step: dict[str, Any] | None = None
    if cleanup_pipeline_manifests or cleanup_retention:
        pipeline_manifest_step = _cleanup_pipeline_manifests(
            pipeline_root=drive_root / "logs" / "pipeline",
            apply=bool(apply),
            pipeline_manifest_ttl_seconds=float(pipeline_manifest_ttl_seconds),
            pipeline_manifest_keep_recent=int(pipeline_manifest_keep_recent),
            pipeline_manifest_hard_max_age_seconds=float(pipeline_manifest_hard_max_age_seconds),
            now_epoch=now_epoch,
        )
    if cleanup_pipeline_manifests and isinstance(pipeline_manifest_step, dict):
        report["steps"]["pipeline_manifest_cleanup"] = pipeline_manifest_step

    completed_job_dirs_step: dict[str, Any] | None = None
    if cleanup_completed_job_dirs or cleanup_retention:
        drive_jobs_root = drive_root / "jobs"
        backup_root = paths.jobs_backup_root or (drive_root / "runtime_backup" / "jobs")
        dedup_roots: list[Path] = []
        for root in [drive_jobs_root, backup_root]:
            if root not in dedup_roots:
                dedup_roots.append(root)
        completed_job_dirs_step = _cleanup_completed_job_dirs(
            job_roots=dedup_roots,
            apply=bool(apply),
            protected_job_ids=protected_job_ids,
            job_dir_ttl_seconds=float(job_dir_ttl_seconds),
            job_dir_keep_recent_per_run_type=int(job_dir_keep_recent_per_run_type),
            job_dir_hard_max_age_seconds=float(job_dir_hard_max_age_seconds),
            now_epoch=now_epoch,
        )
    if cleanup_completed_job_dirs and isinstance(completed_job_dirs_step, dict):
        report["steps"]["completed_job_dirs_cleanup"] = completed_job_dirs_step

    if cleanup_duplicate_source_metadata:
        report["steps"]["duplicate_source_metadata_cleanup"] = _cleanup_duplicate_source_metadata(
            dataset_registry=dataset_registry,
            drive_root=drive_root,
            apply=bool(apply),
            raw_metadata_ttl_seconds=float(source_metadata_raw_ttl_seconds),
            now_epoch=now_epoch,
        )

    if cleanup_quarantine_dirs:
        report["steps"]["quarantine_cleanup"] = _cleanup_quarantine_dirs(
            drive_root=drive_root,
            apply=bool(apply),
            quarantine_ttl_seconds=float(quarantine_ttl_seconds),
            now_epoch=now_epoch,
        )

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

    if cleanup_retention:
        retention_step: dict[str, Any] = {
            "job_stream_ttl_seconds": float(job_stream_ttl_seconds),
            "quarantine_ttl_seconds": float(quarantine_ttl_seconds),
            "benchmark_report_ttl_seconds": float(benchmark_report_ttl_seconds),
            "benchmark_keep_recent": int(benchmark_keep_recent),
            "checkpoint_ttl_seconds": float(checkpoint_ttl_seconds),
            "checkpoint_keep_recent_per_model": int(checkpoint_keep_recent_per_model),
            "global_progress_ttl_seconds": float(global_progress_ttl_seconds),
            "global_progress_keep_recent": int(global_progress_keep_recent),
            "global_progress_hard_max_age_seconds": float(global_progress_hard_max_age_seconds),
            "pipeline_manifest_ttl_seconds": float(pipeline_manifest_ttl_seconds),
            "pipeline_manifest_keep_recent": int(pipeline_manifest_keep_recent),
            "pipeline_manifest_hard_max_age_seconds": float(pipeline_manifest_hard_max_age_seconds),
            "job_dir_ttl_seconds": float(job_dir_ttl_seconds),
            "job_dir_keep_recent_per_run_type": int(job_dir_keep_recent_per_run_type),
            "job_dir_hard_max_age_seconds": float(job_dir_hard_max_age_seconds),
            "source_metadata_raw_ttl_seconds": float(source_metadata_raw_ttl_seconds),
            "drive_root_artifact_ttl_seconds": float(drive_root_artifact_ttl_seconds),
            "drive_root_artifact_keep_recent_per_key": int(drive_root_artifact_keep_recent_per_key),
            "drive_root_artifact_hard_max_age_seconds": float(drive_root_artifact_hard_max_age_seconds),
            "recovered_external_ttl_seconds": float(recovered_external_ttl_seconds),
            "recovered_external_keep_recent_sessions": int(recovered_external_keep_recent_sessions),
            "recovered_external_hard_max_age_seconds": float(recovered_external_hard_max_age_seconds),
            "benchmark_report_hard_max_age_seconds": float(benchmark_report_hard_max_age_seconds),
            "checkpoint_hard_max_age_seconds": float(checkpoint_hard_max_age_seconds),
            "quarantine_removed": [],
            "quarantine_skipped_recent": 0,
            "benchmark_reports_removed": [],
            "old_checkpoints_removed": [],
            "drive_root_operational_artifacts_removed": [],
            "recovered_external_sessions_removed": [],
            "recovered_external_empty_dirs_removed": [],
            "global_progress_removed": [],
            "global_progress_workers_dirs_removed": [],
            "global_progress_worker_snapshots_removed": [],
            "pipeline_manifests_removed": [],
            "completed_job_dirs_removed": [],
            "source_metadata_raw_removed": [],
            "global_progress_removed_hard_max_age": 0,
            "pipeline_manifests_removed_hard_max_age": 0,
            "completed_job_dirs_removed_hard_max_age": 0,
            "drive_root_operational_removed_hard_max_age": 0,
            "recovered_external_removed_hard_max_age": 0,
            "benchmark_reports_removed_hard_max_age": 0,
            "old_checkpoints_removed_hard_max_age": 0,
            "errors": [],
        }

        if isinstance(global_progress_step, dict):
            retention_step["global_progress_removed"] = list(global_progress_step.get("removed_progress_files", []) or [])
            retention_step["global_progress_workers_dirs_removed"] = list(global_progress_step.get("removed_workers_dirs", []) or [])
            retention_step["global_progress_worker_snapshots_removed"] = list(
                global_progress_step.get("removed_worker_snapshots", []) or []
            )
            retention_step["global_progress_removed_hard_max_age"] = int(
                global_progress_step.get("removed_hard_max_age", 0) or 0
            )
            retention_step["errors"].extend(list(global_progress_step.get("errors", []) or []))
        if isinstance(pipeline_manifest_step, dict):
            retention_step["pipeline_manifests_removed"] = list(pipeline_manifest_step.get("removed", []) or [])
            retention_step["pipeline_manifests_removed_hard_max_age"] = int(
                pipeline_manifest_step.get("removed_hard_max_age", 0) or 0
            )
            retention_step["errors"].extend(list(pipeline_manifest_step.get("errors", []) or []))
        if isinstance(completed_job_dirs_step, dict):
            retention_step["completed_job_dirs_removed"] = list(completed_job_dirs_step.get("removed", []) or [])
            retention_step["completed_job_dirs_removed_hard_max_age"] = int(
                completed_job_dirs_step.get("removed_hard_max_age", 0) or 0
            )
            retention_step["errors"].extend(list(completed_job_dirs_step.get("errors", []) or []))
        source_metadata_step = _cleanup_duplicate_source_metadata(
            dataset_registry=dataset_registry,
            drive_root=drive_root,
            apply=bool(apply),
            raw_metadata_ttl_seconds=float(source_metadata_raw_ttl_seconds),
            now_epoch=now_epoch,
        )
        retention_step["source_metadata_raw_removed"] = list(source_metadata_step.get("removed_raw_metadata", []) or [])
        retention_step["errors"].extend(list(source_metadata_step.get("errors", []) or []))

        # 1) Quarantine dirs older than TTL.
        quarantine_step = _cleanup_quarantine_dirs(
            drive_root=drive_root,
            apply=bool(apply),
            quarantine_ttl_seconds=float(quarantine_ttl_seconds),
            now_epoch=now_epoch,
        )
        retention_step["quarantine_removed"] = list(quarantine_step.get("removed", []))
        retention_step["quarantine_skipped_recent"] = int(quarantine_step.get("skipped_recent", 0) or 0)
        retention_step["errors"].extend(list(quarantine_step.get("errors", []) or []))

        # 1.5) Drive root operational artifacts rotation (state/config/run_status/... duplicates).
        drive_root_operational_step = _cleanup_drive_root_operational_artifacts(
            drive_root=drive_root,
            apply=bool(apply),
            artifact_ttl_seconds=float(drive_root_artifact_ttl_seconds),
            artifact_keep_recent_per_key=int(drive_root_artifact_keep_recent_per_key),
            artifact_hard_max_age_seconds=float(drive_root_artifact_hard_max_age_seconds),
            now_epoch=now_epoch,
        )
        retention_step["drive_root_operational_artifacts_removed"] = list(
            drive_root_operational_step.get("removed", []) or []
        )
        retention_step["drive_root_operational_removed_hard_max_age"] = int(
            drive_root_operational_step.get("removed_hard_max_age", 0) or 0
        )
        retention_step["errors"].extend(list(drive_root_operational_step.get("errors", []) or []))

        # 1.6) Recovered external sessions rotation (post-migration holding area).
        recovered_external_step = _cleanup_recovered_external_sessions(
            drive_root=drive_root,
            apply=bool(apply),
            recovered_external_ttl_seconds=float(recovered_external_ttl_seconds),
            recovered_external_keep_recent_sessions=int(recovered_external_keep_recent_sessions),
            recovered_external_hard_max_age_seconds=float(recovered_external_hard_max_age_seconds),
            now_epoch=now_epoch,
        )
        retention_step["recovered_external_sessions_removed"] = list(
            recovered_external_step.get("removed_sessions", []) or []
        )
        retention_step["recovered_external_empty_dirs_removed"] = list(
            recovered_external_step.get("removed_empty_dirs", []) or []
        )
        retention_step["recovered_external_removed_hard_max_age"] = int(
            recovered_external_step.get("removed_hard_max_age", 0) or 0
        )
        retention_step["errors"].extend(list(recovered_external_step.get("errors", []) or []))

        # 2) Benchmark reports rotation with TTL + keep recent.
        benchmark_root = paths.reports_root / "benchmarks"
        benchmark_candidates: list[Path] = []
        if benchmark_root.exists() and _path_within(benchmark_root, drive_root):
            for path in benchmark_root.rglob("*"):
                if not path.is_file():
                    continue
                if path.name == "benchmark_history.jsonl":
                    continue
                if path.suffix.lower() not in {".json", ".jsonl"}:
                    continue
                benchmark_candidates.append(path)
            benchmark_candidates_sorted = sorted(
                benchmark_candidates,
                key=lambda item: float(item.stat().st_mtime),
                reverse=True,
            )
            for idx, path in enumerate(benchmark_candidates_sorted):
                age_seconds = _path_age_seconds(path, now_epoch=now_epoch)
                if idx < int(benchmark_keep_recent):
                    if (
                        float(benchmark_report_hard_max_age_seconds) > 0.0
                        and age_seconds >= float(benchmark_report_hard_max_age_seconds)
                    ):
                        if _remove_file(path, apply=apply):
                            retention_step["benchmark_reports_removed"].append(str(path))
                            retention_step["benchmark_reports_removed_hard_max_age"] = int(
                                retention_step["benchmark_reports_removed_hard_max_age"]
                            ) + 1
                    continue
                if (
                    float(benchmark_report_hard_max_age_seconds) > 0.0
                    and age_seconds >= float(benchmark_report_hard_max_age_seconds)
                ):
                    if _remove_file(path, apply=apply):
                        retention_step["benchmark_reports_removed"].append(str(path))
                        retention_step["benchmark_reports_removed_hard_max_age"] = int(
                            retention_step["benchmark_reports_removed_hard_max_age"]
                        ) + 1
                    continue
                if age_seconds < float(benchmark_report_ttl_seconds):
                    continue
                if _remove_file(path, apply=apply):
                    retention_step["benchmark_reports_removed"].append(str(path))

        # 3) Old per-epoch checkpoints cleanup (keep best/last + recent per model).
        checkpoints_dir = models_root / "checkpoints"
        per_model_candidates: dict[str, list[Path]] = {}
        if checkpoints_dir.exists() and _path_within(checkpoints_dir, drive_root):
            for path in checkpoints_dir.glob("*.pt"):
                if not path.is_file():
                    continue
                stem = str(path.stem or "").strip()
                if not stem or stem.endswith("_best") or stem.endswith("_last"):
                    continue
                model_id = _infer_checkpoint_model_id(stem)
                if not _is_safe_model_id(model_id):
                    continue
                per_model_candidates.setdefault(model_id, []).append(path)
            for model_id, paths_for_model in per_model_candidates.items():
                sorted_paths = sorted(paths_for_model, key=lambda item: float(item.stat().st_mtime), reverse=True)
                for idx, path in enumerate(sorted_paths):
                    age_seconds = _path_age_seconds(path, now_epoch=now_epoch)
                    if idx < int(checkpoint_keep_recent_per_model):
                        if (
                            float(checkpoint_hard_max_age_seconds) > 0.0
                            and age_seconds >= float(checkpoint_hard_max_age_seconds)
                        ):
                            if _remove_file(path, apply=apply):
                                retention_step["old_checkpoints_removed"].append(str(path))
                                retention_step["old_checkpoints_removed_hard_max_age"] = int(
                                    retention_step["old_checkpoints_removed_hard_max_age"]
                                ) + 1
                        continue
                    if (
                        float(checkpoint_hard_max_age_seconds) > 0.0
                        and age_seconds >= float(checkpoint_hard_max_age_seconds)
                    ):
                        if _remove_file(path, apply=apply):
                            retention_step["old_checkpoints_removed"].append(str(path))
                            retention_step["old_checkpoints_removed_hard_max_age"] = int(
                                retention_step["old_checkpoints_removed_hard_max_age"]
                            ) + 1
                        continue
                    if age_seconds < float(checkpoint_ttl_seconds):
                        continue
                    if not _path_within(path, models_root):
                        continue
                    if _remove_file(path, apply=apply):
                        retention_step["old_checkpoints_removed"].append(str(path))
        report["steps"]["retention_cleanup"] = retention_step

    if cleanup_external_artifacts:
        report["steps"]["external_artifacts_cleanup"] = _cleanup_external_drive_artifacts(
            drive_root=drive_root,
            apply=bool(apply),
            now_epoch=now_epoch,
        )

    report["created_at_epoch"] = time.time()
    return report
