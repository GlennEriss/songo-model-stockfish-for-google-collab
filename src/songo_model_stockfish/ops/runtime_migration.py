from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from songo_model_stockfish.ops.io_utils import acquire_lock_dir, release_lock_dir


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"running", "pending", "starting"}


def read_json_safe(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = dict(default or {})
    if not path.exists():
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    return payload if isinstance(payload, dict) else fallback


def parse_iso_to_epoch(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        from datetime import datetime

        return float(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0.0


def is_pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    proc = subprocess.run(
        ["ps", "-p", str(int(pid)), "-o", "pid="],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip())


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(max(1024, int(chunk_size)))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _index_files_with_hash(root: Path, *, chunk_size: int = 1024 * 1024) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return index
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = str(file_path.relative_to(root))
        size = int(file_path.stat().st_size)
        index[rel] = {
            "size": size,
            "sha256": file_sha256(file_path, chunk_size=chunk_size),
        }
    return index


def verify_tree_hash(src: Path, dst: Path, *, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    src_idx = _index_files_with_hash(src, chunk_size=chunk_size)
    dst_idx = _index_files_with_hash(dst, chunk_size=chunk_size)
    missing = [rel for rel in src_idx.keys() if rel not in dst_idx]
    mismatch_size = [rel for rel in src_idx.keys() if rel in dst_idx and int(src_idx[rel]["size"]) != int(dst_idx[rel]["size"])]
    mismatch_hash = [
        rel
        for rel in src_idx.keys()
        if rel in dst_idx and str(src_idx[rel]["sha256"]) != str(dst_idx[rel]["sha256"])
    ]
    return {
        "verified": not (missing or mismatch_size or mismatch_hash),
        "src_files": int(len(src_idx)),
        "dst_files": int(len(dst_idx)),
        "missing_files_count": int(len(missing)),
        "mismatch_size_count": int(len(mismatch_size)),
        "mismatch_hash_count": int(len(mismatch_hash)),
        "missing_files_preview": missing[:10],
        "mismatch_size_preview": mismatch_size[:10],
        "mismatch_hash_preview": mismatch_hash[:10],
    }


def sync_tree_with_hash_verify(src: Path, dst: Path, *, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    copied = 0
    updated = 0
    bytes_copied = 0
    dst.mkdir(parents=True, exist_ok=True)
    for src_file in src.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src)
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        existed_before = dst_file.exists()
        copy_required = True
        if existed_before:
            try:
                src_size = int(src_file.stat().st_size)
                dst_size = int(dst_file.stat().st_size)
                if src_size == dst_size:
                    copy_required = file_sha256(src_file, chunk_size=chunk_size) != file_sha256(dst_file, chunk_size=chunk_size)
            except Exception:
                copy_required = True
        if copy_required:
            shutil.copy2(src_file, dst_file)
            bytes_copied += int(src_file.stat().st_size)
            if existed_before:
                updated += 1
            else:
                copied += 1
    verification = verify_tree_hash(src, dst, chunk_size=chunk_size)
    verification.update(
        {
            "copied": int(copied),
            "updated": int(updated),
            "bytes_copied": int(bytes_copied),
        }
    )
    return verification


def _manifest_process_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    processes = manifest.get("processes", {})
    if isinstance(processes, dict):
        for label, info in processes.items():
            if not isinstance(info, dict):
                continue
            rows.append(
                {
                    "label": str(label),
                    "pid": int(info.get("pid", 0) or 0),
                    "command": str(info.get("command", "")).strip(),
                }
            )
    if rows:
        return rows
    return [
        {
            "label": "dataset-generate",
            "pid": int(manifest.get("generate_pid", 0) or 0),
            "command": str(manifest.get("generate_cmd", "")).strip(),
        },
        {
            "label": "dataset-build",
            "pid": int(manifest.get("build_pid", 0) or 0),
            "command": str(manifest.get("build_cmd", "")).strip(),
        },
    ]


def manifest_pid_candidates_for_job(manifest: dict[str, Any], job_id: str) -> set[int]:
    target = str(job_id).strip()
    if not target or not isinstance(manifest, dict):
        return set()
    pids: set[int] = set()
    if str(manifest.get("generate_job_id", "")).strip() == target:
        pid = int(manifest.get("generate_pid", 0) or 0)
        if pid > 0:
            pids.add(pid)
    if str(manifest.get("build_job_id", "")).strip() == target:
        pid = int(manifest.get("build_pid", 0) or 0)
        if pid > 0:
            pids.add(pid)
    for row in _manifest_process_records(manifest):
        pid = int(row.get("pid", 0) or 0)
        if pid <= 0:
            continue
        cmd = str(row.get("command", "")).strip()
        if f"--job-id {target}" in cmd:
            pids.add(pid)
    return pids


def any_manifest_pid_alive(manifest: dict[str, Any], *, pid_check_fn: Callable[[int], bool] = is_pid_alive) -> bool:
    for row in _manifest_process_records(manifest):
        pid = int(row.get("pid", 0) or 0)
        if pid > 0 and bool(pid_check_fn(pid)):
            return True
    return False


def is_job_active(
    job_dir: Path,
    *,
    pid_candidates: set[int] | None = None,
    active_updated_max_age_seconds: float = 300.0,
    pid_check_fn: Callable[[int], bool] = is_pid_alive,
    now_ts: float | None = None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    now_epoch = float(now_ts) if now_ts is not None else time.time()
    status_payload = read_json_safe(job_dir / "run_status.json", default={})
    status_text = str(status_payload.get("status", "")).strip().lower()
    if status_text in ACTIVE_STATUSES:
        reasons.append(f"status={status_text}")
    updated_ts = parse_iso_to_epoch(status_payload.get("updated_at", ""))
    if updated_ts > 0:
        age_seconds = max(0.0, now_epoch - updated_ts)
        if age_seconds <= max(1.0, float(active_updated_max_age_seconds)) and status_text not in TERMINAL_STATUSES:
            reasons.append(f"updated_recently={int(age_seconds)}s")
    for pid in sorted(pid_candidates or set()):
        if int(pid) > 0 and bool(pid_check_fn(int(pid))):
            reasons.append(f"pid_alive={int(pid)}")
            break
    return (len(reasons) > 0), reasons


def load_manifest_prefer_local(local_manifest_path: Path, *, firestore_manifest: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    local_payload = read_json_safe(local_manifest_path, default={})
    if local_payload:
        return local_payload, f"local:{local_manifest_path}"
    firestore_payload = firestore_manifest if isinstance(firestore_manifest, dict) else {}
    if firestore_payload:
        return firestore_payload, "firestore"
    return {}, "none"


def _path_within(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _resolve_allowed_drive_root() -> Path:
    mydrive_root = Path("/content/drive/MyDrive")
    configured_text = str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip()
    if configured_text:
        configured = Path(configured_text)
    else:
        configured = mydrive_root / "songo-stockfish"
    expected = mydrive_root / "songo-stockfish"
    if configured == mydrive_root:
        return expected
    if _path_within(configured, mydrive_root) and not _path_within(configured, expected):
        return expected
    if _path_within(configured, expected):
        return expected
    if not _path_within(configured, mydrive_root):
        return configured
    return expected


def _resolve_quarantine_root_for_src(src_path: Path, quarantine_root: Path | None) -> Path | None:
    mydrive_root = Path("/content/drive/MyDrive")
    if not _path_within(src_path, mydrive_root):
        return quarantine_root

    allowed_drive_root = _resolve_allowed_drive_root()
    default_drive_quarantine_root = allowed_drive_root / "runtime_migration" / "quarantine"
    if quarantine_root is None:
        return default_drive_quarantine_root

    # Garde-fou: si un quarantine_root MyDrive pointe hors drive_root autorise,
    # on reroute vers le dossier de quarantine du projet.
    if _path_within(quarantine_root, mydrive_root) and not _path_within(quarantine_root, allowed_drive_root):
        return default_drive_quarantine_root
    return quarantine_root


def _build_quarantine_path(src_path: Path, quarantine_root: Path | None) -> Path:
    stem = f".quarantine_{src_path.name}_{int(time.time())}"
    resolved_quarantine_root = _resolve_quarantine_root_for_src(src_path, quarantine_root)
    if resolved_quarantine_root is None:
        candidate = src_path.with_name(stem)
        if not candidate.exists():
            return candidate
        suffix = 1
        while True:
            alt = src_path.with_name(f"{stem}_{suffix}")
            if not alt.exists():
                return alt
            suffix += 1
    resolved_quarantine_root.mkdir(parents=True, exist_ok=True)
    candidate = resolved_quarantine_root / stem
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        alt = resolved_quarantine_root / f"{stem}_{suffix}"
        if not alt.exists():
            return alt
        suffix += 1


def run_drive_to_local_runtime_migration(
    *,
    drive_jobs_root: Path,
    drive_pipeline_logs_root: Path,
    local_jobs_root: Path,
    local_pipeline_logs_root: Path,
    manifest: dict[str, Any] | None = None,
    purge_after_verify: bool = True,
    skip_active_job_dirs: bool = True,
    active_updated_max_age_seconds: float = 300.0,
    verbose: bool = True,
    lock_dir: Path | None = None,
    lock_timeout_seconds: float = 60.0,
    lock_poll_seconds: float = 0.25,
    hash_chunk_size: int = 1024 * 1024,
    pid_check_fn: Callable[[int], bool] = is_pid_alive,
    quarantine_root: Path | None = None,
) -> dict[str, Any]:
    lock_acquired = False
    if lock_dir is not None:
        lock_acquired = acquire_lock_dir(
            lock_dir,
            timeout_seconds=float(lock_timeout_seconds),
            poll_seconds=float(lock_poll_seconds),
            stale_after_seconds=max(60.0, float(lock_timeout_seconds) * 3.0),
        )
        if not lock_acquired:
            raise TimeoutError(f"Impossible d'obtenir le lock migration: {lock_dir}")

    local_jobs_root.mkdir(parents=True, exist_ok=True)
    local_pipeline_logs_root.mkdir(parents=True, exist_ok=True)
    resolved_manifest = manifest if isinstance(manifest, dict) else {}
    effective_quarantine_root = _resolve_quarantine_root_for_src(
        drive_jobs_root if drive_jobs_root.exists() else drive_pipeline_logs_root,
        quarantine_root,
    )

    summary: dict[str, Any] = {
        "lock": {"path": str(lock_dir) if lock_dir is not None else "", "acquired": bool(lock_acquired)},
        "jobs": {
            "migrated": 0,
            "skipped_active": 0,
            "skipped_active_recheck": 0,
            "purged": 0,
            "failed": 0,
            "details": [],
        },
        "pipeline_logs": {
            "migrated": 0,
            "purged": 0,
            "failed": 0,
            "purge_skipped_active": 0,
            "details": [],
        },
    }

    try:
        if drive_jobs_root.exists():
            for src_job_dir in sorted([p for p in drive_jobs_root.iterdir() if p.is_dir()]):
                job_id = src_job_dir.name
                pid_candidates = manifest_pid_candidates_for_job(resolved_manifest, job_id)
                active, reasons = is_job_active(
                    src_job_dir,
                    pid_candidates=pid_candidates,
                    active_updated_max_age_seconds=float(active_updated_max_age_seconds),
                    pid_check_fn=pid_check_fn,
                )
                if skip_active_job_dirs and active:
                    summary["jobs"]["skipped_active"] = int(summary["jobs"]["skipped_active"]) + 1
                    summary["jobs"]["details"].append({"job_id": job_id, "status": "skipped_active", "reasons": list(reasons)})
                    continue

                dst_job_dir = local_jobs_root / job_id
                try:
                    result = sync_tree_with_hash_verify(src_job_dir, dst_job_dir, chunk_size=hash_chunk_size)
                    if not bool(result.get("verified", False)):
                        summary["jobs"]["failed"] = int(summary["jobs"]["failed"]) + 1
                        summary["jobs"]["details"].append(
                            {
                                "job_id": job_id,
                                "status": "verify_failed",
                                "result": result,
                            }
                        )
                        continue
                    summary["jobs"]["migrated"] = int(summary["jobs"]["migrated"]) + 1
                    detail = {"job_id": job_id, "status": "migrated", "result": result}

                    if purge_after_verify:
                        quarantine_path = _build_quarantine_path(src_job_dir, quarantine_root)
                        try:
                            shutil.move(str(src_job_dir), str(quarantine_path))
                            active_after, reasons_after = is_job_active(
                                quarantine_path,
                                pid_candidates=pid_candidates,
                                active_updated_max_age_seconds=float(active_updated_max_age_seconds),
                                pid_check_fn=pid_check_fn,
                            )
                            if active_after:
                                shutil.move(str(quarantine_path), str(src_job_dir))
                                summary["jobs"]["skipped_active_recheck"] = int(summary["jobs"]["skipped_active_recheck"]) + 1
                                detail["status"] = "purge_skipped_active_recheck"
                                detail["reasons_after"] = list(reasons_after)
                            else:
                                shutil.rmtree(quarantine_path, ignore_errors=True)
                                summary["jobs"]["purged"] = int(summary["jobs"]["purged"]) + 1
                                detail["status"] = "purged"
                        except Exception as exc:
                            summary["jobs"]["failed"] = int(summary["jobs"]["failed"]) + 1
                            detail["status"] = "purge_error"
                            detail["error"] = f"{type(exc).__name__}: {exc}"
                    summary["jobs"]["details"].append(detail)
                except Exception as exc:
                    summary["jobs"]["failed"] = int(summary["jobs"]["failed"]) + 1
                    summary["jobs"]["details"].append(
                        {
                            "job_id": job_id,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        if drive_pipeline_logs_root.exists():
            try:
                result = sync_tree_with_hash_verify(drive_pipeline_logs_root, local_pipeline_logs_root, chunk_size=hash_chunk_size)
                if not bool(result.get("verified", False)):
                    summary["pipeline_logs"]["failed"] = 1
                    summary["pipeline_logs"]["details"].append({"status": "verify_failed", "result": result})
                else:
                    summary["pipeline_logs"]["migrated"] = 1
                    summary["pipeline_logs"]["details"].append({"status": "migrated", "result": result})
                    if purge_after_verify:
                        if any_manifest_pid_alive(resolved_manifest, pid_check_fn=pid_check_fn):
                            summary["pipeline_logs"]["purge_skipped_active"] = 1
                            summary["pipeline_logs"]["details"].append({"status": "purge_skipped_active_manifest_pid"})
                        else:
                            quarantine_logs = _build_quarantine_path(drive_pipeline_logs_root, quarantine_root)
                            shutil.move(str(drive_pipeline_logs_root), str(quarantine_logs))
                            if any_manifest_pid_alive(resolved_manifest, pid_check_fn=pid_check_fn):
                                shutil.move(str(quarantine_logs), str(drive_pipeline_logs_root))
                                summary["pipeline_logs"]["purge_skipped_active"] = 1
                                summary["pipeline_logs"]["details"].append({"status": "purge_skipped_active_recheck"})
                            else:
                                shutil.rmtree(quarantine_logs, ignore_errors=True)
                                summary["pipeline_logs"]["purged"] = 1
                                summary["pipeline_logs"]["details"].append({"status": "purged"})
            except Exception as exc:
                summary["pipeline_logs"]["failed"] = 1
                summary["pipeline_logs"]["details"].append({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        if verbose:
            summary["runtime"] = {
                "drive_jobs_root": str(drive_jobs_root),
                "drive_pipeline_logs_root": str(drive_pipeline_logs_root),
                "local_jobs_root": str(local_jobs_root),
                "local_pipeline_logs_root": str(local_pipeline_logs_root),
                "purge_after_verify": bool(purge_after_verify),
                "skip_active_job_dirs": bool(skip_active_job_dirs),
                "active_updated_max_age_seconds": float(active_updated_max_age_seconds),
                "quarantine_root": str(quarantine_root) if quarantine_root is not None else "",
                "effective_quarantine_root": str(effective_quarantine_root) if effective_quarantine_root is not None else "",
            }
    finally:
        if lock_dir is not None and lock_acquired:
            release_lock_dir(lock_dir)

    if purge_after_verify:
        drive_jobs_root.mkdir(parents=True, exist_ok=True)
        drive_pipeline_logs_root.mkdir(parents=True, exist_ok=True)
    return summary
