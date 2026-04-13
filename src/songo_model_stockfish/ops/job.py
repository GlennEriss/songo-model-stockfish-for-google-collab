from __future__ import annotations

import json
import random
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from songo_model_stockfish.ops.io_utils import read_json_dict, write_json_atomic, write_text_atomic
from songo_model_stockfish.ops.logging import JsonlWriter, build_console_logger, utc_now_iso
from songo_model_stockfish.ops.paths import ProjectPaths, build_project_paths


def _first_non_empty(candidates: list[Any]) -> str:
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


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


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_firestore_sync_config(config: dict[str, Any]) -> dict[str, Any]:
    firestore_cfg = config.get("firestore", {}) if isinstance(config.get("firestore"), dict) else {}
    dataset_gen_cfg = config.get("dataset_generation", {}) if isinstance(config.get("dataset_generation"), dict) else {}
    dataset_build_cfg = config.get("dataset_build", {}) if isinstance(config.get("dataset_build"), dict) else {}
    dataset_merge_cfg = config.get("dataset_merge_final", {}) if isinstance(config.get("dataset_merge_final"), dict) else {}
    search_spaces = [firestore_cfg, dataset_gen_cfg, dataset_build_cfg, dataset_merge_cfg]

    def pick(*keys: str) -> str:
        values: list[Any] = []
        for space in search_spaces:
            for key in keys:
                values.append(space.get(key))
        return _first_non_empty(values)

    backend = pick("job_firestore_backend", "global_progress_backend", "global_target_progress_backend").lower() or "file"
    enabled_flag = pick("job_firestore_enabled").lower()
    enabled = backend == "firestore" or enabled_flag in {"1", "true", "yes", "on"}

    strict_flag = pick("job_firestore_strict").lower()
    strict = strict_flag in {"", "1", "true", "yes", "on"}
    checkpoint_min_interval_seconds = max(
        0.0,
        _as_float(
            pick("job_firestore_checkpoint_min_interval_seconds", "worker_checkpoints_min_interval_seconds") or "30",
            30.0,
        ),
    )
    checkpoint_state_only_on_change = _as_bool(
        pick("job_firestore_checkpoint_state_only_on_change") or "1",
        default=True,
    )
    project_id = pick(
        "job_firestore_project_id",
        "global_progress_firestore_project_id",
        "global_target_progress_firestore_project_id",
    )
    collection = pick("worker_checkpoints_firestore_collection", "job_firestore_collection") or "worker_checkpoints"
    credentials_path = pick(
        "job_firestore_credentials_path",
        "global_progress_firestore_credentials_path",
        "global_target_progress_firestore_credentials_path",
    )
    api_key = pick(
        "job_firestore_api_key",
        "global_progress_firestore_api_key",
        "global_target_progress_firestore_api_key",
    )
    return {
        "enabled": bool(enabled),
        "strict": bool(strict),
        "project_id": project_id,
        "collection": collection,
        "credentials_path": credentials_path,
        "api_key": api_key,
        "checkpoint_min_interval_seconds": checkpoint_min_interval_seconds,
        "checkpoint_state_only_on_change": checkpoint_state_only_on_change,
    }


def _firestore_auth_mode(credentials_path: str, api_key: str) -> str:
    if str(credentials_path or "").strip():
        return "service_account_file"
    if str(api_key or "").strip():
        return "api_key"
    return "missing_credentials"


def _firestore_sync_diag(sync: dict[str, Any]) -> dict[str, Any]:
    credentials_path = str(sync.get("credentials_path", "")).strip()
    api_key = str(sync.get("api_key", "")).strip()
    auth_mode = _firestore_auth_mode(credentials_path, api_key)
    return {
        "enabled": bool(sync.get("enabled", False)),
        "strict": bool(sync.get("strict", True)),
        "project_id": str(sync.get("project_id", "")).strip(),
        "collection": str(sync.get("collection", "")).strip() or "worker_checkpoints",
        "auth_mode": auth_mode,
        "credentials_path": credentials_path,
        "credentials_path_exists": bool(Path(credentials_path).exists()) if credentials_path else False,
        "api_key_set": bool(api_key),
        "checkpoint_min_interval_seconds": float(_as_float(sync.get("checkpoint_min_interval_seconds", 30.0), 30.0)),
        "checkpoint_state_only_on_change": bool(_as_bool(sync.get("checkpoint_state_only_on_change", True), default=True)),
    }


def _firestore_error_hint(exc: Exception, *, auth_mode: str) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    auth_mode = str(auth_mode).strip().lower()
    if auth_mode == "api_key":
        return "google-cloud-firestore n'accepte pas API key seule; configure job_firestore_credentials_path."
    if auth_mode == "missing_credentials":
        return "credentials Firestore absents; configure job_firestore_credentials_path."
    if "metadata.google.internal" in text or "compute metadata" in text:
        return "ADC metadata indisponible ici; utilise un service account JSON explicite."
    if "permissiondenied" in text or "permission denied" in text:
        return "acces refuse par IAM/regles Firestore."
    if "unauthenticated" in text or "invalid authentication credentials" in text:
        return "authentification Firestore invalide."
    if "resource_exhausted" in text or "quota exceeded" in text or "429" in text:
        return "quota Firestore depasse; augmente le batching/throttling et reduis le polling."
    if "deadlineexceeded" in text or "timeout" in text:
        return "timeout reseau vers Firestore."
    if "serviceunavailable" in text:
        return "service Firestore temporairement indisponible."
    return ""


def _build_firestore_job_client(*, project_id: str, credentials_path: str, api_key: str):
    from google.cloud import firestore

    credentials = None
    client_options = None
    if credentials_path:
        if not Path(credentials_path).exists():
            raise FileNotFoundError(f"Fichier credentials Firestore introuvable: {credentials_path}")
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(credentials_path)
    elif api_key:
        raise RuntimeError(
            "Mode API key non supporte avec google-cloud-firestore; "
            "configure `job_firestore_credentials_path`."
        )
    else:
        raise RuntimeError(
            "Credentials Firestore absents; configure `job_firestore_credentials_path`."
        )
    return firestore.Client(project=(project_id or None), credentials=credentials, client_options=client_options)


def make_job_id(run_type: str) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    return f"{run_type}_{stamp}_{suffix}"


def _next_cycle_job_id(requested_job_id: str, jobs_root: Path, *, backup_jobs_root: Path | None = None) -> str:
    def _job_id_exists(candidate_job_id: str) -> bool:
        if (jobs_root / candidate_job_id).exists():
            return True
        if backup_jobs_root is not None and (backup_jobs_root / candidate_job_id).exists():
            return True
        return False

    stem = requested_job_id
    if stem.endswith("_") and len(stem) > 1:
        stem = stem[:-1]

    import re

    match = re.search(r"^(.*?)(\d+)$", stem)
    if match:
        prefix = match.group(1)
        width = len(match.group(2))
        candidate = int(match.group(2))
        while True:
            candidate += 1
            next_job_id = f"{prefix}{candidate:0{width}d}"
            if not _job_id_exists(next_job_id):
                return next_job_id

    index = 2
    while True:
        next_job_id = f"{requested_job_id}_{index:03d}"
        if not _job_id_exists(next_job_id):
            return next_job_id
        index += 1


@dataclass
class JobContext:
    config: dict[str, Any]
    paths: ProjectPaths
    job_id: str
    run_type: str
    job_dir: Path
    logger: Any
    events: JsonlWriter
    metrics: JsonlWriter
    status_path: Path
    state_path: Path
    firestore_sync: dict[str, Any]
    _last_firestore_checkpoint_write_ts: float = 0.0
    _last_firestore_status_signature: str = ""
    _last_firestore_state_signature: str = ""
    _firestore_sync_stats: dict[str, int] | None = None
    _last_runtime_backup_state_write_ts: float = 0.0
    _last_runtime_backup_state_signature: str = ""
    _last_runtime_backup_state_change_ts: float = 0.0
    _runtime_backup_state_dirty: bool = False

    def read_status(self) -> dict[str, Any]:
        payload = read_json_dict(self.status_path, default={})
        if payload:
            return payload
        backup_path = self._backup_path_for("run_status.json")
        if backup_path is None or not backup_path.exists():
            return {}
        backup_payload = read_json_dict(backup_path, default={})
        if backup_payload:
            try:
                write_json_atomic(self.status_path, backup_payload, ensure_ascii=True, indent=2)
                self._emit_runtime_restore_note(
                    kind="run_status",
                    source_path=backup_path,
                    target_path=self.status_path,
                    restored=True,
                )
            except Exception:
                self._emit_runtime_restore_note(
                    kind="run_status",
                    source_path=backup_path,
                    target_path=self.status_path,
                    restored=False,
                )
        return backup_payload

    def read_state(self) -> dict[str, Any]:
        payload = read_json_dict(self.state_path, default={})
        if payload:
            return payload
        backup_path = self._backup_path_for("state.json")
        if backup_path is None or not backup_path.exists():
            return {}
        backup_payload = read_json_dict(backup_path, default={})
        if backup_payload:
            try:
                write_json_atomic(self.state_path, backup_payload, ensure_ascii=True, indent=2)
                self._emit_runtime_restore_note(
                    kind="state",
                    source_path=backup_path,
                    target_path=self.state_path,
                    restored=True,
                )
            except Exception:
                self._emit_runtime_restore_note(
                    kind="state",
                    source_path=backup_path,
                    target_path=self.state_path,
                    restored=False,
                )
        return backup_payload

    def _backup_path_for(self, relative_name: str) -> Path | None:
        backup_root = self.paths.jobs_backup_root
        if backup_root is None:
            return None
        return backup_root / self.job_id / relative_name

    def _runtime_backup_min_interval_seconds(self) -> float:
        storage_cfg = self.config.get("storage", {}) if isinstance(self.config.get("storage"), dict) else {}
        return max(0.0, _as_float(storage_cfg.get("runtime_state_backup_min_interval_seconds", 30.0), 30.0))

    def _runtime_backup_force_interval_seconds(self) -> float:
        storage_cfg = self.config.get("storage", {}) if isinstance(self.config.get("storage"), dict) else {}
        return max(0.0, _as_float(storage_cfg.get("runtime_state_backup_force_interval_seconds", 300.0), 300.0))

    def _write_backup_json(self, relative_name: str, payload: dict[str, Any]) -> None:
        backup_path = self._backup_path_for(relative_name)
        if backup_path is None:
            return
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(backup_path, payload, ensure_ascii=True, indent=2)
        except Exception as exc:
            self.logger.warning(
                "runtime backup write failed | job_id=%s | file=%s | error=%s",
                self.job_id,
                str(backup_path),
                f"{type(exc).__name__}: {exc}",
            )

    def _append_backup_jsonl(self, relative_name: str, payload: dict[str, Any]) -> None:
        backup_path = self._backup_path_for(relative_name)
        if backup_path is None:
            return
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with backup_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
        except Exception as exc:
            self.logger.warning(
                "runtime backup write failed | job_id=%s | file=%s | error=%s",
                self.job_id,
                str(backup_path),
                f"{type(exc).__name__}: {exc}",
            )

    def _write_backup_text(self, relative_name: str, text: str) -> None:
        backup_path = self._backup_path_for(relative_name)
        if backup_path is None:
            return
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(backup_path, text, encoding="utf-8")
        except Exception as exc:
            self.logger.warning(
                "runtime backup write failed | job_id=%s | file=%s | error=%s",
                self.job_id,
                str(backup_path),
                f"{type(exc).__name__}: {exc}",
            )

    def _emit_runtime_restore_note(self, *, kind: str, source_path: Path, target_path: Path, restored: bool) -> None:
        message = "runtime_backup_restored" if restored else "runtime_backup_restore_copy_failed"
        self.logger.info(
            "%s | job_id=%s | kind=%s | source=%s | target=%s",
            message,
            self.job_id,
            kind,
            str(source_path),
            str(target_path),
        )
        try:
            self.write_event(
                message,
                kind=str(kind),
                source_path=str(source_path),
                target_path=str(target_path),
                restored=bool(restored),
            )
        except Exception:
            pass
        try:
            self.write_metric(
                {
                    "metric_type": message,
                    "kind": str(kind),
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "restored": bool(restored),
                }
            )
        except Exception:
            pass

    def write_event(self, message: str, **extra: Any) -> None:
        payload = {
            "timestamp": utc_now_iso(),
            "job_id": self.job_id,
            "run_type": self.run_type,
            "level": "INFO",
            "message": message,
            **extra,
        }
        self.events.write(payload)
        self._append_backup_jsonl("events.jsonl", payload)

    def write_metric(self, metric: dict[str, Any]) -> None:
        payload = {
            "timestamp": utc_now_iso(),
            "job_id": self.job_id,
            **metric,
        }
        self.metrics.write(payload)
        self._append_backup_jsonl("metrics.jsonl", payload)

    def write_status(self, status: str, *, phase: str = "initializing", extra: dict[str, Any] | None = None) -> None:
        previous = self.read_status()
        payload = {
            "job_id": self.job_id,
            "run_type": self.run_type,
            "status": status,
            "phase": phase,
            "created_at": previous.get("created_at", utc_now_iso()),
            "updated_at": utc_now_iso(),
            "resume_supported": True,
        }
        if extra:
            payload.update(extra)
        write_json_atomic(self.status_path, payload, ensure_ascii=True, indent=2)
        self._write_backup_json("run_status.json", payload)
        status_normalized = str(status).strip().lower()
        if status_normalized in {"completed", "failed", "cancelled"}:
            current_state = read_json_dict(self.state_path, default={})
            if current_state:
                self._write_backup_json("state.json", current_state)
        self._write_worker_checkpoint_firestore(status_payload=payload, state_payload=None, force=True)
        if status_normalized in {"completed", "failed", "cancelled"}:
            self._emit_firestore_sync_summary(status=status_normalized, phase=str(phase))

    def write_state(self, state: dict[str, Any]) -> None:
        payload = {
            "job_id": self.job_id,
            "run_type": self.run_type,
            **state,
        }
        write_json_atomic(self.state_path, payload, ensure_ascii=True, indent=2)
        state_signature = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        now_ts = time.time()
        min_interval = self._runtime_backup_min_interval_seconds()
        force_interval = self._runtime_backup_force_interval_seconds()
        if state_signature != self._last_runtime_backup_state_signature:
            self._runtime_backup_state_dirty = True
            self._last_runtime_backup_state_change_ts = now_ts

        elapsed_since_backup = now_ts - float(self._last_runtime_backup_state_write_ts)
        should_backup = bool(self._runtime_backup_state_dirty)
        if should_backup and min_interval > 0.0 and elapsed_since_backup < min_interval:
            should_backup = False
        if bool(self._runtime_backup_state_dirty) and not should_backup and force_interval > 0.0:
            should_backup = elapsed_since_backup >= force_interval

        if should_backup:
            self._write_backup_json("state.json", payload)
            self._last_runtime_backup_state_signature = state_signature
            self._last_runtime_backup_state_write_ts = now_ts
            self._runtime_backup_state_dirty = False
        self._write_worker_checkpoint_firestore(status_payload=None, state_payload=payload, force=False)

    def set_phase(self, phase: str) -> None:
        current = read_json_dict(self.status_path, default={})
        current.update({"phase": phase, "updated_at": utc_now_iso()})
        write_json_atomic(self.status_path, current, ensure_ascii=True, indent=2)
        self._write_backup_json("run_status.json", current)
        self._write_worker_checkpoint_firestore(status_payload=current, state_payload=None, force=True)

    def _write_worker_checkpoint_firestore(
        self,
        *,
        status_payload: dict[str, Any] | None,
        state_payload: dict[str, Any] | None,
        force: bool = False,
    ) -> None:
        sync = self.firestore_sync if isinstance(self.firestore_sync, dict) else {}
        if not bool(sync.get("enabled", False)):
            return
        strict = bool(sync.get("strict", True))
        if self._firestore_sync_stats is None:
            self._firestore_sync_stats = {
                "attempted": 0,
                "written": 0,
                "skipped_unchanged": 0,
                "skipped_min_interval": 0,
                "failed": 0,
            }
        min_interval_seconds = max(0.0, _as_float(sync.get("checkpoint_min_interval_seconds", 30.0), 30.0))
        state_only_on_change = _as_bool(sync.get("checkpoint_state_only_on_change", True), default=True)
        status_signature = json.dumps(status_payload, sort_keys=True, ensure_ascii=True, default=str) if status_payload is not None else ""
        state_signature = json.dumps(state_payload, sort_keys=True, ensure_ascii=True, default=str) if state_payload is not None else ""
        now_ts = time.time()
        if not force:
            if state_payload is not None and state_only_on_change and state_signature == self._last_firestore_state_signature:
                self._firestore_sync_stats["skipped_unchanged"] = int(self._firestore_sync_stats.get("skipped_unchanged", 0)) + 1
                return
            if min_interval_seconds > 0 and (now_ts - float(self._last_firestore_checkpoint_write_ts)) < min_interval_seconds:
                self._firestore_sync_stats["skipped_min_interval"] = int(self._firestore_sync_stats.get("skipped_min_interval", 0)) + 1
                return
        self._firestore_sync_stats["attempted"] = int(self._firestore_sync_stats.get("attempted", 0)) + 1
        try:
            client = _build_firestore_job_client(
                project_id=str(sync.get("project_id", "")).strip(),
                credentials_path=str(sync.get("credentials_path", "")).strip(),
                api_key=str(sync.get("api_key", "")).strip(),
            )
            collection = str(sync.get("collection", "worker_checkpoints")).strip() or "worker_checkpoints"
            doc_ref = client.collection(collection).document(self.job_id)
            payload: dict[str, Any] = {}
            payload["job_id"] = self.job_id
            payload["run_type"] = self.run_type
            payload["updated_at"] = utc_now_iso()
            if status_payload is not None:
                payload["status"] = dict(status_payload)
                payload["phase"] = str(status_payload.get("phase", ""))
            if state_payload is not None:
                payload["state"] = dict(state_payload)
            doc_ref.set(payload, merge=True)
            self._last_firestore_checkpoint_write_ts = now_ts
            if status_payload is not None:
                self._last_firestore_status_signature = status_signature
            if state_payload is not None:
                self._last_firestore_state_signature = state_signature
            self._firestore_sync_stats["written"] = int(self._firestore_sync_stats.get("written", 0)) + 1
        except Exception as exc:
            self._firestore_sync_stats["failed"] = int(self._firestore_sync_stats.get("failed", 0)) + 1
            diag = _firestore_sync_diag(sync)
            hint = _firestore_error_hint(exc, auth_mode=str(diag.get("auth_mode", "")))
            self.logger.warning(
                "firestore worker checkpoint sync failed | job_id=%s | strict=%s | project_id=%s | collection=%s | auth_mode=%s | credentials_path_exists=%s | api_key_set=%s | error=%s | hint=%s",
                self.job_id,
                strict,
                str(diag.get("project_id", "")) or "<empty>",
                str(diag.get("collection", "")) or "<empty>",
                str(diag.get("auth_mode", "")) or "<unknown>",
                bool(diag.get("credentials_path_exists", False)),
                bool(diag.get("api_key_set", False)),
                f"{type(exc).__name__}: {exc}",
                hint or "<none>",
            )
            try:
                self.write_event(
                    "firestore_worker_checkpoint_sync_failed",
                    strict=strict,
                    project_id=str(diag.get("project_id", "")),
                    collection=str(diag.get("collection", "")),
                    auth_mode=str(diag.get("auth_mode", "")),
                    credentials_path_exists=bool(diag.get("credentials_path_exists", False)),
                    api_key_set=bool(diag.get("api_key_set", False)),
                    error=f"{type(exc).__name__}: {exc}",
                    hint=hint or "",
                )
            except Exception:
                pass
            if strict:
                raise

    def _emit_firestore_sync_summary(self, *, status: str, phase: str) -> None:
        sync = self.firestore_sync if isinstance(self.firestore_sync, dict) else {}
        stats = self._firestore_sync_stats or {}
        diag = _firestore_sync_diag(sync)
        self.logger.info(
            "firestore checkpoint sync summary | job_id=%s | status=%s | phase=%s | enabled=%s | strict=%s | attempted=%s | written=%s | skipped_unchanged=%s | skipped_min_interval=%s | failed=%s | project_id=%s | collection=%s | auth_mode=%s",
            self.job_id,
            status,
            phase,
            bool(diag.get("enabled", False)),
            bool(diag.get("strict", True)),
            int(stats.get("attempted", 0)),
            int(stats.get("written", 0)),
            int(stats.get("skipped_unchanged", 0)),
            int(stats.get("skipped_min_interval", 0)),
            int(stats.get("failed", 0)),
            str(diag.get("project_id", "")) or "<empty>",
            str(diag.get("collection", "")) or "<empty>",
            str(diag.get("auth_mode", "")) or "<unknown>",
        )
        try:
            self.write_metric(
                {
                    "metric_type": "firestore_checkpoint_sync_summary",
                    "status": status,
                    "phase": phase,
                    "sync_enabled": bool(diag.get("enabled", False)),
                    "sync_strict": bool(diag.get("strict", True)),
                    "attempted": int(stats.get("attempted", 0)),
                    "written": int(stats.get("written", 0)),
                    "skipped_unchanged": int(stats.get("skipped_unchanged", 0)),
                    "skipped_min_interval": int(stats.get("skipped_min_interval", 0)),
                    "failed": int(stats.get("failed", 0)),
                    "project_id": str(diag.get("project_id", "")),
                    "collection": str(diag.get("collection", "")),
                    "auth_mode": str(diag.get("auth_mode", "")),
                }
            )
        except Exception:
            pass


def create_job_context(config: dict[str, Any], *, override_job_id: str | None = None) -> JobContext:
    paths = build_project_paths(config)
    required_roots = [paths.jobs_root, paths.logs_root, paths.reports_root, paths.models_root, paths.data_root]
    if paths.jobs_backup_root is not None:
        required_roots.append(paths.jobs_backup_root)
    for path in required_roots:
        path.mkdir(parents=True, exist_ok=True)

    job_cfg = config.get("job", {})
    run_type = str(job_cfg.get("run_type", "job"))
    requested_job_id = override_job_id or str(job_cfg.get("job_id", "auto"))
    job_id = make_job_id(run_type) if requested_job_id in {"", "auto"} else requested_job_id
    auto_rollover_completed = bool(job_cfg.get("auto_rollover_completed_job", True))
    if requested_job_id not in {"", "auto"} and auto_rollover_completed and run_type in {
        "train",
        "evaluation",
        "benchmark",
        "dataset_generation",
        "dataset_build",
    }:
        existing_status_paths = [paths.jobs_root / job_id / "run_status.json"]
        if paths.jobs_backup_root is not None:
            existing_status_paths.append(paths.jobs_backup_root / job_id / "run_status.json")
        existing_status = {}
        for existing_status_path in existing_status_paths:
            if not existing_status_path.exists():
                continue
            existing_status = read_json_dict(existing_status_path, default={})
            if existing_status:
                break
        if str(existing_status.get("status", "")).lower() == "completed":
            job_id = _next_cycle_job_id(requested_job_id, paths.jobs_root, backup_jobs_root=paths.jobs_backup_root)
    job_dir = paths.jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    backup_job_dir: Path | None = None
    if paths.jobs_backup_root is not None:
        backup_job_dir = paths.jobs_backup_root / job_id
        backup_job_dir.mkdir(parents=True, exist_ok=True)

    restored_from_backup: list[str] = []
    if backup_job_dir is not None:
        for relative_name in ["config.yaml", "run_status.json", "state.json", "events.jsonl", "metrics.jsonl"]:
            local_path = job_dir / relative_name
            backup_path = backup_job_dir / relative_name
            if local_path.exists() or not backup_path.exists():
                continue
            try:
                write_text_atomic(local_path, backup_path.read_text(encoding="utf-8"), encoding="utf-8")
                restored_from_backup.append(relative_name)
            except Exception:
                pass

    logger = build_console_logger(job_id)
    events = JsonlWriter(job_dir / "events.jsonl")
    metrics = JsonlWriter(job_dir / "metrics.jsonl")
    status_path = job_dir / "run_status.json"
    state_path = job_dir / "state.json"

    context = JobContext(
        config=config,
        paths=paths,
        job_id=job_id,
        run_type=run_type,
        job_dir=job_dir,
        logger=logger,
        events=events,
        metrics=metrics,
        status_path=status_path,
        state_path=state_path,
        firestore_sync=_resolve_firestore_sync_config(config),
    )
    sync_diag = _firestore_sync_diag(context.firestore_sync)
    context.logger.info(
        "firestore checkpoint sync config | enabled=%s | strict=%s | project_id=%s | collection=%s | auth_mode=%s | credentials_path_exists=%s | api_key_set=%s | checkpoint_min_interval_seconds=%.2f | checkpoint_state_only_on_change=%s",
        bool(sync_diag.get("enabled", False)),
        bool(sync_diag.get("strict", True)),
        str(sync_diag.get("project_id", "")) or "<empty>",
        str(sync_diag.get("collection", "")) or "<empty>",
        str(sync_diag.get("auth_mode", "")) or "<unknown>",
        bool(sync_diag.get("credentials_path_exists", False)),
        bool(sync_diag.get("api_key_set", False)),
        float(sync_diag.get("checkpoint_min_interval_seconds", 0.0)),
        bool(sync_diag.get("checkpoint_state_only_on_change", True)),
    )
    context.write_event(
        "firestore_checkpoint_sync_config_resolved",
        enabled=bool(sync_diag.get("enabled", False)),
        strict=bool(sync_diag.get("strict", True)),
        project_id=str(sync_diag.get("project_id", "")),
        collection=str(sync_diag.get("collection", "")),
        auth_mode=str(sync_diag.get("auth_mode", "")),
        credentials_path_exists=bool(sync_diag.get("credentials_path_exists", False)),
        api_key_set=bool(sync_diag.get("api_key_set", False)),
        checkpoint_min_interval_seconds=float(sync_diag.get("checkpoint_min_interval_seconds", 0.0)),
        checkpoint_state_only_on_change=bool(sync_diag.get("checkpoint_state_only_on_change", True)),
    )
    if restored_from_backup:
        context.logger.info(
            "runtime backup restored at startup | job_id=%s | files=%s | backup_root=%s",
            context.job_id,
            ",".join(restored_from_backup),
            str(backup_job_dir) if backup_job_dir is not None else "<none>",
        )
        context.write_event(
            "runtime_backup_restored_startup",
            files=list(restored_from_backup),
            backup_root=str(backup_job_dir) if backup_job_dir is not None else "",
        )
        context.write_metric(
            {
                "metric_type": "runtime_backup_restored_startup",
                "files_count": int(len(restored_from_backup)),
            }
        )

    write_text_atomic(job_dir / "config.yaml", _dump_yaml_like(config), encoding="utf-8")
    context._write_backup_text("config.yaml", _dump_yaml_like(config))
    if not status_path.exists():
        context.write_status("pending")
    if not state_path.exists():
        context.write_state({"last_completed_phase": "none"})
    if job_id != requested_job_id and requested_job_id not in {"", "auto"}:
        context.logger.info(
            "completed job id rolled over automatically | previous_job_id=%s | new_job_id=%s | run_type=%s",
            requested_job_id,
            job_id,
            run_type,
        )
        context.write_event(
            "job_id_rollover_completed_job",
            previous_job_id=requested_job_id,
            new_job_id=job_id,
        )
    return context


def _redact_sensitive_config(value: Any, key_path: str = "") -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.strip().lower()
            is_sensitive = (
                "token" in key_lower
                or "secret" in key_lower
                or "password" in key_lower
                or "private_key" in key_lower
                or key_lower.endswith("api_key")
            )
            if is_sensitive:
                redacted[key_text] = "***REDACTED***"
            else:
                redacted[key_text] = _redact_sensitive_config(child, key_text if not key_path else f"{key_path}.{key_text}")
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_config(item, key_path) for item in value]
    return value


def _dump_yaml_like(config: dict[str, Any]) -> str:
    safe_config = _redact_sensitive_config(config)
    try:
        import yaml
    except ImportError:
        return json.dumps(safe_config, indent=2)
    return yaml.safe_dump(safe_config, sort_keys=False)
