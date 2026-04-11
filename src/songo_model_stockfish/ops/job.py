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


def _next_cycle_job_id(requested_job_id: str, jobs_root: Path) -> str:
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
            if not (jobs_root / next_job_id).exists():
                return next_job_id

    index = 2
    while True:
        next_job_id = f"{requested_job_id}_{index:03d}"
        if not (jobs_root / next_job_id).exists():
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

    def read_status(self) -> dict[str, Any]:
        return read_json_dict(self.status_path, default={})

    def read_state(self) -> dict[str, Any]:
        return read_json_dict(self.state_path, default={})

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

    def write_metric(self, metric: dict[str, Any]) -> None:
        payload = {
            "timestamp": utc_now_iso(),
            "job_id": self.job_id,
            **metric,
        }
        self.metrics.write(payload)

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
        self._write_worker_checkpoint_firestore(status_payload=payload, state_payload=None, force=True)

    def write_state(self, state: dict[str, Any]) -> None:
        payload = {
            "job_id": self.job_id,
            "run_type": self.run_type,
            **state,
        }
        write_json_atomic(self.state_path, payload, ensure_ascii=True, indent=2)
        self._write_worker_checkpoint_firestore(status_payload=None, state_payload=payload, force=False)

    def set_phase(self, phase: str) -> None:
        current = read_json_dict(self.status_path, default={})
        current.update({"phase": phase, "updated_at": utc_now_iso()})
        write_json_atomic(self.status_path, current, ensure_ascii=True, indent=2)
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
        min_interval_seconds = max(0.0, _as_float(sync.get("checkpoint_min_interval_seconds", 30.0), 30.0))
        state_only_on_change = _as_bool(sync.get("checkpoint_state_only_on_change", True), default=True)
        status_signature = json.dumps(status_payload, sort_keys=True, ensure_ascii=True, default=str) if status_payload is not None else ""
        state_signature = json.dumps(state_payload, sort_keys=True, ensure_ascii=True, default=str) if state_payload is not None else ""
        now_ts = time.time()
        if not force:
            if state_payload is not None and state_only_on_change and state_signature == self._last_firestore_state_signature:
                return
            if min_interval_seconds > 0 and (now_ts - float(self._last_firestore_checkpoint_write_ts)) < min_interval_seconds:
                return
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
        except Exception as exc:
            self.logger.warning("firestore worker checkpoint sync failed | job_id=%s | error=%s", self.job_id, f"{type(exc).__name__}: {exc}")
            if strict:
                raise


def create_job_context(config: dict[str, Any], *, override_job_id: str | None = None) -> JobContext:
    paths = build_project_paths(config)
    for path in [paths.jobs_root, paths.logs_root, paths.reports_root, paths.models_root, paths.data_root]:
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
        existing_status_path = paths.jobs_root / job_id / "run_status.json"
        if existing_status_path.exists():
            existing_status = read_json_dict(existing_status_path, default={})
            if str(existing_status.get("status", "")).lower() == "completed":
                job_id = _next_cycle_job_id(requested_job_id, paths.jobs_root)
    job_dir = paths.jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

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

    write_text_atomic(job_dir / "config.yaml", _dump_yaml_like(config), encoding="utf-8")
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
