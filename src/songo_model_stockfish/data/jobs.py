from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import json
import math
import multiprocessing
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from songo_model_stockfish.adapters import songo_ai_game
from songo_model_stockfish.ops.io_utils import (
    acquire_lock_dir,
    guard_write_path,
    read_json_dict,
    release_lock_dir,
    resolve_allowed_drive_root,
    write_json_atomic,
    write_jsonl_atomic,
)
from songo_model_stockfish.ops.job import JobContext
from songo_model_stockfish.ops.logging import utc_now_iso
from songo_model_stockfish.ops.model_registry import latest_model_record, load_registry, promoted_best_metadata
from songo_model_stockfish.training.features import adapt_feature_dim, build_runtime_tactical_analysis, encode_model_features


def _slugify_matchup(matchup_spec: str) -> str:
    return matchup_spec.replace(":", "_").replace(" ", "").replace("/", "_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload, ensure_ascii=True, indent=2)


def _write_npz_compressed(path: Path, **arrays: Any) -> None:
    guard_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: OSError | None = None
    for attempt in range(3):
        try:
            np.savez_compressed(path, **arrays)
            return
        except FileNotFoundError as exc:
            # Google Drive FUSE can transiently lose parent directories during long runs.
            last_exc = exc
            path.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(0.02 * (attempt + 1))
        except OSError as exc:
            last_exc = exc
            path.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(0.02 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise OSError(f"Echec ecriture NPZ: {path}")


def _path_within(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _resolve_storage_path(base: Path, configured: str | None, fallback: Path) -> Path:
    if not configured:
        return fallback
    path = Path(configured).expanduser()
    mydrive_root = Path("/content/drive/MyDrive")
    allowed_drive_root = resolve_allowed_drive_root()
    if path.is_absolute():
        if _path_within(path, mydrive_root) and not _path_within(path, allowed_drive_root):
            raise ValueError(
                "Chemin storage absolu refuse (hors drive_root autorise). "
                f"configured={path} | allowed_drive_root={allowed_drive_root}"
            )
        return path
    resolved = base / path
    if _path_within(resolved, mydrive_root) and not _path_within(resolved, allowed_drive_root):
        raise ValueError(
            "Chemin storage relatif refuse (resolution hors drive_root autorise). "
            f"configured={configured} | resolved={resolved} | allowed_drive_root={allowed_drive_root}"
        )
    return resolved


def _normalize_completed_game_detection_mode(value: Any) -> str:
    mode = str(value or "raw_and_sampled").strip().lower()
    if mode not in {"raw_and_sampled", "sampled_only", "raw_only"}:
        return "raw_and_sampled"
    return mode


def _default_raw_dir_name_for_dataset_source(dataset_source_id: str) -> str:
    if dataset_source_id.startswith("sampled_"):
        return "raw_" + dataset_source_id[len("sampled_") :]
    if dataset_source_id.startswith("data/"):
        leaf = Path(dataset_source_id).name
        return _default_raw_dir_name_for_dataset_source(leaf)
    return f"raw_{dataset_source_id}"


def _read_json_file(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return read_json_dict(path, default=default, retries=8, wait_seconds=0.05)


def _acquire_lock_dir(lock_dir: Path, timeout_seconds: float = 30.0, poll_seconds: float = 0.1) -> bool:
    return acquire_lock_dir(lock_dir, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)


def _release_lock_dir(lock_dir: Path) -> None:
    release_lock_dir(lock_dir)


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


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_iso_to_epoch_seconds(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0.0


def _apply_worker_progress_retention(
    *,
    workers: dict[str, dict[str, Any]],
    keep_job_id: str,
    stale_seconds: int,
    max_entries: int,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    now_ts = time.time()
    keep = str(keep_job_id).strip()
    normalized: dict[str, dict[str, Any]] = {}
    dropped_samples = 0
    dropped_games = 0
    ttl_seconds = max(0, int(stale_seconds))
    limit = max(1, int(max_entries))

    for worker_job_id, payload in (workers or {}).items():
        entry = _normalize_worker_progress_entry(job_id=str(worker_job_id), payload=payload)
        if entry is None:
            continue
        if ttl_seconds > 0 and entry["job_id"] != keep:
            updated_ts = _parse_iso_to_epoch_seconds(entry.get("updated_at"))
            if updated_ts > 0 and (now_ts - updated_ts) > float(ttl_seconds):
                dropped_samples += max(0, int(entry.get("contributed_samples", 0)))
                dropped_games += max(0, int(entry.get("contributed_games", 0)))
                continue
        normalized[entry["job_id"]] = {
            "dataset_source_id": str(entry.get("dataset_source_id", "")),
            "contributed_samples": max(0, int(entry.get("contributed_samples", 0))),
            "contributed_games": max(0, int(entry.get("contributed_games", 0))),
            "updated_at": str(entry.get("updated_at", "")),
        }

    if len(normalized) <= limit:
        return normalized, dropped_samples, dropped_games

    sorted_entries = sorted(
        normalized.items(),
        key=lambda item: (_parse_iso_to_epoch_seconds(item[1].get("updated_at")), item[0]),
        reverse=True,
    )
    allowed_job_ids = {job_id for job_id, _payload in sorted_entries[:limit]}
    if keep and keep in normalized:
        allowed_job_ids.add(keep)

    retained: dict[str, dict[str, Any]] = {}
    for job_id, payload in normalized.items():
        if job_id in allowed_job_ids:
            retained[job_id] = payload
            continue
        dropped_samples += max(0, int(payload.get("contributed_samples", 0)))
        dropped_games += max(0, int(payload.get("contributed_games", 0)))

    return retained, dropped_samples, dropped_games


def _resolve_global_progress_backend_config(
    *,
    cfg: dict[str, Any],
    global_target_id: str,
    target_samples: int,
) -> dict[str, Any]:
    backend_raw = str(
        cfg.get(
            "global_progress_backend",
            cfg.get(
                "global_target_progress_backend",
                "file",
            ),
        )
    ).strip().lower() or "file"
    firestore_enabled_flag = _as_bool(
        cfg.get(
            "global_progress_firestore_enabled",
            cfg.get("global_target_progress_firestore_enabled", False),
        ),
        default=False,
    )
    use_firestore = backend_raw == "firestore" or firestore_enabled_flag
    project_id = str(
        cfg.get(
            "global_progress_firestore_project_id",
            cfg.get(
                "global_target_progress_firestore_project_id",
                os.environ.get("FIREBASE_PROJECT_ID", "") or os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            ),
        )
    ).strip()
    collection = str(
        cfg.get(
            "global_progress_firestore_collection",
            cfg.get("global_target_progress_firestore_collection", "global_generation_progress"),
        )
    ).strip() or "global_generation_progress"
    document = str(
        cfg.get(
            "global_progress_firestore_document",
            cfg.get("global_target_progress_firestore_document", global_target_id),
        )
    ).strip() or global_target_id
    credentials_path = str(
        cfg.get(
            "global_progress_firestore_credentials_path",
            cfg.get("global_target_progress_firestore_credentials_path", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")),
        )
    ).strip()
    api_key = str(
        cfg.get(
            "global_progress_firestore_api_key",
            cfg.get("global_target_progress_firestore_api_key", os.environ.get("FIREBASE_API_KEY", "")),
        )
    ).strip()
    redis_enabled = _as_bool(
        cfg.get(
            "global_progress_redis_enabled",
            cfg.get("global_target_progress_redis_enabled", False),
        ),
        default=False,
    )
    redis_url = str(
        cfg.get(
            "global_progress_redis_url",
            cfg.get(
                "global_target_progress_redis_url",
                os.environ.get("UPSTASH_REDIS_REST_URL", ""),
            ),
        )
    ).strip()
    redis_token = str(
        cfg.get(
            "global_progress_redis_token",
            cfg.get(
                "global_target_progress_redis_token",
                os.environ.get("UPSTASH_REDIS_REST_TOKEN", ""),
            ),
        )
    ).strip()
    redis_key_prefix = str(
        cfg.get(
            "global_progress_redis_key_prefix",
            cfg.get("global_target_progress_redis_key_prefix", f"songo:{global_target_id}"),
        )
    ).strip() or f"songo:{global_target_id}"
    redis_cache_ttl_seconds = max(
        1,
        _safe_int(
            cfg.get(
                "global_progress_redis_cache_ttl_seconds",
                cfg.get("global_target_progress_redis_cache_ttl_seconds", 120),
            ),
            120,
        ),
    )
    workers_retention_seconds = max(
        0,
        _safe_int(
            cfg.get(
                "global_progress_workers_retention_seconds",
                cfg.get("global_target_progress_workers_retention_seconds", 86400),
            ),
            86400,
        ),
    )
    workers_max_entries = max(
        1,
        _safe_int(
            cfg.get(
                "global_progress_workers_max_entries",
                cfg.get("global_target_progress_workers_max_entries", 5000),
            ),
            5000,
        ),
    )
    return {
        "backend": "firestore" if use_firestore else "file",
        "global_target_id": str(global_target_id),
        "target_samples": int(target_samples),
        "firestore_project_id": project_id,
        "firestore_collection": collection,
        "firestore_document": document,
        "firestore_credentials_path": credentials_path,
        "firestore_api_key": api_key,
        "redis_enabled": bool(redis_enabled),
        "redis_url": redis_url,
        "redis_token": redis_token,
        "redis_key_prefix": redis_key_prefix,
        "redis_cache_ttl_seconds": int(redis_cache_ttl_seconds),
        "workers_retention_seconds": int(workers_retention_seconds),
        "workers_max_entries": int(workers_max_entries),
    }


def _normalize_global_generation_state_payload(
    *,
    payload: Any,
    global_target_id: str,
    target_samples: int,
) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    state = _default_global_generation_progress_state(global_target_id, target_samples)
    state["global_target_id"] = str(raw.get("global_target_id", global_target_id) or global_target_id)
    state["target_samples"] = int(raw.get("target_samples", target_samples) or target_samples or 0)
    raw_workers = raw.get("workers")
    workers: dict[str, dict[str, Any]] = {}
    if isinstance(raw_workers, dict):
        for worker_job_id, worker_payload in raw_workers.items():
            normalized = _normalize_worker_progress_entry(job_id=str(worker_job_id), payload=worker_payload)
            if normalized is None:
                continue
            workers[normalized["job_id"]] = {
                "dataset_source_id": str(normalized.get("dataset_source_id", "")),
                "contributed_samples": int(normalized.get("contributed_samples", 0)),
                "contributed_games": int(normalized.get("contributed_games", 0)),
                "updated_at": str(normalized.get("updated_at", "")),
            }
    archived_samples = max(0, int(raw.get("archived_samples", 0) or 0))
    archived_games = max(0, int(raw.get("archived_games", 0) or 0))
    worker_total_samples = sum(int(item.get("contributed_samples", 0)) for item in workers.values())
    worker_total_games = sum(int(item.get("contributed_games", 0)) for item in workers.values())
    state["workers"] = workers
    state["archived_samples"] = archived_samples
    state["archived_games"] = archived_games
    state["total_samples"] = max(
        int(raw.get("total_samples", 0) or 0),
        int(archived_samples + worker_total_samples),
    )
    state["total_games"] = max(
        int(raw.get("total_games", 0) or 0),
        int(archived_games + worker_total_games),
    )
    state["updated_at"] = str(raw.get("updated_at", "")).strip() or utc_now_iso()
    return state


def _firestore_backend_diagnostics(progress_backend: dict[str, Any] | None) -> dict[str, Any]:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    credentials_path = str(backend.get("firestore_credentials_path", "")).strip()
    api_key = str(backend.get("firestore_api_key", "")).strip()
    auth_mode = "missing_credentials"
    if credentials_path:
        auth_mode = "service_account_file"
    elif api_key:
        auth_mode = "api_key_anonymous"
    return {
        "backend": str(backend.get("backend", "file")).strip().lower() or "file",
        "project_id": str(backend.get("firestore_project_id", "")).strip(),
        "collection": str(backend.get("firestore_collection", "")).strip(),
        "document": str(backend.get("firestore_document", "")).strip(),
        "global_target_id": str(backend.get("global_target_id", "")).strip(),
        "auth_mode": auth_mode,
        "credentials_path": credentials_path,
        "credentials_path_exists": bool(Path(credentials_path).exists()) if credentials_path else False,
        "api_key_set": bool(api_key),
        "redis_enabled": bool(backend.get("redis_enabled", False)),
        "redis_url_set": bool(str(backend.get("redis_url", "")).strip()),
        "redis_token_set": bool(str(backend.get("redis_token", "")).strip()),
        "workers_retention_seconds": int(_safe_int(backend.get("workers_retention_seconds", 0), 0)),
        "workers_max_entries": int(_safe_int(backend.get("workers_max_entries", 0), 0)),
    }


def _firestore_error_hint(*, exc: Exception | None, diagnostics: dict[str, Any]) -> str:
    exc_text = ""
    if exc is not None:
        exc_text = f"{type(exc).__name__}: {exc}".lower()
    auth_mode = str(diagnostics.get("auth_mode", "")).strip().lower()
    if auth_mode == "api_key_anonymous":
        return (
            "Firestore Python ne supporte pas API key seule; configure "
            "`global_progress_firestore_credentials_path`."
        )
    if auth_mode == "missing_credentials":
        return (
            "Credentials Firestore absents; configure "
            "`global_progress_firestore_credentials_path`."
        )
    if auth_mode == "adc" and ("metadata.google.internal" in exc_text or "compute engine metadata" in exc_text):
        return (
            "Auth ADC indisponible dans ce runtime; configure "
            "`global_progress_firestore_credentials_path`."
        )
    if "permissiondenied" in exc_text or "permission denied" in exc_text:
        return "Acces refuse par les regles Firestore (verifie Rules et role du compte)."
    if "unauthenticated" in exc_text or "invalid authentication credentials" in exc_text:
        return "Authentification Firestore invalide (api key / credentials)."
    if "deadlineexceeded" in exc_text or "timeout" in exc_text:
        return "Timeout reseau vers Firestore (latence/reseau)."
    if "serviceunavailable" in exc_text or "temporarily unavailable" in exc_text:
        return "Service Firestore indisponible temporairement; reessayer."
    if "notfound" in exc_text and "project" in exc_text:
        return "Projet Firestore introuvable (verifie `firestore_project_id`)."
    if diagnostics.get("auth_mode") == "service_account_file" and not diagnostics.get("credentials_path_exists"):
        return "Le fichier credentials n'existe pas au chemin configure."
    return ""


def _raise_firestore_progress_error(
    *,
    stage: str,
    progress_backend: dict[str, Any] | None,
    exc: Exception | None = None,
    details: str = "",
) -> None:
    diagnostics = _firestore_backend_diagnostics(progress_backend)
    message_parts = [
        f"Firestore global progress error",
        f"stage={stage}",
        f"backend={diagnostics.get('backend', '')}",
        f"project_id={diagnostics.get('project_id', '') or '<empty>'}",
        f"collection={diagnostics.get('collection', '') or '<empty>'}",
        f"document={diagnostics.get('document', '') or '<empty>'}",
        f"auth_mode={diagnostics.get('auth_mode', '') or '<unknown>'}",
        f"credentials_path_exists={diagnostics.get('credentials_path_exists', False)}",
        f"api_key_set={diagnostics.get('api_key_set', False)}",
    ]
    if details:
        message_parts.append(f"details={details}")
    if exc is not None:
        message_parts.append(f"cause={type(exc).__name__}: {exc}")
    hint = _firestore_error_hint(exc=exc, diagnostics=diagnostics)
    if hint:
        message_parts.append(f"hint={hint}")
    message = " | ".join(message_parts)
    if exc is not None:
        raise RuntimeError(message) from exc
    raise RuntimeError(message)


@functools.lru_cache(maxsize=16)
def _build_firestore_progress_endpoint(
    project_id: str,
    credentials_path: str,
    api_key: str,
    collection: str,
    document: str,
) -> tuple[Any, Any, Any]:
    try:
        from google.cloud import firestore
    except Exception as exc:
        raise RuntimeError("Import `google.cloud.firestore` impossible.") from exc

    credentials = None
    client_options = None
    if credentials_path:
        if not Path(credentials_path).exists():
            raise FileNotFoundError(f"Fichier credentials Firestore introuvable: {credentials_path}")
        try:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        except Exception as exc:
            raise RuntimeError(f"Chargement credentials Firestore impossible: {credentials_path}") from exc
    elif api_key:
        raise RuntimeError(
            "Mode API key non supporte avec google-cloud-firestore; "
            "configure `global_progress_firestore_credentials_path`."
        )
    else:
        raise RuntimeError(
            "Credentials Firestore absents; configure `global_progress_firestore_credentials_path`."
        )
    try:
        client = firestore.Client(project=(project_id or None), credentials=credentials, client_options=client_options)
    except Exception as exc:
        raise RuntimeError("Creation client Firestore impossible.") from exc
    doc_ref = client.collection(collection).document(document)
    return client, doc_ref, firestore


@functools.lru_cache(maxsize=16)
def _build_redis_progress_client(redis_url: str, redis_token: str):
    url = str(redis_url).strip()
    token = str(redis_token).strip()
    if not url or not token:
        return None
    from upstash_redis import Redis

    return Redis(url=url, token=token)


def _resolve_redis_progress_client(progress_backend: dict[str, Any] | None):
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    if not bool(backend.get("redis_enabled", False)):
        return None
    redis_url = str(backend.get("redis_url", "")).strip()
    redis_token = str(backend.get("redis_token", "")).strip()
    if not redis_url or not redis_token:
        return None
    try:
        return _build_redis_progress_client(redis_url, redis_token)
    except Exception:
        return None


def _redis_global_progress_key(progress_backend: dict[str, Any] | None, global_target_id: str) -> str:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    prefix = str(backend.get("redis_key_prefix", "")).strip() or f"songo:{global_target_id}"
    return f"{prefix}:global_progress"


def _touch_progress_read_telemetry(
    telemetry: dict[str, int] | None,
    key: str,
) -> None:
    if telemetry is None:
        return
    telemetry[key] = int(telemetry.get(key, 0) or 0) + 1


def _read_global_generation_progress_redis_cache(
    *,
    global_target_id: str,
    target_samples: int,
    progress_backend: dict[str, Any] | None = None,
    telemetry: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    client = _resolve_redis_progress_client(backend)
    if client is None:
        return None
    key = _redis_global_progress_key(backend, global_target_id)
    try:
        raw_payload = client.get(key)
    except Exception:
        _touch_progress_read_telemetry(telemetry, "redis_error")
        return None
    if raw_payload is None:
        _touch_progress_read_telemetry(telemetry, "redis_miss")
        return None
    payload: dict[str, Any] | None = None
    if isinstance(raw_payload, dict):
        payload = raw_payload
    elif isinstance(raw_payload, str):
        try:
            parsed = json.loads(raw_payload)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            _touch_progress_read_telemetry(telemetry, "redis_error")
            payload = None
    if payload is None:
        _touch_progress_read_telemetry(telemetry, "redis_error")
        return None
    _touch_progress_read_telemetry(telemetry, "redis_hit")
    return _normalize_global_generation_state_payload(
        payload=payload,
        global_target_id=global_target_id,
        target_samples=target_samples,
    )


def _write_global_generation_progress_redis_cache(
    *,
    global_target_id: str,
    state: dict[str, Any],
    progress_backend: dict[str, Any] | None = None,
) -> None:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    client = _resolve_redis_progress_client(backend)
    if client is None:
        return
    key = _redis_global_progress_key(backend, global_target_id)
    ttl_seconds = max(1, int(backend.get("redis_cache_ttl_seconds", 120) or 120))
    payload = dict(state)
    payload["_redis_cached_at"] = utc_now_iso()
    try:
        client.set(key, json.dumps(payload, ensure_ascii=True), ex=ttl_seconds)
    except Exception:
        return


def _resolve_firestore_progress_endpoint(progress_backend: dict[str, Any] | None) -> tuple[Any, Any, Any] | None:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    if str(backend.get("backend", "file")).strip().lower() != "firestore":
        return None
    project_id = str(backend.get("firestore_project_id", "")).strip()
    collection = str(backend.get("firestore_collection", "global_generation_progress")).strip() or "global_generation_progress"
    document = str(backend.get("firestore_document", backend.get("global_target_id", ""))).strip()
    credentials_path = str(backend.get("firestore_credentials_path", "")).strip()
    api_key = str(backend.get("firestore_api_key", "")).strip()
    if not document:
        _raise_firestore_progress_error(
            stage="resolve_endpoint",
            progress_backend=backend,
            details="firestore_document vide",
        )
    try:
        return _build_firestore_progress_endpoint(project_id, credentials_path, api_key, collection, document)
    except Exception as exc:
        _raise_firestore_progress_error(
            stage="resolve_endpoint",
            progress_backend=backend,
            exc=exc,
        )


def _mirror_global_generation_progress_state(progress_path: Path, state: dict[str, Any]) -> None:
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(progress_path, state, ensure_ascii=True, indent=2)
    except Exception:
        return


def _update_global_generation_progress_firestore(
    *,
    progress_path: Path,
    global_target_id: str,
    target_samples: int,
    job_id: str,
    dataset_source_id: str,
    delta_samples: int = 0,
    delta_games: int = 0,
    progress_backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    workers_retention_seconds = max(0, _safe_int(backend.get("workers_retention_seconds", 86400), 86400))
    workers_max_entries = max(1, _safe_int(backend.get("workers_max_entries", 5000), 5000))
    endpoint = _resolve_firestore_progress_endpoint(backend)
    if endpoint is None:
        _raise_firestore_progress_error(
            stage="resolve_endpoint",
            progress_backend=backend,
            details="endpoint introuvable (backend invalide)",
        )
    client, doc_ref, firestore_mod = endpoint
    transaction = client.transaction()

    @firestore_mod.transactional
    def _tx(tx):
        snap = doc_ref.get(transaction=tx)
        state = _normalize_global_generation_state_payload(
            payload=(snap.to_dict() if snap.exists else {}),
            global_target_id=global_target_id,
            target_samples=target_samples,
        )
        workers = state.get("workers")
        if not isinstance(workers, dict):
            workers = {}
        archived_samples = max(0, int(state.get("archived_samples", 0) or 0))
        archived_games = max(0, int(state.get("archived_games", 0) or 0))
        current = _normalize_worker_progress_entry(
            job_id=job_id,
            payload=workers.get(job_id, {}),
            fallback_dataset_source_id=dataset_source_id,
        ) or {
            "job_id": str(job_id),
            "dataset_source_id": str(dataset_source_id),
            "contributed_samples": 0,
            "contributed_games": 0,
            "updated_at": utc_now_iso(),
        }
        current["dataset_source_id"] = str(dataset_source_id)
        current["contributed_samples"] = max(0, int(current.get("contributed_samples", 0)) + int(delta_samples))
        current["contributed_games"] = max(0, int(current.get("contributed_games", 0)) + int(delta_games))
        current["updated_at"] = utc_now_iso()
        workers[str(job_id)] = {
            "dataset_source_id": str(current.get("dataset_source_id", "")),
            "contributed_samples": int(current.get("contributed_samples", 0)),
            "contributed_games": int(current.get("contributed_games", 0)),
            "updated_at": str(current.get("updated_at", "")),
        }
        workers, dropped_samples, dropped_games = _apply_worker_progress_retention(
            workers=workers,
            keep_job_id=job_id,
            stale_seconds=workers_retention_seconds,
            max_entries=workers_max_entries,
        )
        archived_samples += max(0, int(dropped_samples))
        archived_games += max(0, int(dropped_games))
        active_samples = sum(int(item.get("contributed_samples", 0)) for item in workers.values() if isinstance(item, dict))
        active_games = sum(int(item.get("contributed_games", 0)) for item in workers.values() if isinstance(item, dict))
        state["workers"] = workers
        state["archived_samples"] = int(archived_samples)
        state["archived_games"] = int(archived_games)
        state["total_samples"] = int(archived_samples + active_samples)
        state["total_games"] = int(archived_games + active_games)
        state["updated_at"] = utc_now_iso()
        tx.set(doc_ref, state)
        return state

    try:
        state = _tx(transaction)
    except Exception as exc:
        _raise_firestore_progress_error(
            stage="update_progress_transaction",
            progress_backend=backend,
            exc=exc,
            details=f"job_id={job_id} dataset_source_id={dataset_source_id}",
        )
    _write_global_generation_progress_redis_cache(
        global_target_id=global_target_id,
        state=state,
        progress_backend=backend,
    )
    return state


def _reserve_global_generation_budget_firestore(
    *,
    progress_path: Path,
    global_target_id: str,
    target_samples: int,
    job_id: str,
    dataset_source_id: str,
    requested_samples: int,
    requested_games: int = 1,
    progress_backend: dict[str, Any] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    workers_retention_seconds = max(0, _safe_int(backend.get("workers_retention_seconds", 86400), 86400))
    workers_max_entries = max(1, _safe_int(backend.get("workers_max_entries", 5000), 5000))
    endpoint = _resolve_firestore_progress_endpoint(backend)
    if endpoint is None:
        _raise_firestore_progress_error(
            stage="resolve_endpoint",
            progress_backend=backend,
            details="endpoint introuvable (backend invalide)",
        )
    client, doc_ref, firestore_mod = endpoint
    transaction = client.transaction()

    @firestore_mod.transactional
    def _tx(tx):
        snap = doc_ref.get(transaction=tx)
        state = _normalize_global_generation_state_payload(
            payload=(snap.to_dict() if snap.exists else {}),
            global_target_id=global_target_id,
            target_samples=target_samples,
        )
        state_target = int(state.get("target_samples") or target_samples or 0)
        state["target_samples"] = state_target
        state_total_samples = int(state.get("total_samples", 0))

        req_samples = max(0, int(requested_samples))
        allowed_samples = req_samples
        if state_target > 0:
            remaining = max(0, state_target - state_total_samples)
            allowed_samples = min(req_samples, remaining)
        allowed_games = int(requested_games) if allowed_samples > 0 else 0

        workers = state.get("workers")
        if not isinstance(workers, dict):
            workers = {}
        archived_samples = max(0, int(state.get("archived_samples", 0) or 0))
        archived_games = max(0, int(state.get("archived_games", 0) or 0))
        current = _normalize_worker_progress_entry(
            job_id=job_id,
            payload=workers.get(job_id, {}),
            fallback_dataset_source_id=dataset_source_id,
        ) or {
            "job_id": str(job_id),
            "dataset_source_id": str(dataset_source_id),
            "contributed_samples": 0,
            "contributed_games": 0,
            "updated_at": utc_now_iso(),
        }
        current["dataset_source_id"] = str(dataset_source_id)
        current["contributed_samples"] = int(current.get("contributed_samples", 0)) + int(max(0, allowed_samples))
        current["contributed_games"] = int(current.get("contributed_games", 0)) + int(max(0, allowed_games))
        current["updated_at"] = utc_now_iso()
        workers[str(job_id)] = {
            "dataset_source_id": str(current.get("dataset_source_id", "")),
            "contributed_samples": int(current.get("contributed_samples", 0)),
            "contributed_games": int(current.get("contributed_games", 0)),
            "updated_at": str(current.get("updated_at", "")),
        }
        workers, dropped_samples, dropped_games = _apply_worker_progress_retention(
            workers=workers,
            keep_job_id=job_id,
            stale_seconds=workers_retention_seconds,
            max_entries=workers_max_entries,
        )
        archived_samples += max(0, int(dropped_samples))
        archived_games += max(0, int(dropped_games))
        active_samples = sum(int(item.get("contributed_samples", 0)) for item in workers.values() if isinstance(item, dict))
        active_games = sum(int(item.get("contributed_games", 0)) for item in workers.values() if isinstance(item, dict))
        state["workers"] = workers
        state["archived_samples"] = int(archived_samples)
        state["archived_games"] = int(archived_games)
        state["total_samples"] = int(archived_samples + active_samples)
        state["total_games"] = int(archived_games + active_games)
        state["updated_at"] = utc_now_iso()
        tx.set(doc_ref, state)
        return int(allowed_samples), int(allowed_games), state

    try:
        allowed_samples, allowed_games, state = _tx(transaction)
    except Exception as exc:
        _raise_firestore_progress_error(
            stage="reserve_budget_transaction",
            progress_backend=backend,
            exc=exc,
            details=(
                f"job_id={job_id} dataset_source_id={dataset_source_id} "
                f"requested_samples={requested_samples} requested_games={requested_games}"
            ),
        )
    _write_global_generation_progress_redis_cache(
        global_target_id=global_target_id,
        state=state,
        progress_backend=backend,
    )
    return allowed_samples, allowed_games, state


def _read_global_generation_progress_firestore(
    *,
    progress_path: Path,
    global_target_id: str,
    target_samples: int,
    progress_backend: dict[str, Any] | None = None,
    prefer_cache: bool = True,
    telemetry: dict[str, int] | None = None,
) -> dict[str, Any]:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    redis_enabled = bool(backend.get("redis_enabled", False))
    if prefer_cache:
        cached = _read_global_generation_progress_redis_cache(
            global_target_id=global_target_id,
            target_samples=target_samples,
            progress_backend=backend,
            telemetry=telemetry,
        )
        if cached is not None:
            _mirror_global_generation_progress_state(progress_path, cached)
            return cached
        if redis_enabled:
            _touch_progress_read_telemetry(telemetry, "fallback_firestore")
    endpoint = _resolve_firestore_progress_endpoint(backend)
    if endpoint is None:
        _raise_firestore_progress_error(
            stage="resolve_endpoint",
            progress_backend=backend,
            details="endpoint introuvable (backend invalide)",
        )
    _client, doc_ref, _firestore_mod = endpoint
    try:
        snap = doc_ref.get()
    except Exception as exc:
        _raise_firestore_progress_error(
            stage="read_progress_document",
            progress_backend=backend,
            exc=exc,
        )
    state = _normalize_global_generation_state_payload(
        payload=(snap.to_dict() if snap.exists else {}),
        global_target_id=global_target_id,
        target_samples=target_samples,
    )
    _touch_progress_read_telemetry(telemetry, "firestore_read")
    _write_global_generation_progress_redis_cache(
        global_target_id=global_target_id,
        state=state,
        progress_backend=backend,
    )
    _mirror_global_generation_progress_state(progress_path, state)
    return state


def _default_global_generation_progress_state(global_target_id: str, target_samples: int) -> dict[str, Any]:
    return {
        "global_target_id": global_target_id,
        "target_samples": int(target_samples),
        "total_samples": 0,
        "total_games": 0,
        "archived_samples": 0,
        "archived_games": 0,
        "workers": {},
        "updated_at": utc_now_iso(),
    }


def _global_generation_workers_dir(progress_path: Path) -> Path:
    return progress_path.parent / f"{progress_path.stem}.workers"


def _global_generation_worker_snapshot_path(progress_path: Path, job_id: str) -> Path:
    digest = hashlib.sha1(str(job_id).encode("utf-8")).hexdigest()
    return _global_generation_workers_dir(progress_path) / f"{digest}.json"


def _normalize_worker_progress_entry(
    *,
    job_id: str,
    payload: Any,
    fallback_dataset_source_id: str = "",
) -> dict[str, Any] | None:
    resolved_job_id = str(job_id).strip()
    if not resolved_job_id:
        return None
    raw = payload if isinstance(payload, dict) else {}
    dataset_source_id = str(raw.get("dataset_source_id", fallback_dataset_source_id)).strip()
    contributed_samples = int(raw.get("contributed_samples", 0) or 0)
    contributed_games = int(raw.get("contributed_games", 0) or 0)
    updated_at = str(raw.get("updated_at", "")).strip() or utc_now_iso()
    return {
        "job_id": resolved_job_id,
        "dataset_source_id": dataset_source_id,
        "contributed_samples": max(0, contributed_samples),
        "contributed_games": max(0, contributed_games),
        "updated_at": updated_at,
    }


def _merge_worker_progress_maps(
    *maps: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for mapping in maps:
        if not isinstance(mapping, dict):
            continue
        for job_id, payload in mapping.items():
            normalized = _normalize_worker_progress_entry(job_id=str(job_id), payload=payload)
            if normalized is None:
                continue
            existing = merged.get(normalized["job_id"])
            if existing is None:
                merged[normalized["job_id"]] = normalized
                continue
            existing["contributed_samples"] = max(
                int(existing.get("contributed_samples", 0)),
                int(normalized.get("contributed_samples", 0)),
            )
            existing["contributed_games"] = max(
                int(existing.get("contributed_games", 0)),
                int(normalized.get("contributed_games", 0)),
            )
            if not str(existing.get("dataset_source_id", "")).strip():
                existing["dataset_source_id"] = str(normalized.get("dataset_source_id", ""))
            if str(normalized.get("updated_at", "")) > str(existing.get("updated_at", "")):
                existing["updated_at"] = str(normalized.get("updated_at", ""))
            merged[normalized["job_id"]] = existing
    return merged


def _load_global_generation_worker_snapshots(progress_path: Path) -> dict[str, dict[str, Any]]:
    workers_dir = _global_generation_workers_dir(progress_path)
    if not workers_dir.exists():
        return {}
    snapshots: dict[str, dict[str, Any]] = {}
    for snapshot_path in workers_dir.glob("*.json"):
        payload = _read_json_file(snapshot_path, {})
        if not isinstance(payload, dict):
            continue
        job_id = str(payload.get("job_id", "")).strip()
        normalized = _normalize_worker_progress_entry(
            job_id=job_id,
            payload=payload,
            fallback_dataset_source_id=str(payload.get("dataset_source_id", "")).strip(),
        )
        if normalized is None:
            continue
        snapshots = _merge_worker_progress_maps(snapshots, {normalized["job_id"]: normalized})
    return snapshots


def _write_global_generation_worker_snapshot(
    *,
    progress_path: Path,
    worker_payload: dict[str, Any],
) -> None:
    job_id = str(worker_payload.get("job_id", "")).strip()
    if not job_id:
        return
    workers_dir = _global_generation_workers_dir(progress_path)
    workers_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = _global_generation_worker_snapshot_path(progress_path, job_id)
    write_json_atomic(snapshot_path, worker_payload, ensure_ascii=True, indent=2)


def _sync_global_generation_progress_state(
    *,
    progress_path: Path,
    global_target_id: str,
    target_samples: int,
) -> dict[str, Any]:
    state = _read_json_file(
        progress_path,
        _default_global_generation_progress_state(global_target_id, target_samples),
    )
    state["global_target_id"] = str(state.get("global_target_id") or global_target_id)
    state_target = int(state.get("target_samples") or target_samples or 0)
    state["target_samples"] = state_target

    workers_from_state: dict[str, dict[str, Any]] = {}
    raw_workers = state.get("workers")
    if isinstance(raw_workers, dict):
        for worker_job_id, payload in raw_workers.items():
            normalized = _normalize_worker_progress_entry(job_id=str(worker_job_id), payload=payload)
            if normalized is not None:
                workers_from_state[normalized["job_id"]] = normalized

    workers_from_snapshots = _load_global_generation_worker_snapshots(progress_path)
    merged_workers = _merge_worker_progress_maps(workers_from_state, workers_from_snapshots)

    archived_samples = max(0, int(state.get("archived_samples", 0) or 0))
    archived_games = max(0, int(state.get("archived_games", 0) or 0))
    merged_total_samples = sum(int(item.get("contributed_samples", 0)) for item in merged_workers.values())
    merged_total_games = sum(int(item.get("contributed_games", 0)) for item in merged_workers.values())

    state["workers"] = {
        worker_job_id: {
            "dataset_source_id": str(payload.get("dataset_source_id", "")),
            "contributed_samples": int(payload.get("contributed_samples", 0)),
            "contributed_games": int(payload.get("contributed_games", 0)),
            "updated_at": str(payload.get("updated_at", "")),
        }
        for worker_job_id, payload in merged_workers.items()
    }
    state["archived_samples"] = int(archived_samples)
    state["archived_games"] = int(archived_games)
    state["total_samples"] = max(int(state.get("total_samples", 0)), int(archived_samples + merged_total_samples))
    state["total_games"] = max(int(state.get("total_games", 0)), int(archived_games + merged_total_games))
    state["updated_at"] = utc_now_iso()
    return state


def _update_global_generation_progress(
    *,
    progress_path: Path,
    lock_dir: Path,
    global_target_id: str,
    target_samples: int,
    job_id: str,
    dataset_source_id: str,
    delta_samples: int = 0,
    delta_games: int = 0,
    progress_backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    if str(backend.get("backend", "file")).strip().lower() == "firestore":
        return _update_global_generation_progress_firestore(
            progress_path=progress_path,
            global_target_id=global_target_id,
            target_samples=target_samples,
            job_id=job_id,
            dataset_source_id=dataset_source_id,
            delta_samples=delta_samples,
            delta_games=delta_games,
            progress_backend=backend,
        )

    lock_ok = _acquire_lock_dir(lock_dir)
    if not lock_ok:
        return {}
    try:
        state = _sync_global_generation_progress_state(
            progress_path=progress_path,
            global_target_id=global_target_id,
            target_samples=target_samples,
        )

        workers = state.get("workers")
        if not isinstance(workers, dict):
            workers = {}
        archived_samples = max(0, int(state.get("archived_samples", 0) or 0))
        archived_games = max(0, int(state.get("archived_games", 0) or 0))
        current = _normalize_worker_progress_entry(
            job_id=job_id,
            payload=workers.get(job_id, {}),
            fallback_dataset_source_id=dataset_source_id,
        ) or {
            "job_id": str(job_id),
            "dataset_source_id": str(dataset_source_id),
            "contributed_samples": 0,
            "contributed_games": 0,
            "updated_at": utc_now_iso(),
        }
        current["dataset_source_id"] = str(dataset_source_id)
        current["contributed_samples"] = max(0, int(current.get("contributed_samples", 0)) + int(delta_samples))
        current["contributed_games"] = max(0, int(current.get("contributed_games", 0)) + int(delta_games))
        current["updated_at"] = utc_now_iso()
        workers[str(job_id)] = {
            "dataset_source_id": str(current.get("dataset_source_id", "")),
            "contributed_samples": int(current.get("contributed_samples", 0)),
            "contributed_games": int(current.get("contributed_games", 0)),
            "updated_at": str(current.get("updated_at", "")),
        }
        _write_global_generation_worker_snapshot(progress_path=progress_path, worker_payload=current)
        active_samples = sum(int(item.get("contributed_samples", 0)) for item in workers.values() if isinstance(item, dict))
        active_games = sum(int(item.get("contributed_games", 0)) for item in workers.values() if isinstance(item, dict))
        state["workers"] = workers
        state["archived_samples"] = int(archived_samples)
        state["archived_games"] = int(archived_games)
        state["total_samples"] = int(archived_samples + active_samples)
        state["total_games"] = int(archived_games + active_games)
        state["updated_at"] = utc_now_iso()
        write_json_atomic(progress_path, state, ensure_ascii=True, indent=2)
        return state
    finally:
        _release_lock_dir(lock_dir)


def _reserve_global_generation_budget(
    *,
    progress_path: Path,
    lock_dir: Path,
    global_target_id: str,
    target_samples: int,
    job_id: str,
    dataset_source_id: str,
    requested_samples: int,
    requested_games: int = 1,
    progress_backend: dict[str, Any] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    if str(backend.get("backend", "file")).strip().lower() == "firestore":
        return _reserve_global_generation_budget_firestore(
            progress_path=progress_path,
            global_target_id=global_target_id,
            target_samples=target_samples,
            job_id=job_id,
            dataset_source_id=dataset_source_id,
            requested_samples=requested_samples,
            requested_games=requested_games,
            progress_backend=backend,
        )

    lock_ok = _acquire_lock_dir(lock_dir)
    if not lock_ok:
        return 0, 0, {}
    try:
        state = _sync_global_generation_progress_state(
            progress_path=progress_path,
            global_target_id=global_target_id,
            target_samples=target_samples,
        )
        state_target = int(state.get("target_samples") or target_samples or 0)
        state["target_samples"] = state_target
        state_total_samples = int(state.get("total_samples", 0))

        requested_samples = max(0, int(requested_samples))
        allowed_samples = requested_samples
        if state_target > 0:
            remaining = max(0, state_target - state_total_samples)
            allowed_samples = min(requested_samples, remaining)
        allowed_games = int(requested_games) if allowed_samples > 0 else 0

        workers = state.get("workers")
        if not isinstance(workers, dict):
            workers = {}
        archived_samples = max(0, int(state.get("archived_samples", 0) or 0))
        archived_games = max(0, int(state.get("archived_games", 0) or 0))
        current = _normalize_worker_progress_entry(
            job_id=job_id,
            payload=workers.get(job_id, {}),
            fallback_dataset_source_id=dataset_source_id,
        ) or {
            "job_id": str(job_id),
            "dataset_source_id": str(dataset_source_id),
            "contributed_samples": 0,
            "contributed_games": 0,
            "updated_at": utc_now_iso(),
        }
        current["dataset_source_id"] = str(dataset_source_id)
        current["contributed_samples"] = int(current.get("contributed_samples", 0)) + int(max(0, allowed_samples))
        current["contributed_games"] = int(current.get("contributed_games", 0)) + int(max(0, allowed_games))
        current["updated_at"] = utc_now_iso()
        workers[str(job_id)] = {
            "dataset_source_id": str(current.get("dataset_source_id", "")),
            "contributed_samples": int(current.get("contributed_samples", 0)),
            "contributed_games": int(current.get("contributed_games", 0)),
            "updated_at": str(current.get("updated_at", "")),
        }
        _write_global_generation_worker_snapshot(progress_path=progress_path, worker_payload=current)
        active_samples = sum(int(item.get("contributed_samples", 0)) for item in workers.values() if isinstance(item, dict))
        active_games = sum(int(item.get("contributed_games", 0)) for item in workers.values() if isinstance(item, dict))
        state["workers"] = workers
        state["archived_samples"] = int(archived_samples)
        state["archived_games"] = int(archived_games)
        state["total_samples"] = int(archived_samples + active_samples)
        state["total_games"] = int(archived_games + active_games)
        state["updated_at"] = utc_now_iso()

        write_json_atomic(progress_path, state, ensure_ascii=True, indent=2)
        return allowed_samples, allowed_games, state
    finally:
        _release_lock_dir(lock_dir)


def _read_global_generation_progress(
    progress_path: Path,
    global_target_id: str,
    target_samples: int,
    *,
    progress_backend: dict[str, Any] | None = None,
    prefer_cache: bool = True,
    telemetry: dict[str, int] | None = None,
) -> dict[str, Any]:
    backend = progress_backend if isinstance(progress_backend, dict) else {}
    if str(backend.get("backend", "file")).strip().lower() == "firestore":
        return _read_global_generation_progress_firestore(
            progress_path=progress_path,
            global_target_id=global_target_id,
            target_samples=target_samples,
            progress_backend=backend,
            prefer_cache=prefer_cache,
            telemetry=telemetry,
        )

    state = _sync_global_generation_progress_state(
        progress_path=progress_path,
        global_target_id=global_target_id,
        target_samples=target_samples,
    )
    state.setdefault("global_target_id", global_target_id)
    state.setdefault("target_samples", int(target_samples))
    state.setdefault("total_samples", 0)
    state.setdefault("total_games", 0)
    state.setdefault("workers", {})
    return state


def _dataset_registry_path(job: JobContext) -> Path:
    return job.paths.data_root / "dataset_registry.json"


def _dataset_registry_lock_dir(job: JobContext) -> Path:
    return job.paths.data_root / "dataset_registry.lock"


def _dataset_registry_default_payload() -> dict[str, Any]:
    return {"dataset_sources": [], "built_datasets": []}


def _resolve_dataset_registry_backend_config(job: JobContext) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    for key in ("dataset_generation", "dataset_build", "dataset_merge_final"):
        value = job.config.get(key, {})
        if isinstance(value, dict):
            sections.append(value)

    def _pick(*keys: str, default: str = "") -> str:
        for section in sections:
            for key in keys:
                text = str(section.get(key, "")).strip()
                if text:
                    return text
        return default

    backend = _pick("dataset_registry_backend", "global_progress_backend", "global_target_progress_backend", default="file").lower() or "file"
    project_id = _pick("dataset_registry_firestore_project_id", "global_progress_firestore_project_id", "global_target_progress_firestore_project_id")
    collection = _pick("dataset_registry_firestore_collection", default="dataset_registry") or "dataset_registry"
    document = _pick("dataset_registry_firestore_document", default="primary") or "primary"
    credentials_path = _pick("dataset_registry_firestore_credentials_path", "global_progress_firestore_credentials_path", "global_target_progress_firestore_credentials_path")
    api_key = _pick("dataset_registry_firestore_api_key", "global_progress_firestore_api_key", "global_target_progress_firestore_api_key")
    return {
        "backend": backend,
        "firestore_project_id": project_id,
        "firestore_collection": collection,
        "firestore_document": document,
        "firestore_credentials_path": credentials_path,
        "firestore_api_key": api_key,
    }


def _dataset_registry_doc_ref_from_job(job: JobContext):
    backend = _resolve_dataset_registry_backend_config(job)
    if str(backend.get("backend", "file")).strip().lower() != "firestore":
        return None, backend
    endpoint = _resolve_firestore_progress_endpoint(
        {
            "backend": "firestore",
            "global_target_id": "dataset_registry",
            "firestore_project_id": str(backend.get("firestore_project_id", "")).strip(),
            "firestore_collection": str(backend.get("firestore_collection", "dataset_registry")).strip() or "dataset_registry",
            "firestore_document": str(backend.get("firestore_document", "primary")).strip() or "primary",
            "firestore_credentials_path": str(backend.get("firestore_credentials_path", "")).strip(),
            "firestore_api_key": str(backend.get("firestore_api_key", "")).strip(),
        }
    )
    if endpoint is None:
        return None, backend
    client, doc_ref, firestore_mod = endpoint
    return (client, doc_ref, firestore_mod), backend


def _read_dataset_registry(job: JobContext) -> dict[str, Any]:
    endpoint, _resolved_backend = _dataset_registry_doc_ref_from_job(job)
    if endpoint is not None:
        _client, doc_ref, _firestore_mod = endpoint
        snap = doc_ref.get()
        payload = snap.to_dict() if snap.exists else _dataset_registry_default_payload()
    else:
        payload = _read_json_file(_dataset_registry_path(job), _dataset_registry_default_payload())
    if not isinstance(payload, dict):
        return _dataset_registry_default_payload()
    payload.setdefault("dataset_sources", [])
    payload.setdefault("built_datasets", [])
    return payload


def _write_dataset_registry(job: JobContext, payload: dict[str, Any]) -> None:
    endpoint, _resolved_backend = _dataset_registry_doc_ref_from_job(job)
    if endpoint is not None:
        _client, doc_ref, _firestore_mod = endpoint
        doc_ref.set(dict(payload))
        return
    path = _dataset_registry_path(job)
    write_json_atomic(path, payload, ensure_ascii=True, indent=2)


def _mutate_dataset_registry(job: JobContext, updater: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    endpoint, _resolved_backend = _dataset_registry_doc_ref_from_job(job)
    if endpoint is not None:
        client, doc_ref, firestore_mod = endpoint
        transaction = client.transaction()

        @firestore_mod.transactional
        def _tx(tx):
            snap = doc_ref.get(transaction=tx)
            registry = snap.to_dict() if snap.exists else _dataset_registry_default_payload()
            if not isinstance(registry, dict):
                registry = _dataset_registry_default_payload()
            registry.setdefault("dataset_sources", [])
            registry.setdefault("built_datasets", [])
            updater(registry)
            tx.set(doc_ref, registry)
            return registry

        return _tx(transaction)

    lock_dir = _dataset_registry_lock_dir(job)
    lock_ok = _acquire_lock_dir(lock_dir, timeout_seconds=45.0, poll_seconds=0.1)
    if not lock_ok:
        raise TimeoutError("Impossible d'acquerir le verrou du dataset_registry.")
    try:
        registry = _read_dataset_registry(job)
        updater(registry)
        _write_dataset_registry(job, registry)
        return registry
    finally:
        _release_lock_dir(lock_dir)


def _dataset_merge_final_lock_dir(job: JobContext, dataset_id: str) -> Path:
    safe_dataset_id = str(dataset_id).strip().replace("/", "_").replace(":", "_")
    return job.paths.data_root / f"dataset_merge_final.{safe_dataset_id}.lock"


def _acquire_dataset_merge_final_lock(
    job: JobContext,
    *,
    dataset_id: str,
    lock_ttl_seconds: float,
    lock_wait_seconds: float,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        raise ValueError("dataset_id vide pour acquisition du verrou dataset_merge_final.")

    lock_ttl_seconds = max(30.0, float(lock_ttl_seconds))
    lock_wait_seconds = max(1.0, float(lock_wait_seconds))
    poll_seconds = max(0.2, float(poll_seconds))
    deadline = time.time() + lock_wait_seconds

    endpoint, _backend = _dataset_registry_doc_ref_from_job(job)
    if endpoint is not None:
        client, doc_ref, firestore_mod = endpoint
        lock_path = ("_locks", "dataset_merge_final", dataset_id)

        @firestore_mod.transactional
        def _try_acquire(tx):
            snap = doc_ref.get(transaction=tx)
            payload = snap.to_dict() if snap.exists else _dataset_registry_default_payload()
            if not isinstance(payload, dict):
                payload = _dataset_registry_default_payload()
            locks = payload.get("_locks", {})
            if not isinstance(locks, dict):
                locks = {}
            merge_locks = locks.get("dataset_merge_final", {})
            if not isinstance(merge_locks, dict):
                merge_locks = {}

            now_ts = time.time()
            existing = merge_locks.get(dataset_id)
            existing_owner = ""
            if isinstance(existing, dict):
                existing_owner = str(existing.get("owner_job_id", "")).strip()
            existing_expires_at = float(existing.get("expires_at", 0.0) or 0.0) if isinstance(existing, dict) else 0.0
            lock_active = bool(existing_owner) and existing_expires_at > now_ts
            if lock_active and existing_owner != job.job_id:
                return False, existing_owner, existing_expires_at, ""

            token = f"{job.job_id}:{int(now_ts * 1000)}"
            merge_locks[dataset_id] = {
                "owner_job_id": job.job_id,
                "token": token,
                "acquired_at": now_ts,
                "expires_at": now_ts + lock_ttl_seconds,
            }
            locks["dataset_merge_final"] = merge_locks
            payload["_locks"] = locks
            tx.set(doc_ref, payload, merge=True)
            return True, job.job_id, now_ts + lock_ttl_seconds, token

        while True:
            tx = client.transaction()
            acquired, owner, expires_at, token = _try_acquire(tx)
            if acquired:
                return {
                    "backend": "firestore",
                    "dataset_id": dataset_id,
                    "owner_job_id": str(owner),
                    "expires_at": float(expires_at),
                    "token": str(token),
                }
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timeout acquisition lock merge final (firestore) | dataset_id={dataset_id} | owner={owner}"
                )
            time.sleep(poll_seconds)

    # Fallback lock on shared Drive filesystem.
    lock_dir = _dataset_merge_final_lock_dir(job, dataset_id)
    lock_ok = _acquire_lock_dir(lock_dir, timeout_seconds=lock_wait_seconds, poll_seconds=poll_seconds)
    if not lock_ok:
        raise TimeoutError(
            f"Timeout acquisition lock merge final (drive lock dir) | dataset_id={dataset_id} | lock_dir={lock_dir}"
        )
    return {
        "backend": "drive_lock_dir",
        "dataset_id": dataset_id,
        "lock_dir": str(lock_dir),
    }


def _release_dataset_merge_final_lock(job: JobContext, lock_handle: dict[str, Any] | None) -> None:
    if not isinstance(lock_handle, dict):
        return
    backend = str(lock_handle.get("backend", "")).strip().lower()
    dataset_id = str(lock_handle.get("dataset_id", "")).strip()
    if backend == "drive_lock_dir":
        lock_dir = Path(str(lock_handle.get("lock_dir", "")).strip())
        if lock_dir:
            _release_lock_dir(lock_dir)
        return
    if backend != "firestore" or not dataset_id:
        return

    token = str(lock_handle.get("token", "")).strip()
    endpoint, _resolved_backend = _dataset_registry_doc_ref_from_job(job)
    if endpoint is None:
        return
    client, doc_ref, firestore_mod = endpoint

    @firestore_mod.transactional
    def _tx_release(tx):
        snap = doc_ref.get(transaction=tx)
        payload = snap.to_dict() if snap.exists else _dataset_registry_default_payload()
        if not isinstance(payload, dict):
            payload = _dataset_registry_default_payload()
        locks = payload.get("_locks", {})
        if not isinstance(locks, dict):
            return
        merge_locks = locks.get("dataset_merge_final", {})
        if not isinstance(merge_locks, dict):
            return
        current = merge_locks.get(dataset_id)
        if not isinstance(current, dict):
            return
        current_owner = str(current.get("owner_job_id", "")).strip()
        current_token = str(current.get("token", "")).strip()
        if current_owner != job.job_id or (token and current_token != token):
            return
        merge_locks.pop(dataset_id, None)
        if merge_locks:
            locks["dataset_merge_final"] = merge_locks
        else:
            locks.pop("dataset_merge_final", None)
        if locks:
            payload["_locks"] = locks
        else:
            payload.pop("_locks", None)
        tx.set(doc_ref, payload, merge=True)

    try:
        tx = client.transaction()
        _tx_release(tx)
    except Exception:
        # Never crash caller on lock release path.
        pass


def _upsert_registry_entry(entries: list[dict[str, Any]], *, key: str, value: str, payload: dict[str, Any]) -> None:
    for index, entry in enumerate(entries):
        if str(entry.get(key, "")) == value:
            entries[index] = payload
            return
    entries.append(payload)


def _count_jsonl_files(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jsonl")) if root.exists() else 0


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, _line in enumerate(handle, start=1):
            pass
    return count


def _count_json_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.json") if not path.name.startswith("_"))


def _resolve_pool_max_tasks_per_child(
    *,
    start_method: str,
    configured_value: int,
    logger,
    scope: str,
) -> int | None:
    if start_method == "fork":
        if configured_value > 0:
            logger.info(
                "%s | start_method=fork so max_tasks_per_child is disabled automatically | configured_max_tasks_per_child=%s",
                scope,
                configured_value,
            )
        return None
    return configured_value if configured_value > 0 else None


def _copy_tree_incremental(source_root: Path, target_root: Path, *, pattern: str) -> int:
    copied = 0
    if not source_root.exists():
        return copied
    for source_path in sorted(source_root.rglob(pattern)):
        if source_path.name.startswith("_"):
            continue
        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def _resolve_dataset_source(job: JobContext, dataset_source_id: str) -> dict[str, Any]:
    registry = _read_dataset_registry(job)
    for entry in registry.get("dataset_sources", []):
        if str(entry.get("dataset_source_id", "")) == dataset_source_id:
            return _normalize_dataset_source_entry_paths(job, entry)
    job.logger.info(
        "dataset source missing from registry, probing legacy paths | dataset_source_id=%s",
        dataset_source_id,
    )
    legacy_entry = _discover_legacy_dataset_source(job, dataset_source_id)
    if legacy_entry is not None:
        job.logger.info(
            "dataset source legacy auto-registered | dataset_source_id=%s | sampled_dir=%s | raw_dir=%s",
            dataset_source_id,
            legacy_entry["sampled_dir"],
            legacy_entry["raw_dir"],
        )
        job.write_event(
            "dataset_source_legacy_autoregistered",
            dataset_source_id=dataset_source_id,
            sampled_dir=str(legacy_entry["sampled_dir"]),
            raw_dir=str(legacy_entry["raw_dir"]),
        )
        return legacy_entry
    raise FileNotFoundError(f"Dataset source introuvable dans le registre: {dataset_source_id}")


def _normalize_dataset_source_entry_paths(job: JobContext, entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    dataset_source_id = str(entry.get("dataset_source_id", "")).strip()
    if not dataset_source_id:
        return dict(entry)

    mydrive_root = Path("/content/drive/MyDrive")
    drive_root = job.paths.drive_root
    fallback_sampled_dir = job.paths.data_root / dataset_source_id
    fallback_raw_dir = job.paths.data_root / _default_raw_dir_name_for_dataset_source(dataset_source_id)

    normalized = dict(entry)
    changed = False

    def _normalize_key(key: str, fallback_path: Path) -> None:
        nonlocal changed
        raw_value = str(normalized.get(key, "")).strip()
        if not raw_value:
            normalized[key] = str(fallback_path)
            changed = True
            return
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = drive_root / candidate
        # Hard guard: refuse legacy MyDrive paths outside drive_root.
        if _path_within(candidate, mydrive_root) and not _path_within(candidate, drive_root):
            normalized[key] = str(fallback_path)
            changed = True
            return
        normalized_text = str(candidate)
        if normalized_text != raw_value:
            normalized[key] = normalized_text
            changed = True

    _normalize_key("sampled_dir", fallback_sampled_dir)
    _normalize_key("raw_dir", fallback_raw_dir)
    if not changed:
        return normalized

    normalized["updated_at"] = utc_now_iso()

    def _upsert_normalized(registry_payload: dict[str, Any]) -> None:
        _upsert_registry_entry(
            registry_payload["dataset_sources"],
            key="dataset_source_id",
            value=dataset_source_id,
            payload=normalized,
        )

    _mutate_dataset_registry(job, _upsert_normalized)
    job.logger.warning(
        "dataset source path normalized to drive_root | dataset_source_id=%s | sampled_dir=%s | raw_dir=%s",
        dataset_source_id,
        str(normalized.get("sampled_dir", "")),
        str(normalized.get("raw_dir", "")),
    )
    job.write_event(
        "dataset_source_path_normalized",
        dataset_source_id=dataset_source_id,
        sampled_dir=str(normalized.get("sampled_dir", "")),
        raw_dir=str(normalized.get("raw_dir", "")),
    )
    return normalized


def _discover_legacy_dataset_source(job: JobContext, dataset_source_id: str) -> dict[str, Any] | None:
    candidate_sampled_dirs = [
        job.paths.data_root / dataset_source_id,
        job.paths.drive_root / dataset_source_id,
    ]
    seen_paths: set[Path] = set()

    for sampled_dir in candidate_sampled_dirs:
        sampled_dir = sampled_dir.resolve()
        if sampled_dir in seen_paths:
            continue
        seen_paths.add(sampled_dir)
        if not sampled_dir.exists() or not sampled_dir.is_dir():
            continue
        sampled_files = _count_jsonl_files(sampled_dir)
        if sampled_files <= 0:
            continue

        metadata_path = sampled_dir / "_dataset_source_metadata.json"
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            def _upsert_metadata(registry: dict[str, Any]) -> None:
                _upsert_registry_entry(
                    registry["dataset_sources"],
                    key="dataset_source_id",
                    value=str(payload.get("dataset_source_id", dataset_source_id)),
                    payload=payload,
                )

            _mutate_dataset_registry(job, _upsert_metadata)
            return _normalize_dataset_source_entry_paths(job, payload)

        raw_dir_name = dataset_source_id
        if dataset_source_id.startswith("sampled_"):
            raw_dir_name = "raw_" + dataset_source_id[len("sampled_") :]
        raw_dir = job.paths.data_root / raw_dir_name
        if not raw_dir.exists():
            raw_dir = sampled_dir.parent / raw_dir_name

        # Legacy sources may be large; avoid an expensive full line count during
        # auto-registration and let dataset-build scan the files directly.
        payload = {
            "dataset_source_id": dataset_source_id,
            "source_mode": "legacy_import",
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "target_samples": 0,
            "games_per_matchup": 0,
            "sample_every_n_plies": 0,
            "matchups": [],
            "raw_files": _count_json_files(raw_dir),
            "sampled_files": sampled_files,
            "sampled_positions": 0,
            "source_dataset_id": "",
            "source_dataset_ids": [],
            "derivation_strategy": "",
            "derivation_params": {},
            "dataset_version": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        def _upsert_payload(registry: dict[str, Any]) -> None:
            _upsert_registry_entry(
                registry["dataset_sources"],
                key="dataset_source_id",
                value=dataset_source_id,
                payload=payload,
            )

        _mutate_dataset_registry(job, _upsert_payload)
        _write_json(sampled_dir / "_dataset_source_metadata.json", payload)
        if raw_dir.exists():
            _write_json(raw_dir / "_dataset_source_metadata.json", payload)
        return _normalize_dataset_source_entry_paths(job, payload)
    return None


def _resolve_built_dataset(job: JobContext, dataset_id: str) -> dict[str, Any]:
    registry = _read_dataset_registry(job)
    for entry in registry.get("built_datasets", []):
        if str(entry.get("dataset_id", "")) == dataset_id:
            return entry
    raise FileNotFoundError(f"Built dataset introuvable dans le registre: {dataset_id}")


def _register_dataset_source(
    job: JobContext,
    *,
    dataset_source_id: str,
    source_mode: str,
    raw_dir: Path,
    sampled_dir: Path,
    target_samples: int,
    games_per_matchup: int,
    sample_every_n_plies: int,
    matchups: list[str],
    source_dataset_id: str = "",
    source_dataset_ids: list[str] | None = None,
    derivation_strategy: str = "",
    derivation_params: dict[str, Any] | None = None,
    dataset_version: str | None = None,
    source_status: str = "completed",
    partial_summary: dict[str, Any] | None = None,
    raw_files_override: int | None = None,
    sampled_files_override: int | None = None,
    sampled_positions_override: int | None = None,
) -> dict[str, Any]:
    resolved_source_dataset_ids = source_dataset_ids or ([source_dataset_id] if source_dataset_id else [])
    payload = {
        "dataset_source_id": dataset_source_id,
        "source_mode": source_mode,
        "raw_dir": str(raw_dir),
        "sampled_dir": str(sampled_dir),
        "target_samples": target_samples,
        "games_per_matchup": games_per_matchup,
        "sample_every_n_plies": sample_every_n_plies,
        "matchups": matchups,
        "raw_files": int(raw_files_override) if raw_files_override is not None else _count_json_files(raw_dir),
        "sampled_files": int(sampled_files_override) if sampled_files_override is not None else _count_jsonl_files(sampled_dir),
        "sampled_positions": int(sampled_positions_override) if sampled_positions_override is not None else (_count_total_jsonl_lines(sampled_dir) if sampled_dir.exists() else 0),
        "source_dataset_id": source_dataset_id,
        "source_dataset_ids": resolved_source_dataset_ids,
        "derivation_strategy": derivation_strategy,
        "derivation_params": derivation_params or {},
        "source_status": source_status,
        "partial_summary": partial_summary or {},
        "dataset_version": dataset_version or utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    def _upsert_source(registry: dict[str, Any]) -> None:
        existing_entry: dict[str, Any] | None = None
        for entry in registry.get("dataset_sources", []):
            if str(entry.get("dataset_source_id", "")) == dataset_source_id:
                existing_entry = entry if isinstance(entry, dict) else None
                break
        # Keep source counters monotonic across restarts/checkpoints to avoid
        # visible regressions in monitoring when a worker resumes.
        if existing_entry is not None:
            payload["raw_files"] = max(int(payload.get("raw_files", 0)), int(existing_entry.get("raw_files", 0) or 0))
            payload["sampled_files"] = max(int(payload.get("sampled_files", 0)), int(existing_entry.get("sampled_files", 0) or 0))
            payload["sampled_positions"] = max(
                int(payload.get("sampled_positions", 0)),
                int(existing_entry.get("sampled_positions", 0) or 0),
            )
        _upsert_registry_entry(
            registry["dataset_sources"],
            key="dataset_source_id",
            value=dataset_source_id,
            payload=payload,
        )

    _mutate_dataset_registry(job, _upsert_source)
    _write_json(sampled_dir / "_dataset_source_metadata.json", payload)
    _write_json(raw_dir / "_dataset_source_metadata.json", payload)
    return payload


def _register_built_dataset(
    job: JobContext,
    *,
    dataset_id: str,
    source_dataset_id: str,
    source_dataset_ids: list[str] | None = None,
    sampled_root: Path,
    output_root: Path,
    label_cache_dir: Path,
    teacher_engine: str,
    teacher_level: str,
    split_summary: dict[str, dict[str, int]],
    labeled_samples: int,
    target_labeled_samples: int,
    build_mode: str = "teacher_label",
    parent_dataset_ids: list[str] | None = None,
    dataset_version: str | None = None,
    build_status: str = "completed",
    feature_schema_version: str = "policy_value_tactical_v3",
    input_dim: int | None = None,
    dedupe_sample_ids: bool = True,
    duplicate_samples_removed: int = 0,
) -> dict[str, Any]:
    resolved_source_dataset_ids = source_dataset_ids or ([source_dataset_id] if source_dataset_id else [])
    payload = {
        "dataset_id": dataset_id,
        "source_dataset_id": source_dataset_id,
        "source_dataset_ids": resolved_source_dataset_ids,
        "sampled_root": str(sampled_root),
        "output_dir": str(output_root),
        "label_cache_dir": str(label_cache_dir),
        "teacher_engine": teacher_engine,
        "teacher_level": teacher_level,
        "splits": split_summary,
        "labeled_samples": labeled_samples,
        "target_labeled_samples": target_labeled_samples,
        "build_mode": build_mode,
        "build_status": build_status,
        "parent_dataset_ids": parent_dataset_ids or [],
        "feature_schema_version": feature_schema_version,
        "input_dim": int(input_dim) if input_dim is not None else None,
        "dedupe_sample_ids": bool(dedupe_sample_ids),
        "duplicate_samples_removed": int(duplicate_samples_removed),
        "has_policy_target_full": True,
        "has_tactical_features": True,
        "has_tactical_move_masks": True,
        "has_strategy_features": True,
        "dataset_version": dataset_version or utc_now_iso(),
        "updated_at": utc_now_iso(),
    }
    def _upsert_built(registry: dict[str, Any]) -> None:
        existing_entry: dict[str, Any] | None = None
        for entry in registry.get("built_datasets", []):
            if str(entry.get("dataset_id", "")) == dataset_id:
                existing_entry = entry if isinstance(entry, dict) else None
                break
        # Keep built counters monotonic across restarts/checkpoints to avoid
        # temporary regressions in monitoring views.
        if existing_entry is not None:
            existing_labeled = int(existing_entry.get("labeled_samples", 0) or 0)
            payload_labeled = int(payload.get("labeled_samples", 0) or 0)
            if payload_labeled < existing_labeled:
                payload["labeled_samples"] = existing_labeled
                # If we keep older labeled count, keep its split snapshot too.
                if isinstance(existing_entry.get("splits"), dict):
                    payload["splits"] = existing_entry.get("splits")
            payload["target_labeled_samples"] = max(
                int(payload.get("target_labeled_samples", 0) or 0),
                int(existing_entry.get("target_labeled_samples", 0) or 0),
            )
            payload["duplicate_samples_removed"] = max(
                int(payload.get("duplicate_samples_removed", 0) or 0),
                int(existing_entry.get("duplicate_samples_removed", 0) or 0),
            )
            # Never downgrade a completed build back to partial.
            if str(existing_entry.get("build_status", "")).strip() == "completed":
                payload["build_status"] = "completed"
        _upsert_registry_entry(
            registry["built_datasets"],
            key="dataset_id",
            value=dataset_id,
            payload=payload,
        )

    _mutate_dataset_registry(job, _upsert_built)
    _write_json(output_root / "dataset_metadata.json", payload)
    return payload


def _sample_position_signature(sample: dict[str, Any]) -> str:
    state = sample["state"]
    scores = state["scores"]
    signature_payload = {
        "board": list(state["board"]),
        "south": int(scores["south"]),
        "north": int(scores["north"]),
        "player_to_move": str(state["player_to_move"]),
    }
    encoded = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _write_dataset_generation_progress_snapshot(
    *,
    job: JobContext,
    dataset_dir: Path,
    dataset_source_id: str,
    source_mode: str,
    raw_dir: Path,
    sampled_dir: Path,
    target_samples: int,
    games_per_matchup: int,
    sample_every_n_plies: int,
    matchups: list[str],
    source_dataset_id: str,
    source_dataset_ids: list[str] | None,
    derivation_strategy: str,
    derivation_params: dict[str, Any],
    summaries: list[dict[str, Any]],
    total_samples: int,
) -> dict[str, Any]:
    added_games = sum(int(item.get("games_added", 0)) for item in summaries)
    added_samples = sum(int(item.get("samples_added", 0)) for item in summaries)
    metadata = _register_dataset_source(
        job,
        dataset_source_id=dataset_source_id,
        source_mode=source_mode,
        raw_dir=raw_dir,
        sampled_dir=sampled_dir,
        target_samples=target_samples,
        games_per_matchup=games_per_matchup,
        sample_every_n_plies=sample_every_n_plies,
        matchups=matchups,
        source_dataset_id=source_dataset_id,
        source_dataset_ids=source_dataset_ids,
        derivation_strategy=derivation_strategy,
        derivation_params=derivation_params,
        source_status="partial",
        partial_summary={
            "completed_matchups": len(summaries),
            "added_games": added_games,
            "added_samples": added_samples,
            "total_samples": int(total_samples),
        },
    )
    summary = {
        "job_id": job.job_id,
        "dataset_source_id": dataset_source_id,
        "source_mode": source_mode,
        "matchups": summaries,
        "existing_samples": max(0, int(total_samples) - int(added_samples)),
        "added_games": added_games,
        "added_samples": added_samples,
        "total_samples": int(metadata["sampled_positions"]),
        "target_samples": target_samples,
    }
    _write_json(dataset_dir / "dataset_generation_summary.json", summary)
    return summary


def _write_benchmatch_progress_snapshot(
    *,
    job: JobContext,
    dataset_dir: Path,
    dataset_source_id: str,
    raw_dir: Path,
    sampled_dir: Path,
    target_samples: int,
    games_per_matchup: int,
    sample_every_n_plies: int,
    matchups: list[str],
    source_dataset_id: str,
    source_dataset_ids: list[str] | None,
    derivation_strategy: str,
    derivation_params: dict[str, Any],
    summaries: list[dict[str, Any]],
    total_samples: int,
) -> dict[str, Any]:
    added_games = sum(int(item.get("games_added", 0)) for item in summaries)
    added_samples = sum(int(item.get("samples_added", 0)) for item in summaries)
    metadata = _register_dataset_source(
        job,
        dataset_source_id=dataset_source_id,
        source_mode="benchmatch",
        raw_dir=raw_dir,
        sampled_dir=sampled_dir,
        target_samples=target_samples,
        games_per_matchup=games_per_matchup,
        sample_every_n_plies=sample_every_n_plies,
        matchups=matchups,
        source_dataset_id=source_dataset_id,
        source_dataset_ids=source_dataset_ids,
        derivation_strategy=derivation_strategy,
        derivation_params=derivation_params,
        source_status="partial",
        partial_summary={
            "completed_matchups": len(summaries),
            "added_games": added_games,
            "added_samples": added_samples,
            "total_samples": int(total_samples),
        },
        sampled_positions_override=int(total_samples),
    )
    summary = {
        "job_id": job.job_id,
        "dataset_source_id": dataset_source_id,
        "source_mode": "benchmatch",
        "matchups": summaries,
        "existing_samples": max(0, int(total_samples) - added_samples),
        "added_games": added_games,
        "added_samples": added_samples,
        "total_samples": int(metadata["sampled_positions"]),
        "target_samples": target_samples,
    }
    _write_json(dataset_dir / "dataset_generation_summary.json", summary)
    return summary


def _export_built_dataset_snapshot(
    *,
    job: JobContext,
    dataset_id: str,
    source_dataset_id: str,
    source_dataset_ids: list[str],
    sampled_root: Path,
    labeled_root: Path,
    output_root: Path,
    label_cache_dir: Path,
    teacher_engine: str,
    teacher_level: str,
    sampled_relative_names: list[str],
    completed_files: set[str],
    file_sample_counts: dict[str, int],
    train_ratio: float,
    validation_ratio: float,
    labeled_samples_total: int,
    target_labeled_samples: int,
    build_status: str,
    build_mode: str,
    dedupe_sample_ids: bool,
) -> tuple[dict[str, dict[str, int]], int, int]:
    game_files = [relative_name for relative_name in sampled_relative_names if relative_name in completed_files]
    split_train_end = int(len(game_files) * train_ratio)
    split_validation_end = split_train_end + int(len(game_files) * validation_ratio)
    split_files = {
        "train": game_files[:split_train_end],
        "validation": game_files[split_train_end:split_validation_end],
        "test": game_files[split_validation_end:],
    }

    split_summary: dict[str, dict[str, int]] = {}
    exported_input_dim = 17
    seen_sample_ids: set[str] = set()
    duplicate_samples_removed_total = 0
    for split_name, selected_files in split_files.items():
        selected_sample_count = 0
        split_duplicate_samples = 0
        for relative_name in selected_files:
            sample_count = file_sample_counts.get(relative_name)
            if sample_count is None:
                labeled_path = labeled_root / relative_name
                sample_count = _count_jsonl_lines(labeled_path) if labeled_path.exists() else 0
                file_sample_counts[relative_name] = sample_count
            selected_sample_count += sample_count

        features_list = []
        masks_list = []
        policy_list = []
        policy_full_list = []
        value_list = []
        capture_masks_list = []
        safe_masks_list = []
        risky_masks_list = []
        hard_example_weight_list = []
        sample_ids = []
        game_id_list = []
        for relative_name in selected_files:
            labeled_path = labeled_root / relative_name
            if not labeled_path.exists():
                continue
            for sample in _iter_jsonl(labeled_path):
                sample_id = str(sample.get("sample_id", ""))
                if dedupe_sample_ids and sample_id:
                    if sample_id in seen_sample_ids:
                        split_duplicate_samples += 1
                        duplicate_samples_removed_total += 1
                        continue
                    seen_sample_ids.add(sample_id)
                (
                    features,
                    legal_mask,
                    policy_index,
                    policy_target_full,
                    value,
                    capture_move_mask,
                    safe_move_mask,
                    risky_move_mask,
                    hard_example_weight,
                ) = _encode_features(sample)
                features_list.append(features)
                masks_list.append(legal_mask)
                policy_list.append(policy_index)
                policy_full_list.append(policy_target_full)
                value_list.append(value)
                capture_masks_list.append(capture_move_mask)
                safe_masks_list.append(safe_move_mask)
                risky_masks_list.append(risky_move_mask)
                hard_example_weight_list.append(hard_example_weight)
                sample_ids.append(sample_id or str(sample["sample_id"]))
                game_id_list.append(str(sample["game_id"]))

        x_dim = int(features_list[0].shape[0]) if features_list else exported_input_dim
        exported_input_dim = x_dim
        x = np.asarray(features_list, dtype=np.float32) if features_list else np.zeros((0, x_dim), dtype=np.float32)
        legal_mask = np.asarray(masks_list, dtype=np.float32) if masks_list else np.zeros((0, 7), dtype=np.float32)
        policy_index = np.asarray(policy_list, dtype=np.int64) if policy_list else np.zeros((0,), dtype=np.int64)
        policy_target_full = np.asarray(policy_full_list, dtype=np.float32) if policy_full_list else np.zeros((0, 7), dtype=np.float32)
        value_target = np.asarray(value_list, dtype=np.float32) if value_list else np.zeros((0,), dtype=np.float32)
        capture_move_mask = np.asarray(capture_masks_list, dtype=np.float32) if capture_masks_list else np.zeros((0, 7), dtype=np.float32)
        safe_move_mask = np.asarray(safe_masks_list, dtype=np.float32) if safe_masks_list else np.zeros((0, 7), dtype=np.float32)
        risky_move_mask = np.asarray(risky_masks_list, dtype=np.float32) if risky_masks_list else np.zeros((0, 7), dtype=np.float32)
        hard_example_weight = (
            np.asarray(hard_example_weight_list, dtype=np.float32)
            if hard_example_weight_list
            else np.ones((0,), dtype=np.float32)
        )
        _write_npz_compressed(
            output_root / f"{split_name}.npz",
            x=x,
            legal_mask=legal_mask,
            policy_index=policy_index,
            policy_target_full=policy_target_full,
            value_target=value_target,
            capture_move_mask=capture_move_mask,
            safe_move_mask=safe_move_mask,
            risky_move_mask=risky_move_mask,
            hard_example_weight=hard_example_weight,
            sample_ids=np.asarray(sample_ids, dtype=object),
            game_ids=np.asarray(game_id_list, dtype=object),
        )
        split_summary[split_name] = {
            "games": len(selected_files),
            "samples": selected_sample_count,
            "duplicate_samples_removed": split_duplicate_samples,
        }

    _register_built_dataset(
        job,
        dataset_id=dataset_id,
        source_dataset_id=source_dataset_id,
        source_dataset_ids=source_dataset_ids,
        sampled_root=sampled_root,
        output_root=output_root,
        label_cache_dir=label_cache_dir,
        teacher_engine=teacher_engine,
        teacher_level=teacher_level,
        split_summary=split_summary,
        labeled_samples=labeled_samples_total,
        target_labeled_samples=target_labeled_samples,
        build_mode=str(build_mode).strip() or "teacher_label",
        parent_dataset_ids=[],
        build_status=build_status,
        input_dim=exported_input_dim,
        dedupe_sample_ids=dedupe_sample_ids,
        duplicate_samples_removed=duplicate_samples_removed_total,
    )
    return split_summary, exported_input_dim, duplicate_samples_removed_total


def _derive_existing_dataset_source(
    *,
    source_entry: dict[str, Any],
    target_raw_dir: Path,
    target_sampled_dir: Path,
    target_samples: int,
    derivation_strategy: str,
    derivation_params: dict[str, Any],
) -> dict[str, Any]:
    source_raw_dir = Path(str(source_entry["raw_dir"]))
    source_sampled_dir = Path(str(source_entry["sampled_dir"]))
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    target_sampled_dir.mkdir(parents=True, exist_ok=True)

    supported_strategies = {
        "unique_positions",
        "endgame_focus",
        "high_branching",
        "balanced_score_gap",
        "balanced_legal_moves",
        "rare_seed_profiles",
    }
    if derivation_strategy not in supported_strategies:
        raise ValueError(f"Unsupported derivation_strategy: {derivation_strategy}")

    seen_signatures: set[str] = set()
    selected_files = 0
    selected_samples = 0
    scanned_files = 0
    scanned_samples = 0
    copied_raw_files = 0

    endgame_max_board_seeds = int(derivation_params.get("endgame_max_board_seeds", 24))
    high_branching_min_legal_moves = int(derivation_params.get("high_branching_min_legal_moves", 4))
    balanced_dedupe_positions = _as_bool(derivation_params.get("balanced_dedupe_positions", True), default=True)

    def _parse_int_list(value: Any, default: list[int]) -> list[int]:
        if isinstance(value, (list, tuple)):
            candidates = list(value)
        elif isinstance(value, str) and value.strip():
            candidates = [part.strip() for part in value.split(",") if part.strip()]
        else:
            candidates = list(default)
        parsed: list[int] = []
        for item in candidates:
            try:
                parsed.append(int(item))
            except Exception:
                continue
        if not parsed:
            parsed = list(default)
        parsed = sorted({int(v) for v in parsed})
        return parsed

    score_gap_boundaries = _parse_int_list(
        derivation_params.get("score_gap_boundaries", [0, 2, 5, 9]),
        default=[0, 2, 5, 9],
    )
    legal_moves_boundaries = _parse_int_list(
        derivation_params.get("legal_moves_boundaries", [1, 2, 3, 4, 5, 6]),
        default=[1, 2, 3, 4, 5, 6],
    )

    def _bucket_from_boundaries(value: int, boundaries: list[int], *, prefix: str) -> str:
        if not boundaries:
            return f"{prefix}:all"
        v = int(value)
        ordered = sorted(int(x) for x in boundaries)
        if v <= ordered[0]:
            return f"{prefix}:<= {ordered[0]}"
        for lower, upper in zip(ordered, ordered[1:]):
            if lower < v <= upper:
                return f"{prefix}:{lower + 1}-{upper}"
        return f"{prefix}:>= {ordered[-1] + 1}"

    def _safe_state_scores(sample: dict[str, Any]) -> tuple[int, int]:
        state = sample.get("state", {})
        scores = state.get("scores", [0, 0]) if isinstance(state, dict) else [0, 0]
        if isinstance(scores, dict):
            south = _safe_int(scores.get("south", 0), 0)
            north = _safe_int(scores.get("north", 0), 0)
        else:
            south = _safe_int(scores[0] if isinstance(scores, (list, tuple)) and len(scores) > 0 else 0, 0)
            north = _safe_int(scores[1] if isinstance(scores, (list, tuple)) and len(scores) > 1 else 0, 0)
        return int(south), int(north)

    def _safe_state_board(sample: dict[str, Any]) -> list[int]:
        state = sample.get("state", {})
        board = state.get("board", []) if isinstance(state, dict) else []
        if not isinstance(board, (list, tuple)):
            return []
        return [int(_safe_int(v, 0)) for v in board]

    def _score_gap_bucket(sample: dict[str, Any]) -> str:
        south, north = _safe_state_scores(sample)
        gap = abs(int(south) - int(north))
        return _bucket_from_boundaries(gap, score_gap_boundaries, prefix="score_gap")

    def _legal_moves_bucket(sample: dict[str, Any]) -> str:
        legal_count = len(sample.get("legal_moves", []))
        return _bucket_from_boundaries(int(legal_count), legal_moves_boundaries, prefix="legal_moves")

    def _rare_seed_profile_bucket(sample: dict[str, Any]) -> str:
        board = _safe_state_board(sample)
        if not board:
            return "seed_profile:unknown"
        occupied = sum(1 for value in board if int(value) > 0)
        max_stack = max(int(value) for value in board)
        big_stacks = sum(1 for value in board if int(value) >= 4)

        if occupied <= 4:
            occupied_band = "sparse"
        elif occupied <= 8:
            occupied_band = "mid"
        else:
            occupied_band = "dense"

        if max_stack <= 2:
            max_band = "low"
        elif max_stack <= 5:
            max_band = "mid"
        elif max_stack <= 9:
            max_band = "high"
        else:
            max_band = "extreme"

        if big_stacks == 0:
            big_band = "none"
        elif big_stacks <= 2:
            big_band = "few"
        else:
            big_band = "many"

        return f"seed_profile:occ={occupied_band}|max={max_band}|big={big_band}"

    def _bucket_for_balanced_strategy(sample: dict[str, Any]) -> str:
        if derivation_strategy == "balanced_score_gap":
            return _score_gap_bucket(sample)
        if derivation_strategy == "balanced_legal_moves":
            return _legal_moves_bucket(sample)
        if derivation_strategy == "rare_seed_profiles":
            return _rare_seed_profile_bucket(sample)
        return "default"

    def _build_bucket_quotas(
        bucket_counts: dict[str, int],
        *,
        requested_total: int,
        rare_weighted: bool,
    ) -> dict[str, int]:
        positive_counts = {bucket: int(count) for bucket, count in bucket_counts.items() if int(count) > 0}
        if not positive_counts:
            return {}
        total_candidates = int(sum(positive_counts.values()))
        if requested_total <= 0:
            return dict(positive_counts)
        target_total = min(int(requested_total), total_candidates)

        if rare_weighted:
            weights = {bucket: (1.0 / math.sqrt(float(count))) for bucket, count in positive_counts.items()}
        else:
            weights = {bucket: 1.0 for bucket in positive_counts}
        weight_sum = float(sum(weights.values()))
        if weight_sum <= 0.0:
            weights = {bucket: 1.0 for bucket in positive_counts}
            weight_sum = float(len(weights))

        quotas = {
            bucket: min(
                int(positive_counts[bucket]),
                int(math.floor((float(target_total) * float(weights[bucket])) / weight_sum)),
            )
            for bucket in positive_counts
        }
        assigned = int(sum(quotas.values()))
        remaining = max(0, int(target_total) - assigned)
        adjustable = [bucket for bucket in positive_counts if quotas[bucket] < positive_counts[bucket]]
        while remaining > 0 and adjustable:
            per_bucket = max(1, remaining // max(1, len(adjustable)))
            still_adjustable: list[str] = []
            for bucket in adjustable:
                slack = int(positive_counts[bucket]) - int(quotas[bucket])
                if slack <= 0:
                    continue
                add = min(slack, per_bucket, remaining)
                quotas[bucket] = int(quotas[bucket]) + int(add)
                remaining -= int(add)
                if quotas[bucket] < positive_counts[bucket]:
                    still_adjustable.append(bucket)
                if remaining <= 0:
                    break
            if not still_adjustable and remaining > 0:
                break
            adjustable = still_adjustable

        return quotas

    def _keep_sample(sample: dict[str, Any]) -> bool:
        nonlocal selected_samples
        if target_samples > 0 and selected_samples >= target_samples:
            return False
        if derivation_strategy == "unique_positions":
            signature = _sample_position_signature(sample)
            if signature in seen_signatures:
                return False
            seen_signatures.add(signature)
            return True
        if derivation_strategy == "endgame_focus":
            board_seeds = int(sum(int(value) for value in sample["state"]["board"]))
            return board_seeds <= endgame_max_board_seeds
        if derivation_strategy == "high_branching":
            return len(sample.get("legal_moves", [])) >= high_branching_min_legal_moves
        return False

    source_sampled_files = sorted(source_sampled_dir.rglob("*.jsonl"))

    bucket_counts: dict[str, int] = {}
    selected_by_bucket: dict[str, int] = {}
    bucket_quotas: dict[str, int] = {}
    first_pass_scanned_files = 0
    first_pass_scanned_samples = 0

    if derivation_strategy in {"balanced_score_gap", "balanced_legal_moves", "rare_seed_profiles"}:
        for source_sampled_file in source_sampled_files:
            first_pass_scanned_files += 1
            for sample in _iter_jsonl(source_sampled_file):
                first_pass_scanned_samples += 1
                bucket = _bucket_for_balanced_strategy(sample)
                bucket_counts[bucket] = int(bucket_counts.get(bucket, 0)) + 1
        bucket_quotas = _build_bucket_quotas(
            bucket_counts,
            requested_total=int(target_samples),
            rare_weighted=(derivation_strategy == "rare_seed_profiles"),
        )

    for source_sampled_file in source_sampled_files:
        scanned_files += 1
        relative_path = source_sampled_file.relative_to(source_sampled_dir)
        kept_samples: list[dict[str, Any]] = []
        for sample in _iter_jsonl(source_sampled_file):
            scanned_samples += 1
            keep = False
            if derivation_strategy in {"balanced_score_gap", "balanced_legal_moves", "rare_seed_profiles"}:
                if target_samples > 0 and selected_samples >= target_samples:
                    keep = False
                else:
                    bucket = _bucket_for_balanced_strategy(sample)
                    remaining_quota = int(bucket_quotas.get(bucket, 0))
                    if remaining_quota > 0:
                        if balanced_dedupe_positions:
                            signature = _sample_position_signature(sample)
                            if signature in seen_signatures:
                                keep = False
                            else:
                                seen_signatures.add(signature)
                                keep = True
                        else:
                            keep = True
                        if keep:
                            bucket_quotas[bucket] = remaining_quota - 1
                            selected_by_bucket[bucket] = int(selected_by_bucket.get(bucket, 0)) + 1
            else:
                keep = _keep_sample(sample)

            if keep:
                kept_samples.append(sample)
                selected_samples += 1
                if target_samples > 0 and selected_samples >= target_samples:
                    break

        if kept_samples:
            target_sampled_file = target_sampled_dir / relative_path
            write_jsonl_atomic(target_sampled_file, kept_samples, ensure_ascii=True)
            selected_files += 1

            if source_raw_dir.exists():
                source_raw_file = source_raw_dir / relative_path.with_suffix(".json")
                target_raw_file = target_raw_dir / relative_path.with_suffix(".json")
                if source_raw_file.exists() and not target_raw_file.exists():
                    target_raw_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_raw_file, target_raw_file)
                    copied_raw_files += 1

        if target_samples > 0 and selected_samples >= target_samples:
            break

    return {
        "scanned_files": scanned_files,
        "scanned_samples": scanned_samples,
        "selected_files": selected_files,
        "selected_samples": selected_samples,
        "copied_raw_files": copied_raw_files,
        "first_pass_scanned_files": first_pass_scanned_files,
        "first_pass_scanned_samples": first_pass_scanned_samples,
        "bucket_counts": bucket_counts,
        "selected_by_bucket": selected_by_bucket,
        "remaining_bucket_quotas": bucket_quotas,
        "derivation_params_effective": {
            "balanced_dedupe_positions": balanced_dedupe_positions,
            "score_gap_boundaries": score_gap_boundaries,
            "legal_moves_boundaries": legal_moves_boundaries,
        },
    }


def _augment_existing_dataset_source(
    *,
    job: JobContext,
    source_entry: dict[str, Any],
    target_raw_dir: Path,
    target_sampled_dir: Path,
    target_samples: int,
    augmentation_params: dict[str, Any],
) -> dict[str, Any]:
    source_raw_dir = Path(str(source_entry["raw_dir"]))
    source_sampled_dir = Path(str(source_entry["sampled_dir"]))
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    target_sampled_dir.mkdir(parents=True, exist_ok=True)

    include_original_samples = bool(augmentation_params.get("include_original_samples", True))
    max_depth = max(1, int(augmentation_params.get("max_depth", 2)))
    max_branching = max(1, int(augmentation_params.get("max_branching", 3)))
    max_generated_per_source_sample = max(1, int(augmentation_params.get("max_generated_per_source_sample", 8)))
    counterfactual_teacher_engine = str(
        augmentation_params.get("counterfactual_teacher_engine", augmentation_params.get("teacher_engine", "minimax"))
    ).strip().lower() or "minimax"
    counterfactual_teacher_level = str(
        augmentation_params.get("counterfactual_teacher_level", augmentation_params.get("teacher_level", "insane"))
    ).strip() or "insane"
    counterfactual_top_k = max(
        1,
        int(
            augmentation_params.get(
                "counterfactual_top_k",
                min(2, max_branching),
            )
        ),
    )
    counterfactual_include_exploration = _as_bool(
        augmentation_params.get("counterfactual_include_exploration", True),
        default=True,
    )
    counterfactual_exploration_seed_offset = int(augmentation_params.get("counterfactual_exploration_seed_offset", 17))

    def _select_counterfactual_moves(
        *,
        runtime_state: Any,
        legal_moves: list[int],
        sample_id: str,
        current_depth: int,
        lineage_moves: list[int],
    ) -> tuple[list[int], dict[int, str]]:
        if not legal_moves:
            return [], {}
        if len(legal_moves) <= 1:
            return list(legal_moves), {int(move): "single_legal" for move in legal_moves}

        move_scores: dict[int, float] = {}
        selection_reason: dict[int, str] = {}
        try:
            _best_move, info = _teacher_choose(
                runtime_state,
                engine=counterfactual_teacher_engine,
                level=counterfactual_teacher_level,
            )
            if counterfactual_teacher_engine == "minimax":
                raw_scores = dict(info.get("root_scores", {}))
            elif counterfactual_teacher_engine == "mcts":
                raw_scores = dict(info.get("root_q", {}))
            else:
                raw_scores = {}
            move_scores = {
                int(move): float(score)
                for move, score in raw_scores.items()
                if int(move) in legal_moves
            }
        except Exception:
            move_scores = {}

        selected_moves: list[int] = []
        remaining_moves = list(legal_moves)
        if move_scores:
            ranked = sorted(
                legal_moves,
                key=lambda move: float(move_scores.get(int(move), float("-inf"))),
                reverse=True,
            )
            teacher_k = min(max_branching, counterfactual_top_k, len(ranked))
            for move in ranked[:teacher_k]:
                if move not in selected_moves:
                    selected_moves.append(int(move))
                    selection_reason[int(move)] = "teacher_topk"
            remaining_moves = [int(move) for move in ranked if int(move) not in selected_moves]

        if counterfactual_include_exploration and remaining_moves and len(selected_moves) < max_branching:
            if move_scores:
                exploration_move = min(
                    remaining_moves,
                    key=lambda move: float(move_scores.get(int(move), float("inf"))),
                )
            else:
                hash_input = f"{sample_id}|{current_depth}|{','.join(str(move) for move in lineage_moves)}|{counterfactual_exploration_seed_offset}"
                seed_value = int(hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:8], 16)
                rng = random.Random(seed_value)
                exploration_move = int(rng.choice(remaining_moves))
            if exploration_move not in selected_moves:
                selected_moves.append(int(exploration_move))
                selection_reason[int(exploration_move)] = "exploration_alt"

        if not selected_moves:
            fallback = [int(move) for move in legal_moves[:max_branching]]
            return fallback, {int(move): "fallback_legal_order" for move in fallback}

        trimmed = [int(move) for move in selected_moves[:max_branching]]
        for move in trimmed:
            selection_reason.setdefault(int(move), "teacher_topk")
        return trimmed, selection_reason

    seen_signatures: set[str] = set()
    scanned_files = 0
    scanned_samples = 0
    selected_files = 0
    selected_samples = 0
    selected_original_samples = 0
    selected_augmented_samples = 0
    duplicate_samples = 0
    copied_raw_files = 0
    next_augmented_sample_index = 0
    augmentation_depth_breakdown: dict[str, int] = {}
    source_file_breakdown: dict[str, dict[str, int]] = {}

    for source_sampled_file in sorted(source_sampled_dir.rglob("*.jsonl")):
        if target_samples > 0 and selected_samples >= target_samples:
            break
        scanned_files += 1
        relative_path = source_sampled_file.relative_to(source_sampled_dir)
        relative_name = str(relative_path)
        kept_samples: list[dict[str, Any]] = []
        file_stats = {
            "scanned_samples": 0,
            "selected_original_samples": 0,
            "selected_augmented_samples": 0,
            "duplicate_samples": 0,
            "max_depth_reached": 0,
        }

        for sample in _iter_jsonl(source_sampled_file):
            if target_samples > 0 and selected_samples >= target_samples:
                break
            scanned_samples += 1
            file_stats["scanned_samples"] += 1

            sample_signature = _sample_position_signature(sample)
            if include_original_samples:
                if sample_signature in seen_signatures:
                    duplicate_samples += 1
                    file_stats["duplicate_samples"] += 1
                else:
                    kept_samples.append(sample)
                    seen_signatures.add(sample_signature)
                    selected_samples += 1
                    selected_original_samples += 1
                    file_stats["selected_original_samples"] += 1
                    if target_samples > 0 and selected_samples >= target_samples:
                        break

            runtime_state = _raw_state_to_runtime_state(sample["state"])
            root_sample_id = str(sample.get("sample_id", ""))
            frontier: list[tuple[Any, int, list[int], str, str]] = [
                (runtime_state, 0, [], root_sample_id, root_sample_id)
            ]
            generated_from_sample = 0
            local_signatures: set[str] = {sample_signature}

            while frontier and generated_from_sample < max_generated_per_source_sample:
                if target_samples > 0 and selected_samples >= target_samples:
                    break
                current_state, current_depth, lineage_moves, parent_sample_id, origin_sample_id = frontier.pop(0)
                if current_depth >= max_depth:
                    continue
                legal_moves = list(songo_ai_game.legal_moves(current_state))
                if not legal_moves:
                    continue

                selected_moves, selection_reason = _select_counterfactual_moves(
                    runtime_state=current_state,
                    legal_moves=[int(move) for move in legal_moves],
                    sample_id=origin_sample_id or root_sample_id or str(sample.get("sample_id", "")),
                    current_depth=current_depth,
                    lineage_moves=list(lineage_moves),
                )

                for move in selected_moves:
                    if target_samples > 0 and selected_samples >= target_samples:
                        break
                    if generated_from_sample >= max_generated_per_source_sample:
                        break

                    next_state = songo_ai_game.simulate_move(current_state, int(move))
                    augmented_sample = _sample_position(
                        game_id=f"aug_{sample['game_id']}",
                        matchup_id=f"{sample.get('matchup_id', 'derived')}_aug",
                        sample_index=next_augmented_sample_index,
                        ply=int(sample.get("ply", 0)) + current_depth + 1,
                        seed=int(sample.get("seed", 0)),
                        state=next_state,
                    )
                    next_augmented_sample_index += 1

                    if bool(augmented_sample["state"].get("is_terminal", False)) or not augmented_sample["legal_moves"]:
                        continue

                    augmented_signature = _sample_position_signature(augmented_sample)
                    if augmented_signature in seen_signatures or augmented_signature in local_signatures:
                        duplicate_samples += 1
                        file_stats["duplicate_samples"] += 1
                        continue

                    local_signatures.add(augmented_signature)
                    seen_signatures.add(augmented_signature)
                    augmented_sample["source_engine"] = "augment_existing"
                    augmented_sample["source_level"] = "derived"
                    augmented_sample["augmented_from_sample_id"] = str(sample.get("sample_id", ""))
                    augmented_sample["augmentation_depth"] = current_depth + 1
                    augmented_sample["augmentation_lineage_moves"] = lineage_moves + [int(move)]
                    augmented_sample["counterfactual_depth"] = current_depth + 1
                    augmented_sample["counterfactual_parent_sample_id"] = str(parent_sample_id)
                    augmented_sample["counterfactual_root_sample_id"] = str(origin_sample_id or root_sample_id)
                    augmented_sample["counterfactual_lineage_moves"] = lineage_moves + [int(move)]
                    augmented_sample["counterfactual_selected_by"] = str(selection_reason.get(int(move), "teacher_topk"))
                    augmented_sample["counterfactual_teacher_engine"] = counterfactual_teacher_engine
                    augmented_sample["counterfactual_teacher_level"] = counterfactual_teacher_level
                    kept_samples.append(augmented_sample)
                    selected_samples += 1
                    selected_augmented_samples += 1
                    file_stats["selected_augmented_samples"] += 1
                    generated_from_sample += 1
                    depth_key = str(current_depth + 1)
                    augmentation_depth_breakdown[depth_key] = augmentation_depth_breakdown.get(depth_key, 0) + 1
                    file_stats["max_depth_reached"] = max(file_stats["max_depth_reached"], current_depth + 1)

                    if current_depth + 1 < max_depth:
                        frontier.append(
                            (
                                next_state,
                                current_depth + 1,
                                lineage_moves + [int(move)],
                                str(augmented_sample.get("sample_id", parent_sample_id)),
                                str(origin_sample_id or root_sample_id),
                            )
                        )

        if kept_samples:
            target_sampled_file = target_sampled_dir / relative_path
            write_jsonl_atomic(target_sampled_file, kept_samples, ensure_ascii=True)
            selected_files += 1

            if source_raw_dir.exists():
                source_raw_file = source_raw_dir / relative_path.with_suffix(".json")
                target_raw_file = target_raw_dir / relative_path.with_suffix(".json")
                if source_raw_file.exists() and not target_raw_file.exists():
                    target_raw_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_raw_file, target_raw_file)
                    copied_raw_files += 1

        source_file_breakdown[relative_name] = file_stats
        if scanned_files <= 3 or scanned_files % 100 == 0:
            job.logger.info(
                "dataset generation augment progress | files=%s | scanned_samples=%s | selected_samples=%s | selected_augmented_samples=%s | duplicate_samples=%s | last_file=%s",
                scanned_files,
                scanned_samples,
                selected_samples,
                selected_augmented_samples,
                duplicate_samples,
                relative_name,
            )

    return {
        "scanned_files": scanned_files,
        "scanned_samples": scanned_samples,
        "selected_files": selected_files,
        "selected_samples": selected_samples,
        "selected_original_samples": selected_original_samples,
        "selected_augmented_samples": selected_augmented_samples,
        "duplicate_samples": duplicate_samples,
        "copied_raw_files": copied_raw_files,
        "augmentation_depth_breakdown": augmentation_depth_breakdown,
        "source_file_breakdown": source_file_breakdown,
        "augmentation_params": {
            "include_original_samples": include_original_samples,
            "max_depth": max_depth,
            "max_branching": max_branching,
            "max_generated_per_source_sample": max_generated_per_source_sample,
            "counterfactual_teacher_engine": counterfactual_teacher_engine,
            "counterfactual_teacher_level": counterfactual_teacher_level,
            "counterfactual_top_k": counterfactual_top_k,
            "counterfactual_include_exploration": counterfactual_include_exploration,
            "counterfactual_exploration_seed_offset": counterfactual_exploration_seed_offset,
        },
    }


def _merge_existing_dataset_sources(
    *,
    source_entries: list[dict[str, Any]],
    target_raw_dir: Path,
    target_sampled_dir: Path,
    target_samples: int,
    dedupe_sample_ids: bool,
) -> dict[str, Any]:
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    target_sampled_dir.mkdir(parents=True, exist_ok=True)

    seen_sample_ids: set[str] = set()
    scanned_files = 0
    scanned_samples = 0
    selected_files = 0
    selected_samples = 0
    duplicate_samples = 0
    copied_raw_files = 0
    source_breakdown: dict[str, dict[str, int]] = {}

    for source_entry in source_entries:
        source_dataset_id = str(source_entry["dataset_source_id"])
        source_raw_dir = Path(str(source_entry["raw_dir"]))
        source_sampled_dir = Path(str(source_entry["sampled_dir"]))
        source_stats = {
            "scanned_files": 0,
            "scanned_samples": 0,
            "selected_files": 0,
            "selected_samples": 0,
            "duplicate_samples": 0,
            "copied_raw_files": 0,
        }

        for source_sampled_file in sorted(source_sampled_dir.rglob("*.jsonl")):
            if target_samples > 0 and selected_samples >= target_samples:
                break
            scanned_files += 1
            source_stats["scanned_files"] += 1
            relative_path = source_sampled_file.relative_to(source_sampled_dir)
            target_sampled_file = target_sampled_dir / source_dataset_id / relative_path
            target_sampled_file.parent.mkdir(parents=True, exist_ok=True)

            kept_samples: list[dict[str, Any]] = []
            for sample in _iter_jsonl(source_sampled_file):
                scanned_samples += 1
                source_stats["scanned_samples"] += 1
                if target_samples > 0 and selected_samples >= target_samples:
                    break
                sample_id = str(sample.get("sample_id", ""))
                if dedupe_sample_ids and sample_id:
                    if sample_id in seen_sample_ids:
                        duplicate_samples += 1
                        source_stats["duplicate_samples"] += 1
                        continue
                    seen_sample_ids.add(sample_id)
                kept_samples.append(sample)
                selected_samples += 1
                source_stats["selected_samples"] += 1

            if kept_samples:
                write_jsonl_atomic(target_sampled_file, kept_samples, ensure_ascii=True)
                selected_files += 1
                source_stats["selected_files"] += 1

                source_raw_file = source_raw_dir / relative_path.with_suffix(".json")
                target_raw_file = target_raw_dir / source_dataset_id / relative_path.with_suffix(".json")
                if source_raw_file.exists() and not target_raw_file.exists():
                    target_raw_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_raw_file, target_raw_file)
                    copied_raw_files += 1
                    source_stats["copied_raw_files"] += 1

        if target_samples > 0 and selected_samples >= target_samples:
            source_breakdown[source_dataset_id] = source_stats
            break
        source_breakdown[source_dataset_id] = source_stats

    return {
        "scanned_files": scanned_files,
        "scanned_samples": scanned_samples,
        "selected_files": selected_files,
        "selected_samples": selected_samples,
        "duplicate_samples": duplicate_samples,
        "copied_raw_files": copied_raw_files,
        "source_breakdown": source_breakdown,
    }


def _append_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    guard_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _iter_jsonl(path: Path, *, open_retries: int = 1, retry_delay_seconds: float = 0.0):
    retries = max(1, int(open_retries))
    delay = max(0.0, float(retry_delay_seconds))
    for attempt in range(1, retries + 1):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return
        except FileNotFoundError:
            if attempt >= retries:
                raise
            if delay > 0.0:
                time.sleep(delay)
        except OSError:
            if attempt >= retries:
                raise
            if delay > 0.0:
                time.sleep(delay)


def _count_jsonl_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        # Fichier supprime/renomme entre le listing et l'ouverture (race multi-worker).
        return 0


def _count_total_jsonl_lines(root: Path) -> int:
    total = 0
    for path in sorted(root.rglob("*.jsonl")):
        try:
            total += _count_jsonl_lines(path)
        except FileNotFoundError:
            # Defensive fallback for highly concurrent Drive/FUSE environments.
            continue
    return total


def _format_eta_seconds(total_seconds: float | None) -> str:
    if total_seconds is None:
        return "unknown"
    remaining = max(0, int(round(total_seconds)))
    hours, remainder = divmod(remaining, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    count = int(arrays["x"].shape[0]) if "x" in arrays and arrays["x"].ndim >= 1 else 0
    if "policy_target_full" not in arrays:
        policy_index = arrays.get("policy_index", np.zeros((count,), dtype=np.int64))
        full = np.zeros((count, 7), dtype=np.float32)
        if count > 0:
            rows = np.arange(count, dtype=np.int64)
            valid = np.logical_and(policy_index >= 0, policy_index < 7)
            full[rows[valid], policy_index[valid]] = 1.0
        arrays["policy_target_full"] = full
    for key in ("capture_move_mask", "safe_move_mask", "risky_move_mask"):
        if key not in arrays:
            arrays[key] = np.zeros((count, 7), dtype=np.float32)
    if "hard_example_weight" not in arrays:
        arrays["hard_example_weight"] = np.ones((count,), dtype=np.float32)
    return arrays


def _merge_npz_splits(
    split_paths: list[Path],
    *,
    dedupe_sample_ids: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    merged_chunks: dict[str, list[np.ndarray]] = {
        "x": [],
        "legal_mask": [],
        "policy_index": [],
        "policy_target_full": [],
        "value_target": [],
        "capture_move_mask": [],
        "safe_move_mask": [],
        "risky_move_mask": [],
        "hard_example_weight": [],
        "sample_ids": [],
        "game_ids": [],
    }
    seen_sample_ids: set[str] = set()
    input_samples = 0
    kept_samples = 0
    duplicate_samples = 0
    x_dim: int | None = None

    for split_path in split_paths:
        arrays = _load_npz_arrays(split_path)
        current_x = arrays["x"]
        current_dim = int(current_x.shape[1]) if current_x.ndim == 2 else 0
        if x_dim is None:
            x_dim = current_dim
        elif current_dim != x_dim:
            raise ValueError(f"Incompatible x feature dimensions while merging splits: expected {x_dim}, got {current_dim} for {split_path}")
        sample_ids = arrays["sample_ids"]
        keep_indices: list[int] = []
        for index, sample_id in enumerate(sample_ids.tolist()):
            input_samples += 1
            sample_id_str = str(sample_id)
            if dedupe_sample_ids and sample_id_str in seen_sample_ids:
                duplicate_samples += 1
                continue
            seen_sample_ids.add(sample_id_str)
            keep_indices.append(index)

        if not keep_indices:
            continue

        keep = np.asarray(keep_indices, dtype=np.int64)
        for key in merged_chunks:
            merged_chunks[key].append(arrays[key][keep])
        kept_samples += len(keep_indices)

    if kept_samples == 0:
        merged = {
            "x": np.zeros((0, x_dim or 17), dtype=np.float32),
            "legal_mask": np.zeros((0, 7), dtype=np.float32),
            "policy_index": np.zeros((0,), dtype=np.int64),
            "policy_target_full": np.zeros((0, 7), dtype=np.float32),
            "value_target": np.zeros((0,), dtype=np.float32),
            "capture_move_mask": np.zeros((0, 7), dtype=np.float32),
            "safe_move_mask": np.zeros((0, 7), dtype=np.float32),
            "risky_move_mask": np.zeros((0, 7), dtype=np.float32),
            "hard_example_weight": np.ones((0,), dtype=np.float32),
            "sample_ids": np.asarray([], dtype=object),
            "game_ids": np.asarray([], dtype=object),
        }
    else:
        merged = {}
        for key, chunks in merged_chunks.items():
            merged[key] = np.concatenate(chunks, axis=0)

    summary = {
        "input_samples": input_samples,
        "kept_samples": kept_samples,
        "duplicate_samples": duplicate_samples,
        "unique_games": len({str(value) for value in merged["game_ids"].tolist()}),
    }
    return merged, summary


def _merge_npz_splits_with_source_breakdown(
    split_items: list[tuple[str, Path]],
    *,
    dedupe_sample_ids: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, dict[str, int]]]:
    merged_chunks: dict[str, list[np.ndarray]] = {
        "x": [],
        "legal_mask": [],
        "policy_index": [],
        "policy_target_full": [],
        "value_target": [],
        "capture_move_mask": [],
        "safe_move_mask": [],
        "risky_move_mask": [],
        "hard_example_weight": [],
        "sample_ids": [],
        "game_ids": [],
    }
    seen_sample_ids: set[str] = set()
    source_breakdown: dict[str, dict[str, int]] = {}
    input_samples = 0
    kept_samples = 0
    duplicate_samples = 0
    x_dim: int | None = None

    for source_dataset_id, split_path in split_items:
        arrays = _load_npz_arrays(split_path)
        current_x = arrays["x"]
        current_dim = int(current_x.shape[1]) if current_x.ndim == 2 else 0
        if x_dim is None:
            x_dim = current_dim
        elif current_dim != x_dim:
            raise ValueError(
                f"Incompatible x feature dimensions while merging final datasets: expected {x_dim}, got {current_dim} for {source_dataset_id}:{split_path}"
            )
        sample_ids = arrays["sample_ids"]
        keep_indices: list[int] = []
        stats = {
            "input_samples": int(len(sample_ids)),
            "kept_samples": 0,
            "duplicate_samples": 0,
            "unique_games": 0,
        }
        source_games: set[str] = set()
        for index, sample_id in enumerate(sample_ids.tolist()):
            input_samples += 1
            sample_id_str = str(sample_id)
            if dedupe_sample_ids and sample_id_str in seen_sample_ids:
                duplicate_samples += 1
                stats["duplicate_samples"] += 1
                continue
            seen_sample_ids.add(sample_id_str)
            keep_indices.append(index)
            source_games.add(str(arrays["game_ids"][index]))

        if keep_indices:
            keep = np.asarray(keep_indices, dtype=np.int64)
            for key in merged_chunks:
                merged_chunks[key].append(arrays[key][keep])
            kept_samples += len(keep_indices)
            stats["kept_samples"] = len(keep_indices)
            stats["unique_games"] = len(source_games)
        source_breakdown[source_dataset_id] = stats

    if kept_samples == 0:
        merged = {
            "x": np.zeros((0, x_dim or 17), dtype=np.float32),
            "legal_mask": np.zeros((0, 7), dtype=np.float32),
            "policy_index": np.zeros((0,), dtype=np.int64),
            "policy_target_full": np.zeros((0, 7), dtype=np.float32),
            "value_target": np.zeros((0,), dtype=np.float32),
            "capture_move_mask": np.zeros((0, 7), dtype=np.float32),
            "safe_move_mask": np.zeros((0, 7), dtype=np.float32),
            "risky_move_mask": np.zeros((0, 7), dtype=np.float32),
            "hard_example_weight": np.ones((0,), dtype=np.float32),
            "sample_ids": np.asarray([], dtype=object),
            "game_ids": np.asarray([], dtype=object),
        }
    else:
        merged = {}
        for key, chunks in merged_chunks.items():
            merged[key] = np.concatenate(chunks, axis=0)

    summary = {
        "input_samples": input_samples,
        "kept_samples": kept_samples,
        "duplicate_samples": duplicate_samples,
        "unique_games": len({str(value) for value in merged["game_ids"].tolist()}),
    }
    return merged, summary, source_breakdown


def _existing_game_numbers(
    raw_dir: Path,
    sampled_dir: Path,
    matchup_id: str,
    *,
    completion_mode: str = "raw_and_sampled",
) -> set[int]:
    raw_matchup_dir = raw_dir / matchup_id
    sampled_matchup_dir = sampled_dir / matchup_id
    raw_stems = {path.stem for path in raw_matchup_dir.glob("*.json")} if raw_matchup_dir.exists() else set()
    sampled_stems = {path.stem for path in sampled_matchup_dir.glob("*.jsonl")} if sampled_matchup_dir.exists() else set()
    normalized_mode = _normalize_completed_game_detection_mode(completion_mode)
    if normalized_mode == "sampled_only":
        completed = sampled_stems
    elif normalized_mode == "raw_only":
        completed = raw_stems
    else:
        completed = raw_stems & sampled_stems
    game_numbers: set[int] = set()
    prefix = f"{matchup_id}_game_"
    for stem in completed:
        if stem.startswith(prefix):
            suffix = stem[len(prefix) :]
            if suffix.isdigit():
                game_numbers.add(int(suffix))
    return game_numbers


def _build_pending_incremental_games(
    *,
    raw_dir: Path,
    sampled_dir: Path,
    matchup_id: str,
    games_to_add: int,
    seed_base: int,
    sample_every_n_plies: int = 1,
    completion_mode: str = "raw_and_sampled",
) -> list[dict[str, Any]]:
    normalized_mode = _normalize_completed_game_detection_mode(completion_mode)
    existing_numbers = _existing_game_numbers(
        raw_dir,
        sampled_dir,
        matchup_id,
        completion_mode=normalized_mode,
    )
    pending: list[dict[str, Any]] = []
    sample_stride = max(1, int(sample_every_n_plies))
    candidate = 1
    while len(pending) < games_to_add:
        game_id = f"{matchup_id}_game_{candidate:06d}"
        raw_game_path = raw_dir / matchup_id / f"{game_id}.json"
        sampled_game_path = sampled_dir / matchup_id / f"{game_id}.jsonl"
        if normalized_mode == "sampled_only":
            completed_artifacts_exist = sampled_game_path.exists()
        elif normalized_mode == "raw_only":
            completed_artifacts_exist = raw_game_path.exists()
        else:
            completed_artifacts_exist = raw_game_path.exists() and sampled_game_path.exists()
        if candidate not in existing_numbers or not completed_artifacts_exist:
            pending.append(
                {
                    "game_index": candidate - 1,
                    "game_no": candidate,
                    "starter": (candidate - 1) % 2,
                    "sample_ply_offset": (candidate - 1) % sample_stride,
                    "seed": seed_base + (candidate - 1),
                    "game_id": game_id,
                    "raw_game_path": raw_game_path,
                    "sampled_game_path": sampled_game_path,
                }
            )
        candidate += 1
    return pending


def _accumulate_matchup_summary(summaries: list[dict[str, Any]], summary_payload: dict[str, Any]) -> None:
    matchup_id = str(summary_payload.get("matchup_id", "")).strip()
    if not matchup_id:
        summaries.append(summary_payload)
        return
    for index, entry in enumerate(summaries):
        if str(entry.get("matchup_id", "")).strip() != matchup_id:
            continue
        merged = dict(entry)
        merged["games_added"] = int(entry.get("games_added", 0)) + int(summary_payload.get("games_added", 0))
        merged["games_failed"] = int(entry.get("games_failed", 0)) + int(summary_payload.get("games_failed", 0))
        merged["samples_added"] = int(entry.get("samples_added", 0)) + int(summary_payload.get("samples_added", 0))
        # Keep latest metadata for observability.
        merged["cycle_index"] = int(summary_payload.get("cycle_index", entry.get("cycle_index", 0) or 0))
        merged["existing_games"] = int(summary_payload.get("existing_games", entry.get("existing_games", 0) or 0))
        merged["sample_every_n_plies"] = int(summary_payload.get("sample_every_n_plies", entry.get("sample_every_n_plies", 0) or 0))
        merged["num_workers"] = int(summary_payload.get("num_workers", entry.get("num_workers", 0) or 0))
        summaries[index] = merged
        return
    summaries.append(summary_payload)


def _resolve_model_checkpoint_for_generation(model_spec: str, *, models_root: Path) -> tuple[str, Path]:
    requested = str(model_spec).strip()
    if requested in {"auto", "auto_latest"}:
        latest = latest_model_record(models_root)
        if not latest:
            raise FileNotFoundError("Aucun modele disponible dans le registre pour dataset-generate model:auto_latest.")
        model_id = str(latest.get("model_id", "")).strip()
        checkpoint_path = Path(str(latest.get("checkpoint_path", "")).strip())
    elif requested in {"auto_best", "auto_promoted_best"}:
        metadata = promoted_best_metadata(models_root)
        if not metadata:
            raise FileNotFoundError("Aucun modele promu disponible pour dataset-generate model:auto_best.")
        model_id = str(metadata.get("model_id", "promoted_best")).strip()
        checkpoint_path = models_root / "promoted" / "best" / "model.pt"
    else:
        registry = load_registry(models_root)
        record = next((item for item in registry.get("models", []) if str(item.get("model_id", "")).strip() == requested), None)
        if record:
            model_id = str(record.get("model_id", requested)).strip()
            checkpoint_path = Path(str(record.get("checkpoint_path", "")).strip())
        else:
            # Fallback robuste: si le modele n'est pas reference dans le registre
            # mais existe dans models/final/<model_id>.pt, on l'utilise quand meme.
            model_id = requested
            checkpoint_path = models_root / "final" / f"{requested}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Modele introuvable dans le registre pour dataset-generate: {requested}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable pour dataset-generate model:{requested}: {checkpoint_path}")
    return model_id, checkpoint_path


@functools.lru_cache(maxsize=None)
def _build_external_agent_cached(spec: str, models_root_str: str, device: str):
    from songo_model_stockfish.reference_songo.agents import MCTSAgent, MinimaxAgent
    from songo_model_stockfish.benchmark.model_agent import ModelAgent

    kind, level = spec.split(":", 1)
    if kind == "minimax":
        return MinimaxAgent(level)
    if kind == "mcts":
        return MCTSAgent(level)
    if kind == "model":
        model_id, checkpoint_path = _resolve_model_checkpoint_for_generation(level, models_root=Path(models_root_str))
        return ModelAgent(
            str(checkpoint_path),
            display_name=model_id,
            device=device,
            search_enabled=True,
            search_profile="fort_plusplus",
            search_depth=3,
            search_top_k=6,
            search_top_k_child=4,
            search_alpha_beta=True,
            search_policy_weight=0.35,
            search_value_weight=1.0,
        )
    raise ValueError(f"Unsupported dataset generation agent: {spec}")


def _build_external_agent(spec: str, *, models_root: Path | None = None, device: str = "cpu"):
    normalized_spec = str(spec).strip()
    if ":" not in normalized_spec:
        raise ValueError(f"Unsupported dataset generation agent: {spec}")
    kind, _level = normalized_spec.split(":", 1)
    resolved_models_root = (models_root or Path(".")).resolve()
    resolved_device = "cpu" if kind != "model" else (str(device).strip() or "cpu")
    return _build_external_agent_cached(normalized_spec, str(resolved_models_root), resolved_device)


def _state_signature_for_search(state: Any) -> tuple[Any, ...]:
    raw_state = songo_ai_game.to_raw_state(state)
    scores = raw_state.get("scores", {})
    return (
        tuple(int(value) for value in raw_state.get("board", [])),
        str(raw_state.get("player_to_move", "")),
        int(scores.get("south", 0)),
        int(scores.get("north", 0)),
        bool(raw_state.get("is_terminal", False)),
    )


def _terminal_outcome_for_player(state: Any, *, player: int) -> float:
    winner = songo_ai_game.winner(state)
    if winner is not None:
        return 1.0 if int(winner) == int(player) else -1.0
    south, north = songo_ai_game.scores(state)
    if south == north:
        return 0.0
    if int(player) == 0:
        return 1.0 if south > north else -1.0
    return 1.0 if north > south else -1.0


def _normalize_distribution_for_legal_moves(
    legal_moves: list[int],
    distribution: dict[int | str, Any] | None,
) -> dict[int, float]:
    if not legal_moves:
        return {}
    normalized: dict[int, float] = {}
    source = distribution if isinstance(distribution, dict) else {}
    for move in legal_moves:
        raw = source.get(move, source.get(str(move), 0.0))
        try:
            value = float(raw)
        except Exception:
            value = 0.0
        normalized[int(move)] = max(0.0, value)
    total = float(sum(normalized.values()))
    if total <= 0.0:
        uniform = 1.0 / float(len(legal_moves))
        return {int(move): uniform for move in legal_moves}
    return {int(move): float(value / total) for move, value in normalized.items()}


def _sample_move_from_distribution(
    distribution: dict[int, float],
    *,
    rng: np.random.Generator,
    deterministic: bool,
) -> int:
    if not distribution:
        raise ValueError("Distribution vide pour la selection de coup.")
    ordered = sorted(distribution.items(), key=lambda item: item[0])
    moves = [int(move) for move, _prob in ordered]
    probs = np.asarray([max(0.0, float(prob)) for _move, prob in ordered], dtype=np.float64)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return int(moves[0])
    probs = probs / total
    if deterministic:
        return int(moves[int(np.argmax(probs))])
    sampled_index = int(rng.choice(len(moves), p=probs))
    return int(moves[sampled_index])


class _SelfPlayTreeNode:
    __slots__ = (
        "state",
        "player_to_move",
        "legal_moves",
        "is_terminal",
        "expanded",
        "priors",
        "children",
        "edge_visits",
        "edge_value_sums",
    )

    def __init__(self, state: Any) -> None:
        self.state = state
        self.player_to_move = int(songo_ai_game.current_player(state))
        self.legal_moves = list(songo_ai_game.legal_moves(state))
        self.is_terminal = bool(songo_ai_game.is_terminal(state)) or not self.legal_moves
        self.expanded = False
        self.priors: dict[int, float] = {}
        self.children: dict[int, _SelfPlayTreeNode] = {}
        self.edge_visits: dict[int, int] = {int(move): 0 for move in self.legal_moves}
        self.edge_value_sums: dict[int, float] = {int(move): 0.0 for move in self.legal_moves}


class _SelfPlayPolicyValueModel:
    def __init__(self, checkpoint_path: str, *, model_id: str, device: str = "cpu") -> None:
        import torch
        from songo_model_stockfish.training.model import PolicyValueMLP

        self._torch = torch
        self._checkpoint_path = Path(checkpoint_path)
        self._model_id = str(model_id).strip() or self._checkpoint_path.stem
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint self-play introuvable: {self._checkpoint_path}")
        resolved_device = str(device).strip().lower() or "cpu"
        self._device = torch.device(resolved_device if (resolved_device == "cpu" or torch.cuda.is_available()) else "cpu")
        checkpoint = torch.load(self._checkpoint_path, map_location=self._device)
        model_config = checkpoint.get("model_config", {})
        self._input_dim = int(model_config.get("input_dim", 17))
        self._model = PolicyValueMLP(
            input_dim=self._input_dim,
            hidden_sizes=list(model_config.get("hidden_sizes", [256, 256, 128])),
            policy_dim=int(model_config.get("policy_dim", 7)),
            use_layer_norm=bool(model_config.get("use_layer_norm", False)),
            dropout=float(model_config.get("dropout", 0.0)),
            residual_connections=bool(model_config.get("residual_connections", False)),
        )
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.to(self._device)
        self._model.eval()
        self._inference_cache: dict[tuple[Any, ...], tuple[dict[int, float], float]] = {}
        self._cache_max_entries = 50_000

    @property
    def model_id(self) -> str:
        return self._model_id

    def infer(self, state: Any) -> tuple[dict[int, float], float]:
        legal_moves = list(songo_ai_game.legal_moves(state))
        if not legal_moves:
            player = int(songo_ai_game.current_player(state))
            return {}, _terminal_outcome_for_player(state, player=player)
        signature = _state_signature_for_search(state)
        cached = self._inference_cache.get(signature)
        if cached is not None:
            return cached

        raw_state = songo_ai_game.to_raw_state(state)
        features, legal_mask = encode_model_features(raw_state, legal_moves, tactical_analysis=None)
        features = adapt_feature_dim(features, self._input_dim)

        x = self._torch.from_numpy(features).unsqueeze(0).to(self._device)
        mask = self._torch.from_numpy(legal_mask).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            policy_logits, value = self._model(x)
            mask_value = self._torch.finfo(policy_logits.dtype).min
            masked_logits = policy_logits.masked_fill(mask <= 0, mask_value)
            probs = self._torch.softmax(masked_logits, dim=1).squeeze(0).detach().cpu().numpy()
            value_estimate = float(value.item())

        priors = {int(move): float(max(0.0, probs[int(move) - 1])) for move in legal_moves}
        priors = _normalize_distribution_for_legal_moves(legal_moves, priors)
        result = (priors, value_estimate)
        if len(self._inference_cache) >= self._cache_max_entries:
            self._inference_cache.clear()
        self._inference_cache[signature] = result
        return result


@functools.lru_cache(maxsize=8)
def _load_self_play_model_cached(
    checkpoint_path: str,
    model_id: str,
    device: str,
) -> _SelfPlayPolicyValueModel:
    return _SelfPlayPolicyValueModel(
        checkpoint_path,
        model_id=model_id,
        device=device,
    )


def _expand_self_play_node(
    node: _SelfPlayTreeNode,
    *,
    model: _SelfPlayPolicyValueModel,
) -> float:
    if node.expanded:
        if node.is_terminal:
            return _terminal_outcome_for_player(node.state, player=node.player_to_move)
        priors, value_estimate = model.infer(node.state)
        node.priors = _normalize_distribution_for_legal_moves(node.legal_moves, priors)
        return float(value_estimate)
    node.expanded = True
    if node.is_terminal:
        return _terminal_outcome_for_player(node.state, player=node.player_to_move)
    priors, value_estimate = model.infer(node.state)
    node.priors = _normalize_distribution_for_legal_moves(node.legal_moves, priors)
    return float(value_estimate)


def _run_single_puct_simulation(
    node: _SelfPlayTreeNode,
    *,
    model: _SelfPlayPolicyValueModel,
    c_puct: float,
    root_prior_override: dict[int, float] | None = None,
    is_root: bool = False,
) -> float:
    if node.is_terminal:
        return _terminal_outcome_for_player(node.state, player=node.player_to_move)
    if not node.expanded:
        return _expand_self_play_node(node, model=model)

    total_visits = max(1, sum(int(node.edge_visits.get(move, 0)) for move in node.legal_moves))
    best_move = int(node.legal_moves[0])
    best_score = float("-inf")
    for move in node.legal_moves:
        move_int = int(move)
        prior_source = root_prior_override if (is_root and root_prior_override is not None) else node.priors
        prior = float(prior_source.get(move_int, node.priors.get(move_int, 0.0)))
        visit_count = int(node.edge_visits.get(move_int, 0))
        value_sum = float(node.edge_value_sums.get(move_int, 0.0))
        q_value = (value_sum / visit_count) if visit_count > 0 else 0.0
        exploration = float(c_puct) * prior * math.sqrt(float(total_visits)) / float(1 + visit_count)
        score = q_value + exploration
        if score > best_score:
            best_score = score
            best_move = move_int

    child = node.children.get(best_move)
    if child is None:
        child_state = songo_ai_game.simulate_move(node.state, int(best_move))
        child = _SelfPlayTreeNode(child_state)
        node.children[int(best_move)] = child
    child_value = _run_single_puct_simulation(
        child,
        model=model,
        c_puct=c_puct,
        root_prior_override=None,
        is_root=False,
    )
    value_from_current_player = -float(child_value)
    node.edge_visits[int(best_move)] = int(node.edge_visits.get(int(best_move), 0)) + 1
    node.edge_value_sums[int(best_move)] = float(node.edge_value_sums.get(int(best_move), 0.0)) + value_from_current_player
    return value_from_current_player


def _visit_counts_to_policy_distribution(
    legal_moves: list[int],
    visit_counts: dict[int, int],
    *,
    temperature: float,
) -> dict[int, float]:
    if not legal_moves:
        return {}
    temp = float(temperature)
    if temp <= 1e-6:
        best_move = max(legal_moves, key=lambda move: int(visit_counts.get(int(move), 0)))
        return {int(move): (1.0 if int(move) == int(best_move) else 0.0) for move in legal_moves}

    exponent = 1.0 / max(1e-6, temp)
    values = np.asarray(
        [max(0.0, float(int(visit_counts.get(int(move), 0)))) ** exponent for move in legal_moves],
        dtype=np.float64,
    )
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        uniform = 1.0 / float(len(legal_moves))
        return {int(move): uniform for move in legal_moves}
    probs = values / total
    return {int(move): float(prob) for move, prob in zip(legal_moves, probs.tolist())}


def _run_self_play_puct_search(
    state: Any,
    *,
    model: _SelfPlayPolicyValueModel,
    rng: np.random.Generator,
    num_simulations: int,
    c_puct: float,
    root_dirichlet_alpha: float,
    root_dirichlet_epsilon: float,
    temperature: float,
    deterministic: bool,
) -> dict[str, Any]:
    root = _SelfPlayTreeNode(songo_ai_game.clone_state(state))
    root_value = _expand_self_play_node(root, model=model)
    if root.is_terminal or not root.legal_moves:
        return {
            "move": None,
            "policy_distribution": {},
            "visit_counts": {},
            "root_value": float(root_value),
        }

    root_prior_override: dict[int, float] | None = None
    epsilon = max(0.0, min(1.0, float(root_dirichlet_epsilon)))
    if epsilon > 0.0 and len(root.legal_moves) > 1:
        alpha = max(1e-3, float(root_dirichlet_alpha))
        noise = rng.dirichlet([alpha] * len(root.legal_moves)).tolist()
        root_prior_override = {}
        for move, noise_value in zip(root.legal_moves, noise):
            base_prior = float(root.priors.get(int(move), 0.0))
            root_prior_override[int(move)] = ((1.0 - epsilon) * base_prior) + (epsilon * float(noise_value))
        root_prior_override = _normalize_distribution_for_legal_moves(root.legal_moves, root_prior_override)

    for _ in range(max(1, int(num_simulations))):
        _run_single_puct_simulation(
            root,
            model=model,
            c_puct=float(c_puct),
            root_prior_override=root_prior_override,
            is_root=True,
        )

    visit_counts = {int(move): int(root.edge_visits.get(int(move), 0)) for move in root.legal_moves}
    if sum(visit_counts.values()) <= 0:
        visit_counts = {int(move): 1 for move in root.legal_moves}
    policy_distribution = _visit_counts_to_policy_distribution(
        root.legal_moves,
        visit_counts,
        temperature=float(temperature),
    )
    move = _sample_move_from_distribution(policy_distribution, rng=rng, deterministic=bool(deterministic))
    return {
        "move": int(move),
        "policy_distribution": policy_distribution,
        "visit_counts": visit_counts,
        "root_value": float(root_value),
    }


def _outcome_value_for_player_to_move(
    *,
    winner_index: int | None,
    final_scores: tuple[int, int],
    player_to_move_label: str,
) -> float:
    player = 0 if str(player_to_move_label).strip().lower() == "south" else 1
    if winner_index is not None:
        return 1.0 if int(winner_index) == int(player) else -1.0
    south, north = final_scores
    if south == north:
        return 0.0
    if int(player) == 0:
        return 1.0 if south > north else -1.0
    return 1.0 if north > south else -1.0


def _play_and_sample_self_play_game(
    *,
    model: _SelfPlayPolicyValueModel,
    matchup_id: str,
    game_id: str,
    seed: int,
    sample_every_n_plies: int,
    sample_ply_offset: int = 0,
    max_moves: int,
    num_simulations: int,
    c_puct: float,
    temperature: float,
    temperature_end: float,
    temperature_drop_ply: int,
    root_dirichlet_alpha: float,
    root_dirichlet_epsilon: float,
    deterministic: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = songo_ai_game.create_state()
    started_at = utc_now_iso()
    moves: list[int] = []
    samples: list[dict[str, Any]] = []
    sample_index = 0
    ply = 0
    rng = np.random.default_rng(int(seed))
    end_reason = "finished"
    sample_stride = max(1, int(sample_every_n_plies))
    sample_offset = int(sample_ply_offset) % sample_stride

    while not songo_ai_game.is_terminal(state) and ply < int(max_moves):
        legal_moves = list(songo_ai_game.legal_moves(state))
        if not legal_moves:
            end_reason = "no_legal_moves_available"
            break
        active_temperature = float(temperature if ply < int(max(0, temperature_drop_ply)) else temperature_end)
        search = _run_self_play_puct_search(
            state,
            model=model,
            rng=rng,
            num_simulations=int(num_simulations),
            c_puct=float(c_puct),
            root_dirichlet_alpha=float(root_dirichlet_alpha),
            root_dirichlet_epsilon=float(root_dirichlet_epsilon),
            temperature=active_temperature,
            deterministic=bool(deterministic),
        )
        move = search.get("move")
        if move is None or int(move) not in legal_moves:
            move = int(legal_moves[0])

        if ((int(ply) - int(sample_offset)) % int(sample_stride)) == 0:
            sample_index += 1
            sample = _sample_position(
                game_id=game_id,
                matchup_id=matchup_id,
                sample_index=sample_index,
                ply=ply,
                seed=seed,
                state=state,
            )
            policy_distribution = _normalize_distribution_for_legal_moves(
                legal_moves,
                search.get("policy_distribution", {}),
            )
            best_move = int(max(policy_distribution.items(), key=lambda item: item[1])[0])
            sample["policy_target"] = {
                "best_move": best_move,
                "distribution": {str(key): float(value) for key, value in policy_distribution.items()},
            }
            sample["source_engine"] = "self_play_puct"
            sample["source_level"] = model.model_id
            sample["self_play_player_to_move"] = sample.get("player_to_move")
            sample["self_play_search"] = {
                "num_simulations": int(num_simulations),
                "c_puct": float(c_puct),
                "temperature": float(active_temperature),
                "root_dirichlet_alpha": float(root_dirichlet_alpha),
                "root_dirichlet_epsilon": float(root_dirichlet_epsilon),
                "deterministic": bool(deterministic),
                "visit_counts": {str(key): int(value) for key, value in dict(search.get("visit_counts", {})).items()},
                "root_value": float(search.get("root_value", 0.0)),
            }
            samples.append(sample)

        moves.append(int(move))
        state = songo_ai_game.simulate_move(state, int(move))
        ply += 1

    winner_index = songo_ai_game.winner(state)
    final_scores = songo_ai_game.scores(state)
    for sample in samples:
        player_label = str(sample.get("self_play_player_to_move", sample.get("player_to_move", "south")))
        sample["value_target"] = float(
            _outcome_value_for_player_to_move(
                winner_index=winner_index,
                final_scores=final_scores,
                player_to_move_label=player_label,
            )
        )
        sample.pop("self_play_player_to_move", None)

    winner_label: str | None
    if winner_index is None:
        winner_label = None
    else:
        winner_label = "south" if int(winner_index) == 0 else "north"
    raw_log = {
        "game_id": game_id,
        "matchup_id": matchup_id,
        "seed": int(seed),
        "starter": 0,
        "player_a": f"model:{model.model_id}",
        "player_b": f"model:{model.model_id}",
        "winner": winner_label,
        "winner_index": (None if winner_index is None else int(winner_index)),
        "moves": moves,
        "ply_count": int(ply),
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "scores": [int(final_scores[0]), int(final_scores[1])],
        "reason": "finished" if songo_ai_game.is_terminal(state) else (end_reason if end_reason != "finished" else f"max_moves_reached:{max_moves}"),
        "self_play": {
            "model_id": model.model_id,
            "num_simulations": int(num_simulations),
            "c_puct": float(c_puct),
            "temperature": float(temperature),
            "temperature_end": float(temperature_end),
            "temperature_drop_ply": int(temperature_drop_ply),
            "root_dirichlet_alpha": float(root_dirichlet_alpha),
            "root_dirichlet_epsilon": float(root_dirichlet_epsilon),
            "deterministic": bool(deterministic),
            "sample_every_n_plies": int(sample_stride),
            "sample_ply_offset": int(sample_offset),
        },
    }
    return raw_log, samples


def _play_and_sample_self_play_game_from_checkpoint(
    *,
    checkpoint_path: str,
    model_id: str,
    model_device: str,
    matchup_id: str,
    game_id: str,
    seed: int,
    sample_every_n_plies: int,
    sample_ply_offset: int = 0,
    max_moves: int,
    num_simulations: int,
    c_puct: float,
    temperature: float,
    temperature_end: float,
    temperature_drop_ply: int,
    root_dirichlet_alpha: float,
    root_dirichlet_epsilon: float,
    deterministic: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = _load_self_play_model_cached(
        str(checkpoint_path),
        str(model_id),
        str(model_device),
    )
    return _play_and_sample_self_play_game(
        model=model,
        matchup_id=matchup_id,
        game_id=game_id,
        seed=int(seed),
        sample_every_n_plies=int(sample_every_n_plies),
        sample_ply_offset=int(sample_ply_offset),
        max_moves=int(max_moves),
        num_simulations=int(num_simulations),
        c_puct=float(c_puct),
        temperature=float(temperature),
        temperature_end=float(temperature_end),
        temperature_drop_ply=int(temperature_drop_ply),
        root_dirichlet_alpha=float(root_dirichlet_alpha),
        root_dirichlet_epsilon=float(root_dirichlet_epsilon),
        deterministic=bool(deterministic),
    )


def _teacher_choose(state: Any, *, engine: str, level: str) -> tuple[int, dict[str, Any]]:
    if engine == "minimax":
        from songo_model_stockfish.reference_songo.levels import get_config
        from songo_model_stockfish.reference_songo.minimax import choose_move

        return choose_move(songo_ai_game.clone_state(state), get_config(level))
    if engine == "mcts":
        from songo_model_stockfish.reference_songo.levels import get_mcts_config
        from songo_model_stockfish.reference_songo.mcts import choose_move

        return choose_move(songo_ai_game.clone_state(state), get_mcts_config(level))
    raise ValueError(f"Unsupported teacher engine: {engine}")


def _parse_matchup(matchup_spec: str) -> tuple[str, str]:
    parts = matchup_spec.split(" vs ")
    if len(parts) != 2:
        raise ValueError(f"Invalid matchup spec: {matchup_spec}")
    return parts[0].strip(), parts[1].strip()


def _resolve_benchmark_review_summary_path(job: JobContext, cfg: dict[str, Any]) -> Path | None:
    configured = str(cfg.get("benchmark_review_summary_path", "")).strip()
    if configured:
        candidate = _resolve_storage_path(job.paths.drive_root, configured, Path(configured))
        return candidate

    review_model = str(cfg.get("benchmark_review_model", "auto_best")).strip() or "auto_best"
    resolved_model_id = ""
    if review_model in {"auto_best", "auto_promoted_best"}:
        metadata = promoted_best_metadata(job.paths.models_root)
        resolved_model_id = str(metadata.get("model_id", "")).strip() if metadata else ""
    elif review_model in {"auto", "auto_latest"}:
        latest = latest_model_record(job.paths.models_root)
        resolved_model_id = str(latest.get("model_id", "")).strip() if latest else ""
    else:
        resolved_model_id = review_model
    if not resolved_model_id:
        return None

    registry = load_registry(job.paths.models_root)
    model_record = next(
        (
            item
            for item in registry.get("models", [])
            if str(item.get("model_id", "")).strip() == resolved_model_id
        ),
        None,
    )
    if not isinstance(model_record, dict):
        return None
    summary_path_raw = str(model_record.get("benchmark_summary_path", "")).strip()
    if not summary_path_raw:
        return None
    return _resolve_storage_path(job.paths.drive_root, summary_path_raw, Path(summary_path_raw))


def _augment_matchups_with_benchmark_review(
    job: JobContext,
    *,
    cfg: dict[str, Any],
    base_matchups: list[str],
) -> tuple[list[str], dict[str, Any]]:
    enabled = _as_bool(cfg.get("benchmark_review_enabled", False), default=False)
    if not enabled:
        return list(base_matchups), {"enabled": False, "added": 0}

    summary_path = _resolve_benchmark_review_summary_path(job, cfg)
    if summary_path is None or not summary_path.exists():
        job.logger.warning(
            "dataset generation benchmark review enabled but summary missing | path=%s",
            summary_path,
        )
        return list(base_matchups), {
            "enabled": True,
            "added": 0,
            "reason": "summary_missing",
            "summary_path": str(summary_path) if summary_path is not None else "",
        }

    payload = _read_json_file(summary_path, default={})
    matchups_payload = payload.get("matchups", [])
    if not isinstance(matchups_payload, list) or not matchups_payload:
        return list(base_matchups), {
            "enabled": True,
            "added": 0,
            "reason": "empty_matchups",
            "summary_path": str(summary_path),
        }

    engine = str(payload.get("engine", "")).strip()
    if not engine:
        return list(base_matchups), {
            "enabled": True,
            "added": 0,
            "reason": "missing_engine",
            "summary_path": str(summary_path),
        }

    top_k = max(1, int(cfg.get("benchmark_review_top_k", 2)))
    repeat_factor = max(1, int(cfg.get("benchmark_review_repeat_factor", 2)))
    winrate_threshold = float(cfg.get("benchmark_review_max_winrate", 0.35))
    max_added_matchups = max(1, int(cfg.get("benchmark_review_max_added_matchups", 12)))
    include_only_supported = _as_bool(cfg.get("benchmark_review_supported_only", True), default=True)

    normalized: list[dict[str, Any]] = []
    for item in matchups_payload:
        if not isinstance(item, dict):
            continue
        opponent = str(item.get("opponent", "")).strip()
        if not opponent or ":" not in opponent:
            continue
        if include_only_supported:
            kind = opponent.split(":", 1)[0].strip().lower()
            if kind not in {"minimax", "mcts"}:
                continue
        try:
            winrate = float(item.get("winrate", 0.0))
        except Exception:
            winrate = 0.0
        try:
            score_rate = float(item.get("score_rate", winrate))
        except Exception:
            score_rate = winrate
        normalized.append(
            {
                "opponent": opponent,
                "winrate": winrate,
                "score_rate": score_rate,
            }
        )

    if not normalized:
        return list(base_matchups), {
            "enabled": True,
            "added": 0,
            "reason": "no_supported_opponents",
            "summary_path": str(summary_path),
        }

    normalized.sort(key=lambda item: (float(item["winrate"]), float(item["score_rate"])))
    selected = [item for item in normalized if float(item["winrate"]) <= winrate_threshold]
    if not selected:
        selected = normalized[:top_k]
    else:
        selected = selected[:top_k]

    added: list[str] = []
    for item in selected:
        matchup = f"model:{engine} vs {item['opponent']}"
        for _ in range(repeat_factor):
            added.append(matchup)
            if len(added) >= max_added_matchups:
                break
        if len(added) >= max_added_matchups:
            break

    final_matchups = list(base_matchups) + added
    info = {
        "enabled": True,
        "added": len(added),
        "summary_path": str(summary_path),
        "engine": engine,
        "selected_opponents": [str(item.get("opponent", "")) for item in selected],
        "repeat_factor": repeat_factor,
        "top_k": top_k,
        "max_winrate": winrate_threshold,
        "max_added_matchups": max_added_matchups,
    }
    return final_matchups, info


def _resolve_tournament_review_summary_path(job: JobContext, cfg: dict[str, Any]) -> Path | None:
    configured = str(cfg.get("tournament_review_summary_path", "")).strip()
    if configured:
        return _resolve_storage_path(job.paths.drive_root, configured, Path(configured))

    tournaments_dir = _resolve_storage_path(
        job.paths.drive_root,
        "reports/benchmarks/model_tournaments",
        job.paths.drive_root / "reports" / "benchmarks" / "model_tournaments",
    )
    if not tournaments_dir.exists():
        return None
    candidates = sorted(
        tournaments_dir.glob("model_tournament_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_tournament_review_focus_model(
    job: JobContext,
    *,
    cfg: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    requested = str(cfg.get("tournament_review_focus_model", "auto_tournament_winner")).strip() or "auto_tournament_winner"
    normalized = requested.lower()
    if normalized in {"auto_tournament_winner", "auto_winner"}:
        auto_actions = payload.get("auto_actions", {})
        if isinstance(auto_actions, dict):
            winner_model_id = str(auto_actions.get("winner_model_id", "")).strip()
            if winner_model_id:
                return winner_model_id
        ranking = payload.get("ranking", [])
        if isinstance(ranking, list) and ranking:
            first = ranking[0]
            if isinstance(first, dict):
                first_model_id = str(first.get("model_id", "")).strip()
                if first_model_id:
                    return first_model_id
        return ""
    if normalized in {"auto_best", "auto_promoted_best"}:
        metadata = promoted_best_metadata(job.paths.models_root)
        return str(metadata.get("model_id", "")).strip() if isinstance(metadata, dict) else ""
    if normalized in {"auto_latest", "auto"}:
        latest = latest_model_record(job.paths.models_root)
        return str(latest.get("model_id", "")).strip() if isinstance(latest, dict) else ""
    return requested


def _augment_matchups_with_tournament_review(
    job: JobContext,
    *,
    cfg: dict[str, Any],
    base_matchups: list[str],
) -> tuple[list[str], dict[str, Any]]:
    enabled = _as_bool(cfg.get("tournament_review_enabled", False), default=False)
    if not enabled:
        return list(base_matchups), {"enabled": False, "added": 0}

    summary_path = _resolve_tournament_review_summary_path(job, cfg)
    if summary_path is None or not summary_path.exists():
        job.logger.warning(
            "dataset generation tournament review enabled but summary missing | path=%s",
            summary_path,
        )
        return list(base_matchups), {
            "enabled": True,
            "added": 0,
            "reason": "summary_missing",
            "summary_path": str(summary_path) if summary_path is not None else "",
        }

    payload = _read_json_file(summary_path, default={})
    if not isinstance(payload, dict):
        return list(base_matchups), {
            "enabled": True,
            "added": 0,
            "reason": "invalid_payload",
            "summary_path": str(summary_path),
        }

    focus_model = _resolve_tournament_review_focus_model(job, cfg=cfg, payload=payload)
    if not focus_model:
        return list(base_matchups), {
            "enabled": True,
            "added": 0,
            "reason": "missing_focus_model",
            "summary_path": str(summary_path),
        }

    pairs_payload = payload.get("pairs", [])
    top_k = max(1, int(cfg.get("tournament_review_top_k", 4)))
    repeat_factor = max(1, int(cfg.get("tournament_review_repeat_factor", 2)))
    max_added_matchups = max(1, int(cfg.get("tournament_review_max_added_matchups", 12)))
    max_score_rate = float(cfg.get("tournament_review_max_score_rate", 0.55))
    include_reverse = _as_bool(cfg.get("tournament_review_include_reverse_matchup", True), default=True)

    # opponent -> {"weighted_score_sum": float, "weighted_max_score_sum": float, "games": int}
    aggregated: dict[str, dict[str, float]] = {}
    if isinstance(pairs_payload, list):
        for item in pairs_payload:
            if not isinstance(item, dict):
                continue
            model_a = str(item.get("model_a", "")).strip()
            model_b = str(item.get("model_b", "")).strip()
            if not model_a or not model_b:
                continue
            if focus_model not in {model_a, model_b}:
                continue
            opponent = model_b if model_a == focus_model else model_a
            if not opponent or opponent == focus_model:
                continue

            games = max(1, int(item.get("games", 0) or 0))
            draws = max(0, int(item.get("draws", 0) or 0))
            wins_focus = max(
                0,
                int(item.get("wins_a", 0) or 0) if model_a == focus_model else int(item.get("wins_b", 0) or 0),
            )
            points_focus_default = (wins_focus * 3) + draws
            points_focus = (
                max(0, int(item.get("points_a", points_focus_default) or points_focus_default))
                if model_a == focus_model
                else max(0, int(item.get("points_b", points_focus_default) or points_focus_default))
            )
            max_points = max(1, games * 3)
            row = aggregated.setdefault(
                opponent,
                {"weighted_score_sum": 0.0, "weighted_max_score_sum": 0.0, "games": 0.0},
            )
            row["weighted_score_sum"] += float(points_focus)
            row["weighted_max_score_sum"] += float(max_points)
            row["games"] += float(games)

    normalized: list[dict[str, Any]] = []
    for opponent, stats in aggregated.items():
        denom = float(stats.get("weighted_max_score_sum", 0.0) or 0.0)
        score_rate = (float(stats.get("weighted_score_sum", 0.0)) / denom) if denom > 0 else 0.0
        normalized.append(
            {
                "opponent": opponent,
                "score_rate": score_rate,
                "games": int(stats.get("games", 0.0) or 0.0),
            }
        )

    if not normalized:
        ranking_payload = payload.get("ranking", [])
        ranking_model_ids: list[str] = []
        if isinstance(ranking_payload, list):
            for row in ranking_payload:
                if not isinstance(row, dict):
                    continue
                model_id = str(row.get("model_id", "")).strip()
                if model_id and model_id != focus_model:
                    ranking_model_ids.append(model_id)
        if not ranking_model_ids:
            return list(base_matchups), {
                "enabled": True,
                "added": 0,
                "reason": "no_pairs_for_focus_model",
                "summary_path": str(summary_path),
                "focus_model": focus_model,
            }
        normalized = [
            {"opponent": model_id, "score_rate": 0.0, "games": 0}
            for model_id in ranking_model_ids[:top_k]
        ]

    normalized.sort(key=lambda item: (float(item["score_rate"]), -int(item["games"]), str(item["opponent"])))
    selected = [item for item in normalized if float(item["score_rate"]) <= max_score_rate]
    if not selected:
        selected = normalized[:top_k]
    else:
        selected = selected[:top_k]

    added_raw: list[str] = []
    for item in selected:
        opponent = str(item.get("opponent", "")).strip()
        if not opponent or opponent == focus_model:
            continue
        direct_matchup = f"model:{focus_model} vs model:{opponent}"
        reverse_matchup = f"model:{opponent} vs model:{focus_model}"
        for _ in range(repeat_factor):
            added_raw.append(direct_matchup)
            if include_reverse:
                added_raw.append(reverse_matchup)
            if len(added_raw) >= max_added_matchups:
                break
        if len(added_raw) >= max_added_matchups:
            break

    existing = {str(matchup).strip() for matchup in base_matchups if str(matchup).strip()}
    added_unique: list[str] = []
    for matchup in added_raw:
        normalized_matchup = str(matchup).strip()
        if not normalized_matchup or normalized_matchup in existing:
            continue
        existing.add(normalized_matchup)
        added_unique.append(normalized_matchup)
        if len(added_unique) >= max_added_matchups:
            break

    final_matchups = list(base_matchups) + added_unique
    info = {
        "enabled": True,
        "added": len(added_unique),
        "summary_path": str(summary_path),
        "focus_model": focus_model,
        "selected_opponents": [str(item.get("opponent", "")) for item in selected],
        "repeat_factor": repeat_factor,
        "top_k": top_k,
        "max_score_rate": max_score_rate,
        "max_added_matchups": max_added_matchups,
        "include_reverse_matchup": include_reverse,
    }
    return final_matchups, info


def _validate_generation_agent_spec(
    spec: str,
    *,
    models_root: Path,
    cache: dict[str, tuple[bool, str]],
) -> tuple[bool, str]:
    normalized = str(spec).strip()
    if not normalized:
        return False, "agent vide"
    cached = cache.get(normalized)
    if cached is not None:
        return cached
    if ":" not in normalized:
        result = (False, f"agent invalide (format attendu kind:level): {normalized}")
        cache[normalized] = result
        return result
    kind, level = normalized.split(":", 1)
    kind = kind.strip().lower()
    level = str(level).strip()
    if kind in {"minimax", "mcts"}:
        result = (True, "")
        cache[normalized] = result
        return result
    if kind == "model":
        try:
            _resolve_model_checkpoint_for_generation(level, models_root=models_root)
            result = (True, "")
            cache[normalized] = result
            return result
        except Exception as exc:
            result = (False, f"{type(exc).__name__}: {exc}")
            cache[normalized] = result
            return result
    result = (False, f"agent non supporte: {kind}")
    cache[normalized] = result
    return result


def _sample_position(
    *,
    game_id: str,
    matchup_id: str,
    sample_index: int,
    ply: int,
    seed: int,
    state: Any,
) -> dict[str, Any]:
    raw_state = songo_ai_game.to_raw_state(state)
    raw_state["turn_index"] = ply
    legal_moves = songo_ai_game.legal_moves(state)
    effective_terminal = bool(raw_state.get("is_terminal", False)) or not legal_moves
    raw_state["is_terminal"] = effective_terminal
    return {
        "sample_id": f"{game_id}_sample_{sample_index:06d}",
        "game_id": game_id,
        "matchup_id": matchup_id,
        "ply": ply,
        "seed": seed,
        "player_to_move": raw_state["player_to_move"],
        "state": raw_state,
        "legal_moves": [] if effective_terminal else legal_moves,
        "source_engine": "match_replay",
        "source_level": "mixed",
    }


def _raw_state_to_runtime_state(raw_state: dict[str, Any]) -> Any:
    board_flat = list(raw_state["board"])
    if len(board_flat) != 14:
        raise ValueError("Invalid raw_state board length")
    current_player = 0 if raw_state["player_to_move"] == "south" else 1
    scores = raw_state["scores"]
    return {
        "board": [board_flat[:7], board_flat[7:]],
        "scores": [int(scores["south"]), int(scores["north"])],
        "current_player": current_player,
        "finished": bool(raw_state.get("is_terminal", False)),
        "winner": None,
        "reason": "",
        "turn_index": int(raw_state.get("turn_index", 0)),
    }


def _normalize_value(value: float) -> float:
    clipped = max(-1_000_000.0, min(1_000_000.0, float(value)))
    return float(np.tanh(clipped / 200.0))


def _clip_unit(value: float) -> float:
    return float(max(-1.0, min(1.0, float(value))))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_sample_outcome_value_with_presence(sample: dict[str, Any]) -> tuple[float, bool]:
    if "game_outcome_for_player_to_move" not in sample:
        return 0.0, False
    raw = sample.get("game_outcome_for_player_to_move", 0.0)
    if raw is None:
        return 0.0, False
    value = _safe_float(raw, 0.0)
    if not np.isfinite(value):
        return 0.0, False
    return _clip_unit(value), True


def _safe_sample_outcome_value(sample: dict[str, Any]) -> float:
    value, _present = _safe_sample_outcome_value_with_presence(sample)
    return float(value)


def _compute_hard_example_annotation(
    *,
    teacher_move_scores: dict[int, float],
    legal_moves: list[int],
    best_move: int,
    outcome_value: float,
    enabled: bool,
    margin_threshold: float,
    outcome_focus: float,
    weight_multiplier: float,
) -> tuple[float, float, float, float]:
    if not enabled:
        return 0.0, 0.0, 0.0, 1.0

    best_score = float(teacher_move_scores.get(int(best_move), 0.0))
    second_best_score = float(best_score)
    if len(legal_moves) > 1:
        ordered = sorted(
            [float(teacher_move_scores.get(int(move), best_score)) for move in legal_moves],
            reverse=True,
        )
        best_score = float(ordered[0])
        second_best_score = float(ordered[1])
    normalized_margin = max(0.0, _normalize_value(best_score) - _normalize_value(second_best_score))
    safe_threshold = max(1e-6, float(margin_threshold))
    margin_hardness = max(0.0, 1.0 - min(1.0, normalized_margin / safe_threshold))
    outcome_hardness = max(0.0, -_clip_unit(outcome_value))
    outcome_focus_clamped = max(0.0, min(1.0, float(outcome_focus)))
    hard_score = max(
        0.0,
        min(
            1.0,
            ((1.0 - outcome_focus_clamped) * margin_hardness) + (outcome_focus_clamped * outcome_hardness),
        ),
    )
    multiplier = max(1.0, float(weight_multiplier))
    hard_weight = 1.0 + (hard_score * (multiplier - 1.0))
    return float(normalized_margin), float(margin_hardness), float(hard_score), float(hard_weight)


def _build_policy_distribution_from_scores(
    legal_moves: list[int],
    move_scores: dict[int, float],
    *,
    temperature: float = 0.35,
) -> dict[str, float]:
    distribution = {str(move): 0.0 for move in legal_moves}
    if not legal_moves:
        return distribution
    if len(legal_moves) == 1:
        distribution[str(legal_moves[0])] = 1.0
        return distribution

    logits = np.asarray(
        [_normalize_value(move_scores.get(int(move), 0.0)) / max(temperature, 1e-6) for move in legal_moves],
        dtype=np.float64,
    )
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        uniform = 1.0 / float(len(legal_moves))
        for move in legal_moves:
            distribution[str(move)] = uniform
        return distribution

    probs = weights / total
    for move, prob in zip(legal_moves, probs.tolist()):
        distribution[str(move)] = float(prob)
    return distribution


def _build_tactical_analysis(
    runtime_state: Any,
    legal_moves: list[int],
    move_scores: dict[int, float],
    *,
    best_move: int,
) -> dict[str, Any]:
    return build_runtime_tactical_analysis(
        runtime_state,
        legal_moves,
        move_scores=move_scores,
        best_move=best_move,
    )
def _label_sample(
    sample: dict[str, Any],
    *,
    teacher_engine: str,
    teacher_level: str,
    include_tactical_analysis: bool = True,
    value_target_mix_teacher_weight: float = 1.0,
    hard_examples_enabled: bool = True,
    hard_examples_margin_threshold: float = 0.08,
    hard_examples_outcome_focus: float = 0.35,
    hard_examples_weight_multiplier: float = 2.0,
) -> dict[str, Any]:
    if bool(sample["state"].get("is_terminal", False)):
        raise ValueError(f"Cannot label terminal sample: {sample['sample_id']}")

    if not sample["legal_moves"]:
        raise ValueError(f"Cannot label sample without legal moves: {sample['sample_id']}")

    runtime_state = _raw_state_to_runtime_state(sample["state"])
    best_move, info = _teacher_choose(runtime_state, engine=teacher_engine, level=teacher_level)
    legal_moves = list(sample["legal_moves"])
    move_scores: dict[int, float] = {}
    if teacher_engine == "minimax":
        move_scores = {int(move): float(score) for move, score in dict(info.get("root_scores", {})).items() if int(move) in legal_moves}
    elif teacher_engine == "mcts":
        root_q = info.get("root_q", {})
        move_scores = {int(move): float(score) for move, score in dict(root_q).items() if int(move) in legal_moves}

    if not move_scores:
        fallback_score = float(info.get("score", 0.0))
        move_scores = {int(move): fallback_score for move in legal_moves}

    if best_move not in legal_moves and legal_moves:
        best_move = legal_moves[0]
    if legal_moves:
        best_move = max(legal_moves, key=lambda move: float(move_scores.get(int(move), float("-inf"))))

    teacher_score = float(move_scores.get(int(best_move), info.get("score", 0.0)))
    teacher_value_target = _normalize_value(teacher_score)
    outcome_value_target, outcome_available = _safe_sample_outcome_value_with_presence(sample)
    teacher_mix = max(0.0, min(1.0, float(value_target_mix_teacher_weight)))
    if not bool(outcome_available):
        teacher_mix = 1.0
    outcome_mix = 1.0 - teacher_mix
    mixed_value_target = _clip_unit((teacher_mix * float(teacher_value_target)) + (outcome_mix * float(outcome_value_target)))
    normalized_margin, margin_hardness, hard_example_score, hard_example_weight = _compute_hard_example_annotation(
        teacher_move_scores=move_scores,
        legal_moves=legal_moves,
        best_move=int(best_move),
        outcome_value=outcome_value_target,
        enabled=bool(hard_examples_enabled),
        margin_threshold=float(hard_examples_margin_threshold),
        outcome_focus=float(hard_examples_outcome_focus),
        weight_multiplier=float(hard_examples_weight_multiplier),
    )

    labeled = dict(sample)
    labeled["teacher_engine"] = teacher_engine
    labeled["teacher_level"] = teacher_level
    labeled["teacher_move_scores"] = {str(move): float(move_scores.get(int(move), 0.0)) for move in legal_moves}
    labeled["policy_target"] = {
        "best_move": int(best_move),
        "distribution": _build_policy_distribution_from_scores(legal_moves, move_scores),
    }
    if include_tactical_analysis:
        labeled["tactical_analysis"] = _build_tactical_analysis(
            runtime_state,
            legal_moves,
            move_scores,
            best_move=int(best_move),
        )
    labeled["value_target_teacher"] = float(teacher_value_target)
    labeled["value_target_outcome"] = float(outcome_value_target)
    labeled["value_target_mix"] = float(mixed_value_target)
    labeled["value_target_mix_teacher_weight"] = float(teacher_mix)
    labeled["value_target_mix_outcome_weight"] = float(outcome_mix)
    labeled["value_target"] = float(mixed_value_target)
    labeled["hard_example_score"] = float(hard_example_score)
    labeled["hard_example_margin"] = float(normalized_margin)
    labeled["hard_example_margin_hardness"] = float(margin_hardness)
    labeled["hard_example_outcome_hardness"] = float(max(0.0, -float(outcome_value_target)))
    labeled["hard_example_weight"] = float(hard_example_weight)
    return labeled


def _extract_policy_target_full(sample: dict[str, Any]) -> np.ndarray:
    distribution = dict(sample.get("policy_target", {}).get("distribution", {}))
    target = np.zeros(7, dtype=np.float32)
    for move in range(1, 8):
        target[move - 1] = float(distribution.get(str(move), 0.0))
    total = float(np.sum(target))
    if total > 0:
        target /= total
    return target


def _extract_tactical_move_mask(sample: dict[str, Any], *, mode: str) -> np.ndarray:
    mask = np.zeros(7, dtype=np.float32)
    tactical = sample.get("tactical_analysis", {})
    per_move = tactical.get("per_move", {}) if isinstance(tactical, dict) else {}
    legal_moves = set(int(move) for move in sample.get("legal_moves", []))
    for move in range(1, 8):
        move_payload = per_move.get(str(move), {})
        if move not in legal_moves:
            continue
        if mode == "capture" and bool(move_payload.get("has_immediate_capture", False)):
            mask[move - 1] = 1.0
        elif mode == "risky" and bool(move_payload.get("exposes_to_immediate_capture", False)):
            mask[move - 1] = 1.0
        elif mode == "safe" and not bool(move_payload.get("exposes_to_immediate_capture", False)):
            mask[move - 1] = 1.0
    return mask


def _encode_features(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]:
    features, legal_mask = encode_model_features(
        sample["state"],
        sample["legal_moves"],
        tactical_analysis=sample.get("tactical_analysis"),
    )
    policy_index = int(sample["policy_target"]["best_move"]) - 1
    policy_target_full = _extract_policy_target_full(sample)
    value = float(sample["value_target"])
    hard_example_weight = _safe_float(sample.get("hard_example_weight", 1.0), 1.0)
    if (not np.isfinite(hard_example_weight)) or hard_example_weight <= 0.0:
        hard_example_weight = 1.0
    capture_move_mask = _extract_tactical_move_mask(sample, mode="capture")
    safe_move_mask = _extract_tactical_move_mask(sample, mode="safe")
    risky_move_mask = _extract_tactical_move_mask(sample, mode="risky")
    return (
        features,
        legal_mask,
        policy_index,
        policy_target_full,
        value,
        capture_move_mask,
        safe_move_mask,
        risky_move_mask,
        float(hard_example_weight),
    )


def _label_samples_from_file(
    sampled_file_path: str,
    *,
    teacher_engine: str,
    teacher_level: str,
    include_tactical_analysis: bool = True,
    value_target_mix_teacher_weight: float = 1.0,
    hard_examples_enabled: bool = True,
    hard_examples_margin_threshold: float = 0.08,
    hard_examples_outcome_focus: float = 0.35,
    hard_examples_weight_multiplier: float = 2.0,
) -> tuple[int, list[dict[str, Any]], int, int]:
    source_samples = list(
        _iter_jsonl(
            Path(sampled_file_path),
            open_retries=8,
            retry_delay_seconds=0.5,
        )
    )
    source_count = len(source_samples)
    labeled_samples: list[dict[str, Any]] = []
    skipped_terminal = 0
    skipped_no_legal = 0
    for sample in source_samples:
        if bool(sample["state"].get("is_terminal", False)):
            skipped_terminal += 1
            continue
        if not sample["legal_moves"]:
            skipped_no_legal += 1
            continue
        labeled_samples.append(
            _label_sample(
                sample,
                teacher_engine=teacher_engine,
                teacher_level=teacher_level,
                include_tactical_analysis=include_tactical_analysis,
                value_target_mix_teacher_weight=value_target_mix_teacher_weight,
                hard_examples_enabled=hard_examples_enabled,
                hard_examples_margin_threshold=hard_examples_margin_threshold,
                hard_examples_outcome_focus=hard_examples_outcome_focus,
                hard_examples_weight_multiplier=hard_examples_weight_multiplier,
            )
        )
    return source_count, labeled_samples, skipped_terminal, skipped_no_legal


def _prepare_prelabeled_sample(
    sample: dict[str, Any],
    *,
    include_tactical_analysis: bool = True,
    hard_examples_enabled: bool = True,
    hard_examples_outcome_focus: float = 0.35,
    hard_examples_weight_multiplier: float = 2.0,
) -> dict[str, Any]:
    legal_moves = [int(move) for move in list(sample.get("legal_moves", []))]
    if bool(sample.get("state", {}).get("is_terminal", False)):
        raise ValueError(f"Cannot use terminal sample in source_prelabeled mode: {sample.get('sample_id', '<unknown>')}")
    if not legal_moves:
        raise ValueError(f"Cannot use sample without legal moves in source_prelabeled mode: {sample.get('sample_id', '<unknown>')}")

    policy_target = sample.get("policy_target", {})
    if not isinstance(policy_target, dict):
        raise ValueError(f"Missing policy_target in source_prelabeled sample: {sample.get('sample_id', '<unknown>')}")
    distribution = _normalize_distribution_for_legal_moves(legal_moves, policy_target.get("distribution", {}))
    if not distribution:
        raise ValueError(f"Invalid policy_target distribution in source_prelabeled sample: {sample.get('sample_id', '<unknown>')}")
    best_move = _safe_int(policy_target.get("best_move", 0), 0)
    if best_move not in legal_moves:
        best_move = int(max(distribution.items(), key=lambda item: item[1])[0])

    if "value_target" not in sample:
        raise ValueError(f"Missing value_target in source_prelabeled sample: {sample.get('sample_id', '<unknown>')}")
    value_target = float(sample.get("value_target", 0.0))
    if not np.isfinite(value_target):
        raise ValueError(f"Invalid value_target in source_prelabeled sample: {sample.get('sample_id', '<unknown>')}")
    value_target = float(max(-1.0, min(1.0, value_target)))

    prepared = dict(sample)
    prepared["policy_target"] = {
        "best_move": int(best_move),
        "distribution": {str(move): float(prob) for move, prob in distribution.items()},
    }
    prepared["value_target"] = value_target
    if "hard_example_weight" in prepared:
        existing_weight = _safe_float(prepared.get("hard_example_weight"), 1.0)
        if np.isfinite(existing_weight) and existing_weight > 0.0:
            prepared["hard_example_weight"] = float(existing_weight)
        else:
            prepared["hard_example_weight"] = 1.0
        prepared["hard_example_score"] = float(max(0.0, min(1.0, _safe_float(prepared.get("hard_example_score", 0.0), 0.0))))
    else:
        distribution_probs = np.asarray(
            [float(distribution.get(int(move), 0.0)) for move in legal_moves],
            dtype=np.float64,
        )
        entropy = 0.0
        if distribution_probs.size > 0:
            probs = np.clip(distribution_probs, 1e-9, 1.0)
            probs = probs / float(np.sum(probs))
            entropy = float(-np.sum(probs * np.log(probs)))
            entropy = entropy / float(np.log(max(2, int(len(probs)))))
        outcome_value = _safe_sample_outcome_value(prepared)
        outcome_hardness = max(0.0, -float(outcome_value))
        outcome_focus = max(0.0, min(1.0, float(hard_examples_outcome_focus)))
        hard_score = max(0.0, min(1.0, ((1.0 - outcome_focus) * entropy) + (outcome_focus * outcome_hardness)))
        if not bool(hard_examples_enabled):
            hard_score = 0.0
        multiplier = max(1.0, float(hard_examples_weight_multiplier))
        prepared["hard_example_score"] = float(hard_score)
        prepared["hard_example_weight"] = float(1.0 + (hard_score * (multiplier - 1.0)))

    if include_tactical_analysis and not isinstance(prepared.get("tactical_analysis"), dict):
        runtime_state = _raw_state_to_runtime_state(prepared["state"])
        move_scores = {int(move): float(distribution.get(int(move), 0.0)) for move in legal_moves}
        prepared["tactical_analysis"] = _build_tactical_analysis(
            runtime_state,
            legal_moves,
            move_scores,
            best_move=int(best_move),
        )
    return prepared


def _prepare_prelabeled_samples_from_file(
    sampled_file_path: str,
    *,
    include_tactical_analysis: bool = True,
    hard_examples_enabled: bool = True,
    hard_examples_outcome_focus: float = 0.35,
    hard_examples_weight_multiplier: float = 2.0,
) -> tuple[int, list[dict[str, Any]], int, int, int]:
    source_samples = list(
        _iter_jsonl(
            Path(sampled_file_path),
            open_retries=8,
            retry_delay_seconds=0.5,
        )
    )
    source_count = len(source_samples)
    prepared_samples: list[dict[str, Any]] = []
    skipped_terminal = 0
    skipped_no_legal = 0
    skipped_invalid = 0
    for sample in source_samples:
        if bool(sample.get("state", {}).get("is_terminal", False)):
            skipped_terminal += 1
            continue
        if not list(sample.get("legal_moves", [])):
            skipped_no_legal += 1
            continue
        try:
            prepared_samples.append(
                _prepare_prelabeled_sample(
                    sample,
                    include_tactical_analysis=include_tactical_analysis,
                    hard_examples_enabled=hard_examples_enabled,
                    hard_examples_outcome_focus=hard_examples_outcome_focus,
                    hard_examples_weight_multiplier=hard_examples_weight_multiplier,
                )
            )
        except Exception:
            skipped_invalid += 1
            continue
    return source_count, prepared_samples, skipped_terminal, skipped_no_legal, skipped_invalid


def _detect_source_samples_are_prelabeled(sampled_file_path: Path, *, max_samples: int = 32) -> bool:
    try:
        for idx, sample in enumerate(_iter_jsonl(sampled_file_path)):
            if idx >= int(max_samples):
                break
            if bool(sample.get("state", {}).get("is_terminal", False)):
                continue
            if not list(sample.get("legal_moves", [])):
                continue
            policy_target = sample.get("policy_target", {})
            distribution = policy_target.get("distribution", {}) if isinstance(policy_target, dict) else {}
            if "value_target" in sample and isinstance(distribution, dict) and bool(distribution):
                return True
    except Exception:
        return False
    return False


def _play_and_sample_game(
    agent_a,
    agent_b,
    *,
    matchup_id: str,
    game_id: str,
    seed: int,
    starter: int,
    sample_every_n_plies: int,
    sample_ply_offset: int = 0,
    max_moves: int = 300,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = songo_ai_game.create_state()
    agents = [agent_a, agent_b] if starter == 0 else [agent_b, agent_a]
    moves: list[int] = []
    samples: list[dict[str, Any]] = []
    ply = 0
    sample_index = 0
    started_at = utc_now_iso()
    end_reason = "finished"
    sample_stride = max(1, int(sample_every_n_plies))
    sample_offset = int(sample_ply_offset) % sample_stride

    while not songo_ai_game.is_terminal(state) and ply < max_moves:
        if ((int(ply) - int(sample_offset)) % int(sample_stride)) == 0:
            sample_index += 1
            samples.append(
                _sample_position(
                    game_id=game_id,
                    matchup_id=matchup_id,
                    sample_index=sample_index,
                    ply=ply,
                    seed=seed,
                    state=state,
                )
            )

        legal = songo_ai_game.legal_moves(state)
        if not legal:
            end_reason = "no_legal_moves_available"
            break

        current = songo_ai_game.current_player(state)
        try:
            move, _info = agents[current].choose(songo_ai_game.clone_state(state))
        except Exception as exc:
            raise RuntimeError(
                "Benchmatch agent choose failed in strict mode | "
                f"matchup_id={matchup_id} | game_id={game_id} | ply={ply} | "
                f"player={current} | legal={legal} | cause={type(exc).__name__}: {exc}"
            ) from exc
        if move not in legal:
            raise ValueError(
                "Benchmatch agent returned illegal move in strict mode | "
                f"matchup_id={matchup_id} | game_id={game_id} | ply={ply} | "
                f"player={current} | move={move} | legal={legal}"
            )
        moves.append(int(move))
        state = songo_ai_game.simulate_move(state, int(move))
        ply += 1

    if sample_every_n_plies > 0 and (not samples or samples[-1]["ply"] != ply):
        sample_index += 1
        samples.append(
            _sample_position(
                game_id=game_id,
                matchup_id=matchup_id,
                sample_index=sample_index,
                ply=ply,
                seed=seed,
                state=state,
            )
        )

    winner = songo_ai_game.winner(state)
    if starter == 1 and winner is not None:
        logical_winner = "player_a" if winner == 1 else "player_b"
    elif winner is not None:
        logical_winner = "player_a" if winner == 0 else "player_b"
    else:
        logical_winner = None

    raw_log = {
        "game_id": game_id,
        "matchup_id": matchup_id,
        "seed": seed,
        "starter": starter,
        "sample_every_n_plies": int(sample_stride),
        "sample_ply_offset": int(sample_offset),
        "player_a": agent_a.display_name,
        "player_b": agent_b.display_name,
        "winner": logical_winner,
        "winner_index": (None if winner is None else int(winner)),
        "moves": moves,
        "ply_count": ply,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "scores": list(songo_ai_game.scores(state)),
        "reason": "finished" if songo_ai_game.is_terminal(state) else (end_reason if end_reason != "finished" else f"max_moves_reached:{max_moves}"),
    }
    return raw_log, samples


def _play_and_sample_game_from_specs(
    matchup_a: str,
    matchup_b: str,
    *,
    matchup_id: str,
    game_id: str,
    seed: int,
    starter: int,
    sample_every_n_plies: int,
    sample_ply_offset: int = 0,
    max_moves: int = 300,
    models_root: str = ".",
    model_agent_device: str = "cpu",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved_models_root = Path(models_root)
    agent_a = _build_external_agent(matchup_a, models_root=resolved_models_root, device=model_agent_device)
    agent_b = _build_external_agent(matchup_b, models_root=resolved_models_root, device=model_agent_device)
    return _play_and_sample_game(
        agent_a,
        agent_b,
        matchup_id=matchup_id,
        game_id=game_id,
        seed=seed,
        starter=starter,
        sample_every_n_plies=sample_every_n_plies,
        sample_ply_offset=sample_ply_offset,
        max_moves=max_moves,
    )


def _materialize_completed_game(
    job: JobContext,
    *,
    pending: dict[str, Any],
    raw_payload: dict[str, Any],
    samples: list[dict[str, Any]],
    matchup_id: str,
    games: int,
    execution_mode: str,
    max_samples_to_write: int | None = None,
) -> tuple[int, int]:
    if max_samples_to_write is not None:
        capped = max(0, int(max_samples_to_write))
        samples = samples[:capped]
    sample_count = len(samples)
    if sample_count <= 0:
        job.logger.info(
            "dataset game skipped at materialization | matchup=%s | game=%s/%s | reason=no_admitted_samples | mode=%s",
            matchup_id,
            pending["game_no"],
            games,
            execution_mode,
        )
        return 0, 0

    winner_index_raw = raw_payload.get("winner_index")
    winner_index: int | None
    if winner_index_raw is None:
        winner_index = None
    else:
        try:
            winner_index = int(winner_index_raw)
        except Exception:
            winner_index = None
    scores_raw = raw_payload.get("scores", [0, 0])
    try:
        final_scores = (
            int(scores_raw[0]) if isinstance(scores_raw, (list, tuple)) and len(scores_raw) > 0 else 0,
            int(scores_raw[1]) if isinstance(scores_raw, (list, tuple)) and len(scores_raw) > 1 else 0,
        )
    except Exception:
        final_scores = (0, 0)

    # Persist game-level outcome context on each sampled position so labelers can
    # mix local teacher value with long-horizon game outcome when requested.
    enriched_samples: list[dict[str, Any]] = []
    for sample in samples:
        payload = dict(sample)
        player_label = str(payload.get("player_to_move", "south"))
        payload["game_winner_index"] = (None if winner_index is None else int(winner_index))
        payload["game_final_scores"] = {
            "south": int(final_scores[0]),
            "north": int(final_scores[1]),
        }
        payload["game_outcome_for_player_to_move"] = float(
            _outcome_value_for_player_to_move(
                winner_index=winner_index,
                final_scores=final_scores,
                player_to_move_label=player_label,
            )
        )
        enriched_samples.append(payload)

    _write_json(Path(pending["raw_game_path"]), raw_payload)
    sampled_game_path = Path(pending["sampled_game_path"])
    write_jsonl_atomic(sampled_game_path, enriched_samples, ensure_ascii=True)

    job.logger.info(
        "dataset game completed | matchup=%s | game=%s/%s | moves=%s | samples=%s | winner=%s | mode=%s",
        matchup_id,
        pending["game_no"],
        games,
        raw_payload["ply_count"],
        sample_count,
        raw_payload["winner"],
        execution_mode,
    )
    job.write_event(
        "dataset_game_completed",
        matchup=matchup_id,
        game_id=pending["game_id"],
        game_index=pending["game_no"],
        samples=sample_count,
        winner=raw_payload["winner"],
        execution_mode=execution_mode,
    )
    job.write_metric(
        {
            "metric_type": "dataset_game_completed",
            "matchup_id": matchup_id,
            "game_id": pending["game_id"],
            "samples": sample_count,
            "moves": raw_payload["ply_count"],
        }
    )
    return 1, sample_count


def _run_pending_games_sequential(
    job: JobContext,
    *,
    pending_games: list[dict[str, Any]],
    matchup_a: str,
    matchup_b: str,
    matchup_id: str,
    games: int,
    sample_every_n_plies: int,
    max_moves: int,
    models_root: Path,
    model_agent_device: str,
    on_game_completed: Callable[[dict[str, Any], int, int], None] | None = None,
    sample_cap_resolver: Callable[[dict[str, Any], int], int] | None = None,
    on_game_failed: Callable[[dict[str, Any], Exception], None] | None = None,
    on_materialization_error: Callable[[dict[str, Any], int, Exception], None] | None = None,
) -> tuple[int, int, str]:
    agent_a = _build_external_agent(matchup_a, models_root=models_root, device=model_agent_device)
    agent_b = _build_external_agent(matchup_b, models_root=models_root, device=model_agent_device)
    completed_games = 0
    completed_samples = 0
    for pending in pending_games:
        job.logger.info(
            "dataset game running | matchup=%s | game=%s/%s | seed=%s | starter=%s | sample_offset=%s | mode=sequential",
            matchup_id,
            pending["game_no"],
            games,
            pending["seed"],
            pending["starter"],
            pending.get("sample_ply_offset", 0),
        )
        try:
            raw_payload, samples = _play_and_sample_game(
                agent_a,
                agent_b,
                matchup_id=matchup_id,
                game_id=str(pending["game_id"]),
                seed=int(pending["seed"]),
                starter=int(pending["starter"]),
                sample_every_n_plies=sample_every_n_plies,
                sample_ply_offset=int(pending.get("sample_ply_offset", 0)),
                max_moves=max_moves,
            )
        except Exception as exc:
            if on_game_failed is not None:
                on_game_failed(pending, exc)
                continue
            raise
        max_samples_to_write = len(samples)
        if sample_cap_resolver is not None:
            max_samples_to_write = max(0, int(sample_cap_resolver(pending, len(samples))))
        try:
            games_inc, samples_inc = _materialize_completed_game(
                job,
                pending=pending,
                raw_payload=raw_payload,
                samples=samples,
                matchup_id=matchup_id,
                games=games,
                execution_mode="sequential",
                max_samples_to_write=max_samples_to_write,
            )
        except Exception as exc:
            if on_materialization_error is not None:
                on_materialization_error(pending, int(max_samples_to_write), exc)
            if on_game_failed is not None:
                on_game_failed(pending, exc)
                continue
            raise
        completed_games += games_inc
        completed_samples += samples_inc
        if on_game_completed is not None and games_inc > 0:
            on_game_completed(pending, completed_games, completed_samples)
    return completed_games, completed_samples, "sequential"


def _run_dataset_generation_self_play_puct(
    job: JobContext,
    *,
    cfg: dict[str, Any],
    source_mode: str,
    dataset_source_id: str,
    source_dataset_id: str,
    source_dataset_ids: list[str],
    derivation_strategy: str,
    derivation_params: dict[str, Any],
    raw_dir: Path,
    sampled_dir: Path,
    target_samples: int,
    games: int,
    sample_every_n_plies: int,
    max_moves: int,
    num_workers: int,
    max_pending_futures: int,
    multiprocessing_start_method: str,
    effective_max_tasks_per_child: int | None,
    cycle_matchups_until_target: bool,
    max_matchup_cycles: int,
    base_seed: int,
) -> dict[str, Any]:
    model_spec = str(cfg.get("self_play_model", "auto_best")).strip() or "auto_best"
    model_device = str(cfg.get("self_play_model_device", cfg.get("model_agent_device", "cpu"))).strip().lower() or "cpu"
    search_simulations = max(1, int(cfg.get("self_play_num_simulations", 64)))
    search_c_puct = float(cfg.get("self_play_c_puct", 1.5))
    search_temperature = max(1e-6, float(cfg.get("self_play_temperature", 1.0)))
    search_temperature_end = max(1e-6, float(cfg.get("self_play_temperature_end", 0.1)))
    search_temperature_drop_ply = max(0, int(cfg.get("self_play_temperature_drop_ply", 12)))
    search_root_dirichlet_alpha = max(1e-6, float(cfg.get("self_play_root_dirichlet_alpha", 0.3)))
    search_root_dirichlet_epsilon = max(0.0, min(1.0, float(cfg.get("self_play_root_dirichlet_epsilon", 0.25))))
    search_deterministic = _as_bool(cfg.get("self_play_deterministic", False), default=False)
    progress_update_every_n_games = max(1, int(cfg.get("progress_update_every_n_games", 10)))

    model_id, checkpoint_path = _resolve_model_checkpoint_for_generation(model_spec, models_root=job.paths.models_root)
    model_matchup = f"model:{model_id} vs model:{model_id}"
    matchup_id = _slugify_matchup(f"self_play_{model_id}")

    state = job.read_state()
    summaries: list[dict[str, Any]] = []
    initial_total_samples = _count_total_jsonl_lines(sampled_dir)
    initial_total_games = _count_jsonl_files(sampled_dir)
    total_samples_after_run = initial_total_samples
    last_completed_game_id = str(state.get("last_completed_game_id", "")).strip()

    job.logger.info(
        "dataset generation self-play startup | dataset_source_id=%s | model_spec=%s | resolved_model_id=%s | checkpoint=%s | target_samples=%s | games_per_cycle=%s | simulations=%s | c_puct=%.3f | temp_start=%.3f | temp_end=%.3f | temp_drop_ply=%s | dir_alpha=%.3f | dir_eps=%.3f | deterministic=%s | workers=%s | max_pending_futures=%s",
        dataset_source_id,
        model_spec,
        model_id,
        checkpoint_path,
        target_samples,
        games,
        search_simulations,
        search_c_puct,
        search_temperature,
        search_temperature_end,
        search_temperature_drop_ply,
        search_root_dirichlet_alpha,
        search_root_dirichlet_epsilon,
        search_deterministic,
        num_workers,
        max_pending_futures,
    )
    if _as_bool(cfg.get("global_target_enabled", False), default=False):
        job.logger.info(
            "dataset generation self-play global target integration disabled for this mode | source_mode=%s",
            source_mode,
        )
    job.write_event(
        "dataset_generation_self_play_started",
        dataset_source_id=dataset_source_id,
        model_spec=model_spec,
        model_id=model_id,
        checkpoint_path=str(checkpoint_path),
        target_samples=target_samples,
        games_per_cycle=games,
        num_simulations=search_simulations,
        c_puct=search_c_puct,
        temperature=search_temperature,
        temperature_end=search_temperature_end,
        temperature_drop_ply=search_temperature_drop_ply,
        root_dirichlet_alpha=search_root_dirichlet_alpha,
        root_dirichlet_epsilon=search_root_dirichlet_epsilon,
        deterministic=search_deterministic,
    )
    job.set_phase("dataset_generation")
    job.write_metric(
        {
            "metric_type": "dataset_generation_started",
            "source_mode": source_mode,
            "dataset_source_id": dataset_source_id,
            "target_samples": target_samples,
        }
    )

    if target_samples > 0 and initial_total_samples >= target_samples:
        _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[model_matchup],
            source_dataset_id=source_dataset_id,
            source_dataset_ids=source_dataset_ids,
            derivation_strategy=derivation_strategy,
            derivation_params=derivation_params,
            source_status="completed",
            raw_files_override=_count_json_files(raw_dir),
            sampled_files_override=_count_jsonl_files(sampled_dir),
            sampled_positions_override=initial_total_samples,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "matchups": [],
            "cycles_completed": 0,
            "source_status": "completed",
            "existing_samples": initial_total_samples,
            "added_games": 0,
            "failed_games": 0,
            "added_samples": 0,
            "total_samples": initial_total_samples,
            "target_samples": target_samples,
            "self_play": {
                "model_id": model_id,
                "checkpoint_path": str(checkpoint_path),
                "num_simulations": search_simulations,
                "c_puct": search_c_puct,
                "temperature": search_temperature,
                "temperature_end": search_temperature_end,
                "temperature_drop_ply": search_temperature_drop_ply,
                "root_dirichlet_alpha": search_root_dirichlet_alpha,
                "root_dirichlet_epsilon": search_root_dirichlet_epsilon,
                "deterministic": search_deterministic,
            },
        }
        _write_json(job.job_dir / "dataset_generation" / "dataset_generation_summary.json", summary)
        return summary

    cycle_index = 0
    while True:
        cycle_index += 1
        job.set_phase(f"dataset_generation:{matchup_id}")
        cycle_samples_before = int(total_samples_after_run)
        pending_games = _build_pending_incremental_games(
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            matchup_id=matchup_id,
            games_to_add=games,
            seed_base=base_seed + (cycle_index * 1_000_000),
            sample_every_n_plies=sample_every_n_plies,
            completion_mode=completed_game_detection_mode,
        )
        if not pending_games:
            job.logger.warning(
                "dataset generation self-play found no pending games | cycle=%s | matchup_id=%s",
                cycle_index,
                matchup_id,
            )
            break

        matchup_game_count = 0
        matchup_sample_count = 0
        matchup_failed_game_count = 0

        def _remaining_sample_budget() -> int:
            if target_samples <= 0:
                return 1_000_000_000
            return max(0, int(target_samples) - int(total_samples_after_run))

        def _update_runtime_progress(*, completed_game_id: str) -> None:
            nonlocal last_completed_game_id
            last_completed_game_id = str(completed_game_id).strip() or last_completed_game_id
            job.write_state(
                {
                    "current_matchup": matchup_id,
                    "completed_games": matchup_game_count,
                    "remaining_games": max(0, games - matchup_game_count),
                    "last_completed_game_id": last_completed_game_id,
                    "sample_count": int(total_samples_after_run),
                    "target_samples": int(target_samples),
                    "source_mode": source_mode,
                }
            )
            _register_dataset_source(
                job,
                dataset_source_id=dataset_source_id,
                source_mode=source_mode,
                raw_dir=raw_dir,
                sampled_dir=sampled_dir,
                target_samples=target_samples,
                games_per_matchup=games,
                sample_every_n_plies=sample_every_n_plies,
                matchups=[model_matchup],
                source_dataset_id=source_dataset_id,
                source_dataset_ids=source_dataset_ids,
                derivation_strategy=derivation_strategy,
                derivation_params=derivation_params,
                source_status="partial",
                partial_summary={
                    "cycles_completed": cycle_index,
                    "added_games": int(max(0, _count_jsonl_files(sampled_dir) - initial_total_games)),
                    "added_samples": int(max(0, int(total_samples_after_run) - int(initial_total_samples))),
                    "total_samples": int(total_samples_after_run),
                    "target_samples": int(target_samples),
                },
                raw_files_override=_count_json_files(raw_dir),
                sampled_files_override=_count_jsonl_files(sampled_dir),
                sampled_positions_override=int(total_samples_after_run),
            )
            partial_summary = {
                "job_id": job.job_id,
                "dataset_source_id": dataset_source_id,
                "source_mode": source_mode,
                "matchups": summaries,
                "cycles_completed": cycle_index,
                "source_status": "partial",
                "existing_samples": initial_total_samples,
                "added_games": int(max(0, _count_jsonl_files(sampled_dir) - initial_total_games)),
                "failed_games": int(sum(int(item.get("games_failed", 0)) for item in summaries)),
                "added_samples": int(max(0, int(total_samples_after_run) - int(initial_total_samples))),
                "total_samples": int(total_samples_after_run),
                "target_samples": int(target_samples),
                "self_play": {
                    "model_spec": model_spec,
                    "model_id": model_id,
                    "checkpoint_path": str(checkpoint_path),
                    "model_device": model_device,
                    "num_simulations": search_simulations,
                    "c_puct": search_c_puct,
                },
            }
            _write_json(job.job_dir / "dataset_generation" / "dataset_generation_summary.json", partial_summary)

        def _run_single_pending_game(pending: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            return _play_and_sample_self_play_game_from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                model_id=str(model_id),
                model_device=model_device,
                matchup_id=matchup_id,
                game_id=str(pending["game_id"]),
                seed=int(pending["seed"]),
                sample_every_n_plies=sample_every_n_plies,
                sample_ply_offset=int(pending.get("sample_ply_offset", 0)),
                max_moves=max_moves,
                num_simulations=search_simulations,
                c_puct=search_c_puct,
                temperature=search_temperature,
                temperature_end=search_temperature_end,
                temperature_drop_ply=search_temperature_drop_ply,
                root_dirichlet_alpha=search_root_dirichlet_alpha,
                root_dirichlet_epsilon=search_root_dirichlet_epsilon,
                deterministic=search_deterministic,
            )

        if num_workers <= 1:
            for pending in pending_games:
                remaining_before_game = _remaining_sample_budget()
                if remaining_before_game <= 0:
                    break
                try:
                    raw_payload, samples = _run_single_pending_game(pending)
                except Exception as exc:
                    matchup_failed_game_count += 1
                    job.logger.warning(
                        "dataset self-play game failed | matchup=%s | game=%s/%s | error=%s: %s | mode=sequential",
                        matchup_id,
                        pending["game_no"],
                        games,
                        type(exc).__name__,
                        exc,
                    )
                    job.write_event(
                        "dataset_game_failed",
                        matchup=matchup_id,
                        game_id=pending["game_id"],
                        game_index=pending["game_no"],
                        execution_mode="self_play_sequential",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue
                admitted_samples = min(len(samples), max(0, int(remaining_before_game)))
                games_inc, samples_inc = _materialize_completed_game(
                    job,
                    pending=pending,
                    raw_payload=raw_payload,
                    samples=samples,
                    matchup_id=matchup_id,
                    games=games,
                    execution_mode="self_play_sequential",
                    max_samples_to_write=admitted_samples,
                )
                matchup_game_count += games_inc
                matchup_sample_count += samples_inc
                total_samples_after_run += samples_inc
                if games_inc > 0 and (matchup_game_count % progress_update_every_n_games == 0):
                    _update_runtime_progress(completed_game_id=str(pending["game_id"]))
                if _remaining_sample_budget() <= 0:
                    break
        else:
            future_map: dict[concurrent.futures.Future, dict[str, Any]] = {}
            pending_queue = list(pending_games)
            try:
                mp_context = multiprocessing.get_context(multiprocessing_start_method)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=mp_context,
                    max_tasks_per_child=effective_max_tasks_per_child,
                ) as executor:
                    while pending_queue or future_map:
                        while pending_queue and len(future_map) < max_pending_futures and _remaining_sample_budget() > 0:
                            pending = pending_queue.pop(0)
                            future = executor.submit(
                                _play_and_sample_self_play_game_from_checkpoint,
                                checkpoint_path=str(checkpoint_path),
                                model_id=str(model_id),
                                model_device=model_device,
                                matchup_id=matchup_id,
                                game_id=str(pending["game_id"]),
                                seed=int(pending["seed"]),
                                sample_every_n_plies=sample_every_n_plies,
                                sample_ply_offset=int(pending.get("sample_ply_offset", 0)),
                                max_moves=max_moves,
                                num_simulations=search_simulations,
                                c_puct=search_c_puct,
                                temperature=search_temperature,
                                temperature_end=search_temperature_end,
                                temperature_drop_ply=search_temperature_drop_ply,
                                root_dirichlet_alpha=search_root_dirichlet_alpha,
                                root_dirichlet_epsilon=search_root_dirichlet_epsilon,
                                deterministic=search_deterministic,
                            )
                            future_map[future] = pending
                        if not future_map:
                            break
                        done, _ = concurrent.futures.wait(
                            future_map.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            pending = future_map.pop(future)
                            try:
                                raw_payload, samples = future.result()
                            except Exception as exc:
                                matchup_failed_game_count += 1
                                job.logger.warning(
                                    "dataset self-play game failed | matchup=%s | game=%s/%s | error=%s: %s | mode=parallel",
                                    matchup_id,
                                    pending["game_no"],
                                    games,
                                    type(exc).__name__,
                                    exc,
                                )
                                job.write_event(
                                    "dataset_game_failed",
                                    matchup=matchup_id,
                                    game_id=pending["game_id"],
                                    game_index=pending["game_no"],
                                    execution_mode="self_play_parallel",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                                continue
                            admitted_samples = min(len(samples), max(0, int(_remaining_sample_budget())))
                            games_inc, samples_inc = _materialize_completed_game(
                                job,
                                pending=pending,
                                raw_payload=raw_payload,
                                samples=samples,
                                matchup_id=matchup_id,
                                games=games,
                                execution_mode="self_play_parallel",
                                max_samples_to_write=admitted_samples,
                            )
                            matchup_game_count += games_inc
                            matchup_sample_count += samples_inc
                            total_samples_after_run += samples_inc
                            if games_inc > 0 and (matchup_game_count % progress_update_every_n_games == 0):
                                _update_runtime_progress(completed_game_id=str(pending["game_id"]))
                            if _remaining_sample_budget() <= 0:
                                pending_queue.clear()
            except concurrent.futures.process.BrokenProcessPool as exc:
                failed_pending = list(future_map.values()) + list(pending_queue)
                job.logger.warning(
                    "dataset self-play parallel pool broken | cycle=%s | completed_games=%s | remaining_fallback=%s | error=%s",
                    cycle_index,
                    matchup_game_count,
                    len(failed_pending),
                    exc,
                )
                job.write_event(
                    "dataset_parallel_execution_broken",
                    matchup=matchup_id,
                    completed_games=matchup_game_count,
                    remaining_fallback=len(failed_pending),
                    error=str(exc),
                    execution_mode="self_play_parallel",
                )
                for pending in failed_pending:
                    if _remaining_sample_budget() <= 0:
                        break
                    try:
                        raw_payload, samples = _run_single_pending_game(pending)
                    except Exception as inner_exc:
                        matchup_failed_game_count += 1
                        job.logger.warning(
                            "dataset self-play game failed | matchup=%s | game=%s/%s | error=%s: %s | mode=sequential_fallback",
                            matchup_id,
                            pending["game_no"],
                            games,
                            type(inner_exc).__name__,
                            inner_exc,
                        )
                        continue
                    admitted_samples = min(len(samples), max(0, int(_remaining_sample_budget())))
                    games_inc, samples_inc = _materialize_completed_game(
                        job,
                        pending=pending,
                        raw_payload=raw_payload,
                        samples=samples,
                        matchup_id=matchup_id,
                        games=games,
                        execution_mode="self_play_sequential_fallback",
                        max_samples_to_write=admitted_samples,
                    )
                    matchup_game_count += games_inc
                    matchup_sample_count += samples_inc
                    total_samples_after_run += samples_inc
                    if games_inc > 0 and (matchup_game_count % progress_update_every_n_games == 0):
                        _update_runtime_progress(completed_game_id=str(pending["game_id"]))

        if last_completed_game_id:
            _update_runtime_progress(completed_game_id=last_completed_game_id)

        summary_payload = {
            "matchup_id": matchup_id,
            "cycle_index": cycle_index,
            "player_a": f"model:{model_id}",
            "player_b": f"model:{model_id}",
            "existing_games": int(_count_jsonl_files(sampled_dir) - matchup_game_count),
            "games_added": matchup_game_count,
            "games_failed": matchup_failed_game_count,
            "samples_added": matchup_sample_count,
            "sample_every_n_plies": sample_every_n_plies,
            "num_workers": num_workers,
            "self_play_model_id": model_id,
            "self_play_num_simulations": search_simulations,
        }
        _accumulate_matchup_summary(summaries, summary_payload)
        job.write_metric({"metric_type": "dataset_matchup_completed", **summary_payload})
        job.write_event(
            "dataset_matchup_completed",
            matchup=matchup_id,
            samples=matchup_sample_count,
            cycle_index=cycle_index,
            execution_mode="self_play_puct",
        )

        cycle_added_samples = int(total_samples_after_run) - int(cycle_samples_before)
        if target_samples > 0 and int(total_samples_after_run) >= int(target_samples):
            break
        if not bool(cycle_matchups_until_target):
            break
        if int(max_matchup_cycles) > 0 and int(cycle_index) >= int(max_matchup_cycles):
            break
        if cycle_added_samples <= 0:
            job.logger.warning(
                "dataset generation self-play cycle produced no new samples, stopping | cycle=%s | total_samples=%s | target_samples=%s",
                cycle_index,
                total_samples_after_run,
                target_samples,
            )
            break

    final_total_samples = _count_total_jsonl_lines(sampled_dir)
    final_total_games = _count_jsonl_files(sampled_dir)
    added_games = max(0, int(final_total_games) - int(initial_total_games))
    failed_games = sum(int(item.get("games_failed", 0)) for item in summaries)
    added_samples = max(0, int(final_total_samples) - int(initial_total_samples))
    completed_cycles = max(0, int(len(summaries)))
    source_status = "completed" if (target_samples > 0 and int(final_total_samples) >= int(target_samples)) else "partial"
    summary = {
        "job_id": job.job_id,
        "dataset_source_id": dataset_source_id,
        "source_mode": source_mode,
        "matchups": summaries,
        "cycles_completed": completed_cycles,
        "source_status": source_status,
        "existing_samples": initial_total_samples,
        "added_games": added_games,
        "failed_games": failed_games,
        "added_samples": added_samples,
        "total_samples": final_total_samples,
        "target_samples": target_samples,
        "self_play": {
            "model_spec": model_spec,
            "model_id": model_id,
            "checkpoint_path": str(checkpoint_path),
            "model_device": model_device,
            "num_simulations": search_simulations,
            "c_puct": search_c_puct,
            "temperature": search_temperature,
            "temperature_end": search_temperature_end,
            "temperature_drop_ply": search_temperature_drop_ply,
            "root_dirichlet_alpha": search_root_dirichlet_alpha,
            "root_dirichlet_epsilon": search_root_dirichlet_epsilon,
            "deterministic": search_deterministic,
        },
    }
    _register_dataset_source(
        job,
        dataset_source_id=dataset_source_id,
        source_mode=source_mode,
        raw_dir=raw_dir,
        sampled_dir=sampled_dir,
        target_samples=target_samples,
        games_per_matchup=games,
        sample_every_n_plies=sample_every_n_plies,
        matchups=[model_matchup],
        source_dataset_id=source_dataset_id,
        source_dataset_ids=source_dataset_ids,
        derivation_strategy=derivation_strategy,
        derivation_params=derivation_params,
        source_status=source_status,
        partial_summary={} if source_status == "completed" else {
            "cycles_completed": completed_cycles,
            "added_games": added_games,
            "failed_games": failed_games,
            "added_samples": added_samples,
            "total_samples": final_total_samples,
            "target_samples": target_samples,
        },
        raw_files_override=_count_json_files(raw_dir),
        sampled_files_override=_count_jsonl_files(sampled_dir),
        sampled_positions_override=final_total_samples,
    )
    _write_json(job.job_dir / "dataset_generation" / "dataset_generation_summary.json", summary)
    job.write_metric(
        {
            "metric_type": "dataset_generation_completed",
            "source_mode": source_mode,
            "dataset_source_id": dataset_source_id,
            "source_status": source_status,
            "added_games": added_games,
            "added_samples": added_samples,
            "total_samples": final_total_samples,
            "target_samples": target_samples,
        }
    )
    job.write_event(
        "dataset_generation_completed",
        source_mode=source_mode,
        dataset_source_id=dataset_source_id,
        source_status=source_status,
        added_games=added_games,
        added_samples=added_samples,
        total_samples=final_total_samples,
        target_samples=target_samples,
    )
    job.write_state(
        {
            "current_matchup": None,
            "completed_games": added_games,
            "remaining_games": 0,
            "last_completed_game_id": last_completed_game_id,
            "sample_count": final_total_samples,
            "target_samples": target_samples,
            "source_mode": source_mode,
        }
    )
    return summary


def run_dataset_generation(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("dataset_generation", {})
    runtime_cfg = job.config.get("runtime", {})
    source_mode = str(cfg.get("source_mode", "benchmatch")).strip().lower() or "benchmatch"
    dataset_source_id = str(cfg.get("dataset_source_id", "")).strip() or Path(str(cfg.get("output_sampled_dir", "sampled_positions"))).name
    source_dataset_id = str(cfg.get("source_dataset_id", "")).strip()
    source_dataset_ids = [str(value).strip() for value in cfg.get("source_dataset_ids", []) if str(value).strip()]
    derivation_strategy = str(cfg.get("derivation_strategy", "unique_positions")).strip().lower() or "unique_positions"
    derivation_params = dict(cfg.get("derivation_params", {}))
    merge_dedupe_sample_ids = _as_bool(cfg.get("merge_dedupe_sample_ids", True), default=True)
    games = int(cfg.get("games", 20))
    matchups = list(cfg.get("matchups", []))
    cycle_matchups_until_target = _as_bool(
        cfg.get(
            "cycle_matchups_until_target",
            cfg.get("benchmatch_cycle_matchups_until_target", source_mode == "benchmatch"),
        ),
        default=(source_mode == "benchmatch"),
    )
    max_matchup_cycles = max(
        0,
        int(cfg.get("max_matchup_cycles", cfg.get("benchmatch_max_matchup_cycles", 0))),
    )
    sample_every_n_plies = int(cfg.get("sample_every_n_plies", 2))
    if sample_every_n_plies <= 0:
        raise ValueError("`sample_every_n_plies` doit etre strictement positif")
    base_seed = int(job.config.get("runtime", {}).get("seed", 42))
    max_moves = int(cfg.get("max_moves", 300))
    num_workers = max(1, int(runtime_cfg.get("num_workers", 1)))
    max_pending_futures = max(1, int(cfg.get("max_pending_futures", num_workers * 2)))
    multiprocessing_start_method = str(runtime_cfg.get("multiprocessing_start_method", "spawn")).strip().lower() or "spawn"
    max_tasks_per_child = int(runtime_cfg.get("max_tasks_per_child", 25))
    effective_max_tasks_per_child = _resolve_pool_max_tasks_per_child(
        start_method=multiprocessing_start_method,
        configured_value=max_tasks_per_child,
        logger=job.logger,
        scope="dataset generation pool configuration",
    )
    target_samples = int(cfg.get("target_samples", 0))
    completed_game_detection_mode = _normalize_completed_game_detection_mode(
        cfg.get("completed_game_detection_mode", "raw_and_sampled")
    )
    model_agent_device = str(cfg.get("model_agent_device", "cpu")).strip().lower() or "cpu"
    global_target_enabled = _as_bool(cfg.get("global_target_enabled", False), default=False)
    global_target_id = str(cfg.get("global_target_id", "")).strip() or dataset_source_id
    global_target_samples = int(cfg.get("global_target_samples", target_samples or 0))
    global_progress_path = _resolve_storage_path(
        job.paths.drive_root,
        cfg.get("global_progress_path"),
        job.paths.data_root / "global_generation_progress" / f"{global_target_id}.json",
    )
    global_progress_backend = _resolve_global_progress_backend_config(
        cfg=cfg,
        global_target_id=global_target_id,
        target_samples=global_target_samples,
    )
    global_progress_lock_dir = Path(str(global_progress_path) + ".lock")

    dataset_dir = job.job_dir / "dataset_generation"
    configured_output_raw_dir = cfg.get("output_raw_dir")
    configured_output_sampled_dir = cfg.get("output_sampled_dir")
    raw_dir = _resolve_storage_path(job.paths.drive_root, configured_output_raw_dir, dataset_dir / "raw_match_logs")
    sampled_dir = _resolve_storage_path(job.paths.drive_root, configured_output_sampled_dir, dataset_dir / "sampled_positions")

    if source_mode != "benchmatch":
        sampled_dir = _resolve_storage_path(
            job.paths.drive_root,
            None,
            job.paths.data_root / dataset_source_id,
        )
        raw_dir = _resolve_storage_path(
            job.paths.drive_root,
            None,
            job.paths.data_root / _default_raw_dir_name_for_dataset_source(dataset_source_id),
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    if source_mode != "benchmatch":
        job.logger.info(
            "dataset generation output isolation enabled | dataset_source_id=%s | isolated_raw_dir=%s | isolated_sampled_dir=%s",
            dataset_source_id,
            raw_dir,
            sampled_dir,
        )

    job.logger.info(
        "dataset generation startup | source_mode=%s | dataset_source_id=%s | source_dataset_id=%s | source_dataset_ids=%s | target_samples=%s | output_raw_dir=%s | output_sampled_dir=%s | workers=%s | max_pending_futures=%s | completed_game_detection_mode=%s",
        source_mode,
        dataset_source_id,
        source_dataset_id or "<none>",
        source_dataset_ids,
        target_samples,
        raw_dir,
        sampled_dir,
        num_workers,
        max_pending_futures,
        completed_game_detection_mode,
    )
    if global_target_enabled and source_mode == "benchmatch":
        firestore_diag = _firestore_backend_diagnostics(global_progress_backend)
        job.logger.info(
            "dataset generation global target backend config | backend=%s | project_id=%s | collection=%s | document=%s | auth_mode=%s | credentials_path_exists=%s | api_key_set=%s",
            str(firestore_diag.get("backend", "file")),
            str(firestore_diag.get("project_id", "")) or "<empty>",
            str(firestore_diag.get("collection", "")) or "<empty>",
            str(firestore_diag.get("document", "")) or "<empty>",
            str(firestore_diag.get("auth_mode", "")) or "<unknown>",
            bool(firestore_diag.get("credentials_path_exists", False)),
            bool(firestore_diag.get("api_key_set", False)),
        )
        state = _update_global_generation_progress(
            progress_path=global_progress_path,
            lock_dir=global_progress_lock_dir,
            global_target_id=global_target_id,
            target_samples=global_target_samples,
            job_id=job.job_id,
            dataset_source_id=dataset_source_id,
            delta_samples=0,
            delta_games=0,
            progress_backend=global_progress_backend,
        )
        job.logger.info(
            "dataset generation global target enabled | global_target_id=%s | target_samples=%s | current_global_samples=%s | progress_path=%s | backend=%s",
            global_target_id,
            global_target_samples,
            int(state.get("total_samples", 0)),
            global_progress_path,
            str(global_progress_backend.get("backend", "file")),
        )

    if source_mode not in {"benchmatch", "self_play_puct", "clone_existing", "derive_existing", "augment_existing", "merge_existing"}:
        raise ValueError(f"Unsupported dataset generation source_mode: {source_mode}")

    if source_mode == "benchmatch":
        matchups, benchmark_review_info = _augment_matchups_with_benchmark_review(
            job,
            cfg=cfg,
            base_matchups=[str(matchup) for matchup in matchups],
        )
        if bool(benchmark_review_info.get("enabled", False)):
            job.logger.info(
                "dataset generation benchmark review enrichment | added_matchups=%s | engine=%s | selected_opponents=%s | summary_path=%s",
                int(benchmark_review_info.get("added", 0)),
                str(benchmark_review_info.get("engine", "")) or "<none>",
                benchmark_review_info.get("selected_opponents", []),
                str(benchmark_review_info.get("summary_path", "")) or "<none>",
            )
            job.write_event(
                "dataset_generation_benchmark_review_enrichment",
                added_matchups=int(benchmark_review_info.get("added", 0)),
                engine=str(benchmark_review_info.get("engine", "")),
                selected_opponents=benchmark_review_info.get("selected_opponents", []),
                summary_path=str(benchmark_review_info.get("summary_path", "")),
                reason=str(benchmark_review_info.get("reason", "")),
            )
        matchups, tournament_review_info = _augment_matchups_with_tournament_review(
            job,
            cfg=cfg,
            base_matchups=[str(matchup) for matchup in matchups],
        )
        if bool(tournament_review_info.get("enabled", False)):
            job.logger.info(
                "dataset generation tournament review enrichment | added_matchups=%s | focus_model=%s | selected_opponents=%s | summary_path=%s",
                int(tournament_review_info.get("added", 0)),
                str(tournament_review_info.get("focus_model", "")) or "<none>",
                tournament_review_info.get("selected_opponents", []),
                str(tournament_review_info.get("summary_path", "")) or "<none>",
            )
            job.write_event(
                "dataset_generation_tournament_review_enrichment",
                added_matchups=int(tournament_review_info.get("added", 0)),
                focus_model=str(tournament_review_info.get("focus_model", "")),
                selected_opponents=tournament_review_info.get("selected_opponents", []),
                summary_path=str(tournament_review_info.get("summary_path", "")),
                reason=str(tournament_review_info.get("reason", "")),
                include_reverse_matchup=bool(tournament_review_info.get("include_reverse_matchup", False)),
            )

        validation_cache: dict[str, tuple[bool, str]] = {}
        valid_matchups: list[str] = []
        skipped_matchups: list[dict[str, str]] = []
        for matchup_spec in matchups:
            matchup_text = str(matchup_spec).strip()
            if not matchup_text:
                continue
            try:
                matchup_a, matchup_b = _parse_matchup(matchup_text)
            except Exception as exc:
                skipped_matchups.append({"matchup": matchup_text, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            ok_a, reason_a = _validate_generation_agent_spec(
                matchup_a,
                models_root=job.paths.models_root,
                cache=validation_cache,
            )
            ok_b, reason_b = _validate_generation_agent_spec(
                matchup_b,
                models_root=job.paths.models_root,
                cache=validation_cache,
            )
            if ok_a and ok_b:
                valid_matchups.append(matchup_text)
                continue
            failure_reasons: list[str] = []
            if not ok_a:
                failure_reasons.append(f"player_a={reason_a}")
            if not ok_b:
                failure_reasons.append(f"player_b={reason_b}")
            skipped_matchups.append({"matchup": matchup_text, "reason": " | ".join(failure_reasons)})
        if skipped_matchups:
            preview = skipped_matchups[:20]
            job.logger.warning(
                "dataset generation benchmatch filtered invalid matchups | kept=%s | skipped=%s | examples=%s",
                len(valid_matchups),
                len(skipped_matchups),
                preview,
            )
            job.write_event(
                "dataset_matchups_filtered",
                kept=len(valid_matchups),
                skipped=len(skipped_matchups),
                examples=preview,
            )
        if not valid_matchups:
            raise FileNotFoundError("Aucun matchup benchmatch valide (modeles manquants ou specs invalides).")
        matchups = valid_matchups

    if source_mode == "clone_existing":
        if not source_dataset_id:
            raise ValueError("`source_dataset_id` est requis quand `source_mode=clone_existing`")
        source_entry = _resolve_dataset_source(job, source_dataset_id)
        source_raw_dir = Path(str(source_entry["raw_dir"]))
        source_sampled_dir = Path(str(source_entry["sampled_dir"]))
        copied_raw_files = _copy_tree_incremental(source_raw_dir, raw_dir, pattern="*.json")
        copied_sampled_files = _copy_tree_incremental(source_sampled_dir, sampled_dir, pattern="*.jsonl")
        metadata = _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_id,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "source_dataset_id": source_dataset_id,
            "copied_raw_files": copied_raw_files,
            "copied_sampled_files": copied_sampled_files,
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "total_samples": int(metadata["sampled_positions"]),
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation cloned existing source | source_dataset_id=%s | dataset_source_id=%s | copied_raw_files=%s | copied_sampled_files=%s | total_samples=%s",
            source_dataset_id,
            dataset_source_id,
            copied_raw_files,
            copied_sampled_files,
            metadata["sampled_positions"],
        )
        job.write_event(
            "dataset_generation_cloned_existing",
            source_dataset_id=source_dataset_id,
            dataset_source_id=dataset_source_id,
            copied_raw_files=copied_raw_files,
            copied_sampled_files=copied_sampled_files,
        )
        job.write_metric(
            {
                "metric_type": "dataset_generation_cloned_existing",
                "source_dataset_id": source_dataset_id,
                "dataset_source_id": dataset_source_id,
                "copied_raw_files": copied_raw_files,
                "copied_sampled_files": copied_sampled_files,
            }
        )
        return summary

    if source_mode == "derive_existing":
        if not source_dataset_id:
            raise ValueError("`source_dataset_id` est requis quand `source_mode=derive_existing`")
        source_entry = _resolve_dataset_source(job, source_dataset_id)
        derived_summary = _derive_existing_dataset_source(
            source_entry=source_entry,
            target_raw_dir=raw_dir,
            target_sampled_dir=sampled_dir,
            target_samples=target_samples,
            derivation_strategy=derivation_strategy,
            derivation_params=derivation_params,
        )
        metadata = _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_id,
            derivation_strategy=derivation_strategy,
            derivation_params=derivation_params,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "source_dataset_id": source_dataset_id,
            "derivation_strategy": derivation_strategy,
            "derivation_params": derivation_params,
            **derived_summary,
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "total_samples": int(metadata["sampled_positions"]),
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation derived existing source | source_dataset_id=%s | dataset_source_id=%s | strategy=%s | selected_files=%s | selected_samples=%s",
            source_dataset_id,
            dataset_source_id,
            derivation_strategy,
            derived_summary["selected_files"],
            derived_summary["selected_samples"],
        )
        job.write_event(
            "dataset_generation_derived_existing",
            source_dataset_id=source_dataset_id,
            dataset_source_id=dataset_source_id,
            derivation_strategy=derivation_strategy,
            selected_files=derived_summary["selected_files"],
            selected_samples=derived_summary["selected_samples"],
        )
        job.write_metric(
            {
                "metric_type": "dataset_generation_derived_existing",
                "source_dataset_id": source_dataset_id,
                "dataset_source_id": dataset_source_id,
                "derivation_strategy": derivation_strategy,
                "selected_files": derived_summary["selected_files"],
                "selected_samples": derived_summary["selected_samples"],
            }
        )
        return summary

    if source_mode == "augment_existing":
        if not source_dataset_id:
            raise ValueError("`source_dataset_id` est requis quand `source_mode=augment_existing`")
        source_entry = _resolve_dataset_source(job, source_dataset_id)
        augmented_summary = _augment_existing_dataset_source(
            job=job,
            source_entry=source_entry,
            target_raw_dir=raw_dir,
            target_sampled_dir=sampled_dir,
            target_samples=target_samples,
            augmentation_params=derivation_params,
        )
        metadata = _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_id,
            derivation_params=augmented_summary["augmentation_params"],
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "source_dataset_id": source_dataset_id,
            **augmented_summary,
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "total_samples": int(metadata["sampled_positions"]),
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation augmented existing source | source_dataset_id=%s | dataset_source_id=%s | selected_original_samples=%s | selected_augmented_samples=%s | duplicate_samples=%s | total_samples=%s",
            source_dataset_id,
            dataset_source_id,
            augmented_summary["selected_original_samples"],
            augmented_summary["selected_augmented_samples"],
            augmented_summary["duplicate_samples"],
            metadata["sampled_positions"],
        )
        job.write_event(
            "dataset_generation_augmented_existing",
            source_dataset_id=source_dataset_id,
            dataset_source_id=dataset_source_id,
            selected_original_samples=augmented_summary["selected_original_samples"],
            selected_augmented_samples=augmented_summary["selected_augmented_samples"],
            duplicate_samples=augmented_summary["duplicate_samples"],
        )
        job.write_metric(
            {
                "metric_type": "dataset_generation_augmented_existing",
                "source_dataset_id": source_dataset_id,
                "dataset_source_id": dataset_source_id,
                "selected_original_samples": augmented_summary["selected_original_samples"],
                "selected_augmented_samples": augmented_summary["selected_augmented_samples"],
                "duplicate_samples": augmented_summary["duplicate_samples"],
                "total_samples": int(metadata["sampled_positions"]),
            }
        )
        return summary

    if source_mode == "merge_existing":
        if not source_dataset_ids:
            raise ValueError("`source_dataset_ids` est requis quand `source_mode=merge_existing`")
        source_entries = [_resolve_dataset_source(job, value) for value in source_dataset_ids]
        merged_summary = _merge_existing_dataset_sources(
            source_entries=source_entries,
            target_raw_dir=raw_dir,
            target_sampled_dir=sampled_dir,
            target_samples=target_samples,
            dedupe_sample_ids=merge_dedupe_sample_ids,
        )
        metadata = _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_ids[0],
            source_dataset_ids=source_dataset_ids,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "source_dataset_ids": source_dataset_ids,
            "merge_dedupe_sample_ids": merge_dedupe_sample_ids,
            **merged_summary,
            "raw_dir": str(raw_dir),
            "sampled_dir": str(sampled_dir),
            "total_samples": int(metadata["sampled_positions"]),
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation merged existing sources | dataset_source_id=%s | source_datasets=%s | selected_files=%s | selected_samples=%s | duplicate_samples=%s",
            dataset_source_id,
            len(source_dataset_ids),
            merged_summary["selected_files"],
            merged_summary["selected_samples"],
            merged_summary["duplicate_samples"],
        )
        for merged_source_id, merged_stats in merged_summary["source_breakdown"].items():
            job.logger.info(
                "dataset generation merged source breakdown | dataset_source_id=%s | source_dataset_id=%s | scanned_files=%s | scanned_samples=%s | selected_files=%s | selected_samples=%s | duplicate_samples=%s | copied_raw_files=%s",
                dataset_source_id,
                merged_source_id,
                merged_stats["scanned_files"],
                merged_stats["scanned_samples"],
                merged_stats["selected_files"],
                merged_stats["selected_samples"],
                merged_stats["duplicate_samples"],
                merged_stats["copied_raw_files"],
            )
        job.write_event(
            "dataset_generation_merged_existing",
            dataset_source_id=dataset_source_id,
            source_dataset_ids=source_dataset_ids,
            selected_files=merged_summary["selected_files"],
            selected_samples=merged_summary["selected_samples"],
            duplicate_samples=merged_summary["duplicate_samples"],
            source_breakdown=merged_summary["source_breakdown"],
        )
        job.write_metric(
            {
                "metric_type": "dataset_generation_merged_existing",
                "dataset_source_id": dataset_source_id,
                "source_datasets": len(source_dataset_ids),
                "selected_files": merged_summary["selected_files"],
                "selected_samples": merged_summary["selected_samples"],
                "duplicate_samples": merged_summary["duplicate_samples"],
            }
        )
        return summary

    if source_mode == "self_play_puct":
        return _run_dataset_generation_self_play_puct(
            job,
            cfg=cfg,
            source_mode=source_mode,
            dataset_source_id=dataset_source_id,
            source_dataset_id=source_dataset_id,
            source_dataset_ids=source_dataset_ids,
            derivation_strategy=derivation_strategy,
            derivation_params=derivation_params,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games=games,
            sample_every_n_plies=sample_every_n_plies,
            max_moves=max_moves,
            num_workers=num_workers,
            max_pending_futures=max_pending_futures,
            multiprocessing_start_method=multiprocessing_start_method,
            effective_max_tasks_per_child=effective_max_tasks_per_child,
            cycle_matchups_until_target=cycle_matchups_until_target,
            max_matchup_cycles=max_matchup_cycles,
            base_seed=base_seed,
        )

    state = job.read_state()
    summaries: list[dict[str, Any]] = []
    initial_total_samples = _count_total_jsonl_lines(sampled_dir)
    initial_total_games = _count_jsonl_files(sampled_dir)
    total_samples_after_run = initial_total_samples
    last_completed_game_id = str(state.get("last_completed_game_id", "")).strip()
    progress_update_every_n_games = max(
        1,
        int(job.config.get("dataset_generation", {}).get("progress_update_every_n_games", 25)),
    )
    global_budget_mode_cfg = str(cfg.get("global_budget_enforcement_mode", "")).strip().lower()
    if global_budget_mode_cfg not in {"strict", "batched"}:
        backend_name = str(global_progress_backend.get("backend", "file")).strip().lower()
        global_budget_mode_cfg = "batched" if backend_name == "firestore" else "strict"
    global_budget_reservation_enabled = bool(
        global_target_enabled
        and source_mode == "benchmatch"
        and global_target_samples > 0
        and global_budget_mode_cfg == "strict"
    )
    global_progress_flush_every_n_games = max(
        1,
        int(cfg.get("global_progress_flush_every_n_games", progress_update_every_n_games)),
    )
    global_target_poll_interval_seconds = max(
        1.0,
        float(cfg.get("global_target_poll_interval_seconds", 15.0)),
    )
    global_target_poll_cache: dict[str, Any] = {"checked_at": 0.0, "reached": False}
    global_progress_read_telemetry: dict[str, int] = {}
    job.logger.info(
        "dataset generation global budget control | mode=%s | strict_reservation=%s | progress_flush_every_n_games=%s | target_poll_interval_seconds=%.1f",
        global_budget_mode_cfg,
        global_budget_reservation_enabled,
        global_progress_flush_every_n_games,
        global_target_poll_interval_seconds,
    )

    job.logger.info("dataset generation started")
    job.set_phase("dataset_generation")
    job.write_event(
        "dataset_generation_started",
        existing_samples=initial_total_samples,
        target_samples=target_samples,
    )
    job.write_metric(
        {
            "metric_type": "dataset_generation_started",
            "existing_samples": initial_total_samples,
            "target_samples": target_samples,
        }
    )
    _write_dataset_generation_progress_snapshot(
        job=job,
        dataset_dir=dataset_dir,
        dataset_source_id=dataset_source_id,
        source_mode=source_mode,
        raw_dir=raw_dir,
        sampled_dir=sampled_dir,
        target_samples=target_samples,
        games_per_matchup=games,
        sample_every_n_plies=sample_every_n_plies,
        matchups=[str(matchup) for matchup in matchups],
        source_dataset_id=source_dataset_id,
        source_dataset_ids=source_dataset_ids,
        derivation_strategy=derivation_strategy,
        derivation_params=derivation_params,
        summaries=summaries,
        total_samples=total_samples_after_run,
    )

    if target_samples > 0 and initial_total_samples >= target_samples:
        _register_dataset_source(
            job,
            dataset_source_id=dataset_source_id,
            source_mode=source_mode,
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            target_samples=target_samples,
            games_per_matchup=games,
            sample_every_n_plies=sample_every_n_plies,
            matchups=[str(matchup) for matchup in matchups],
            source_dataset_id=source_dataset_id,
            derivation_strategy=derivation_strategy,
            derivation_params=derivation_params,
        )
        summary = {
            "job_id": job.job_id,
            "dataset_source_id": dataset_source_id,
            "source_mode": source_mode,
            "matchups": [],
            "existing_samples": initial_total_samples,
            "added_samples": 0,
            "total_samples": initial_total_samples,
            "target_samples": target_samples,
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation skipped | target already reached | existing_samples=%s | target_samples=%s",
            initial_total_samples,
            target_samples,
        )
        job.write_state(
            {
                "current_matchup": None,
                "completed_games": 0,
                "remaining_games": 0,
                "last_completed_game_id": state.get("last_completed_game_id"),
                "sample_count": initial_total_samples,
                "target_samples": target_samples,
            }
        )
        return summary

    cycle_index = 0
    while True:
        cycle_index += 1
        cycle_samples_before = total_samples_after_run
        cycle_stop_requested = False
        for matchup_index, matchup_spec in enumerate(matchups):
            matchup_a, matchup_b = _parse_matchup(str(matchup_spec))
            matchup_id = _slugify_matchup(str(matchup_spec))
            job.set_phase(f"dataset_generation:{matchup_id}")
            summary_path = dataset_dir / f"{matchup_id}_summary.json"
            if target_samples > 0 and total_samples_after_run >= target_samples:
                job.logger.info(
                    "dataset generation target reached during run | current_samples=%s | target_samples=%s",
                    total_samples_after_run,
                    target_samples,
                )
                break
    
            matchup_game_count = 0
            matchup_sample_count = 0
            matchup_failed_game_count = 0
            global_stop_requested = False
            pending_global_delta_samples = 0
            pending_global_delta_games = 0
            existing_numbers = _existing_game_numbers(
                raw_dir,
                sampled_dir,
                matchup_id,
                completion_mode=completed_game_detection_mode,
            )
            existing_game_count = len(existing_numbers)
            job.logger.info(
                "dataset matchup started | %s/%s | %s vs %s | add_games=%s | existing_games=%s | sample_every=%s | workers=%s | target_samples=%s",
                matchup_index + 1,
                len(matchups),
                matchup_a,
                matchup_b,
                games,
                existing_game_count,
                sample_every_n_plies,
                num_workers,
                target_samples,
            )
            job.write_event(
                "dataset_matchup_started",
                matchup=matchup_id,
                matchup_index=matchup_index + 1,
                total_matchups=len(matchups),
                player_a=matchup_a,
                player_b=matchup_b,
                add_games=games,
                existing_games=existing_game_count,
                sample_every_n_plies=sample_every_n_plies,
                num_workers=num_workers,
            )
    
            pending_games = _build_pending_incremental_games(
                raw_dir=raw_dir,
                sampled_dir=sampled_dir,
                matchup_id=matchup_id,
                games_to_add=games,
                seed_base=base_seed + (matchup_index * 1_000_000),
                sample_every_n_plies=sample_every_n_plies,
                completion_mode=completed_game_detection_mode,
            )
    
            def _update_benchmatch_progress(completed_game_id: str) -> None:
                nonlocal last_completed_game_id
                resolved_game_id = str(completed_game_id).strip()
                if resolved_game_id:
                    last_completed_game_id = resolved_game_id
                _write_benchmatch_progress_snapshot(
                    job=job,
                    dataset_dir=dataset_dir,
                    dataset_source_id=dataset_source_id,
                    raw_dir=raw_dir,
                    sampled_dir=sampled_dir,
                    target_samples=target_samples,
                    games_per_matchup=games,
                    sample_every_n_plies=sample_every_n_plies,
                    matchups=[str(matchup) for matchup in matchups],
                    source_dataset_id=source_dataset_id,
                    source_dataset_ids=source_dataset_ids,
                    derivation_strategy=derivation_strategy,
                    derivation_params=derivation_params,
                    summaries=summaries + [{
                        "matchup_id": matchup_id,
                        "player_a": matchup_a,
                        "player_b": matchup_b,
                        "existing_games": existing_game_count,
                        "games_added": matchup_game_count,
                        "samples_added": matchup_sample_count,
                        "sample_every_n_plies": sample_every_n_plies,
                        "num_workers": num_workers,
                    }],
                    total_samples=total_samples_after_run,
                )
                job.write_state(
                    {
                        "current_matchup": matchup_id,
                        "completed_games": matchup_game_count,
                        "remaining_games": games - matchup_game_count,
                        "last_completed_game_id": last_completed_game_id,
                        "sample_count": total_samples_after_run,
                        "target_samples": target_samples,
                    }
                )
    
            def _global_target_reached() -> bool:
                if not (global_target_enabled and source_mode == "benchmatch" and global_target_samples > 0):
                    return False
                now_ts = time.time()
                cached_checked_at = float(global_target_poll_cache.get("checked_at", 0.0) or 0.0)
                if (now_ts - cached_checked_at) < float(global_target_poll_interval_seconds):
                    return bool(global_target_poll_cache.get("reached", False))
                state = _read_global_generation_progress(
                    global_progress_path,
                    global_target_id=global_target_id,
                    target_samples=global_target_samples,
                    progress_backend=global_progress_backend,
                    prefer_cache=False,
                    telemetry=global_progress_read_telemetry,
                )
                reached = int(state.get("total_samples", 0)) >= int(global_target_samples)
                global_target_poll_cache["checked_at"] = now_ts
                global_target_poll_cache["reached"] = bool(reached)
                return bool(reached)
    
            def _local_target_remaining() -> int:
                if target_samples <= 0:
                    return 1_000_000_000
                return max(0, int(target_samples) - int(total_samples_after_run))

            def _flush_global_progress_deltas(*, force: bool = False) -> None:
                nonlocal pending_global_delta_samples, pending_global_delta_games
                if global_budget_reservation_enabled:
                    return
                if not (global_target_enabled and source_mode == "benchmatch"):
                    return
                if pending_global_delta_samples <= 0 and pending_global_delta_games <= 0:
                    return
                if not force and (matchup_game_count % global_progress_flush_every_n_games != 0):
                    return
                state = _update_global_generation_progress(
                    progress_path=global_progress_path,
                    lock_dir=global_progress_lock_dir,
                    global_target_id=global_target_id,
                    target_samples=global_target_samples,
                    job_id=job.job_id,
                    dataset_source_id=dataset_source_id,
                    delta_samples=int(pending_global_delta_samples),
                    delta_games=int(pending_global_delta_games),
                    progress_backend=global_progress_backend,
                )
                pending_global_delta_samples = 0
                pending_global_delta_games = 0
                global_target_poll_cache["checked_at"] = time.time()
                global_target_poll_cache["reached"] = (
                    int(state.get("total_samples", 0)) >= int(global_target_samples)
                )

            def _accumulate_global_progress_deltas(*, games_inc: int, samples_inc: int) -> None:
                nonlocal pending_global_delta_samples, pending_global_delta_games
                if global_budget_reservation_enabled:
                    return
                if int(games_inc) <= 0 and int(samples_inc) <= 0:
                    return
                pending_global_delta_samples += max(0, int(samples_inc))
                pending_global_delta_games += max(0, int(games_inc))
                _flush_global_progress_deltas(force=False)

            def _release_reserved_global_budget(
                *,
                pending: dict[str, Any],
                reserved_samples: int,
                reserved_games: int,
                reason: str,
                error: Exception | None = None,
            ) -> None:
                release_samples = max(0, int(reserved_samples))
                release_games = max(0, int(reserved_games))
                if not global_budget_reservation_enabled or (release_samples <= 0 and release_games <= 0):
                    return
                try:
                    released_state = _update_global_generation_progress(
                        progress_path=global_progress_path,
                        lock_dir=global_progress_lock_dir,
                        global_target_id=global_target_id,
                        target_samples=global_target_samples,
                        job_id=job.job_id,
                        dataset_source_id=dataset_source_id,
                        delta_samples=-release_samples,
                        delta_games=-release_games,
                        progress_backend=global_progress_backend,
                    )
                    job.logger.warning(
                        "dataset global budget released | reason=%s | matchup=%s | game=%s/%s | released_samples=%s | released_games=%s | global_total_samples=%s",
                        reason,
                        matchup_id,
                        pending["game_no"],
                        games,
                        release_samples,
                        release_games,
                        int(released_state.get("total_samples", 0)),
                    )
                    job.write_event(
                        "dataset_global_budget_released",
                        reason=reason,
                        matchup=matchup_id,
                        game_id=pending["game_id"],
                        game_index=pending["game_no"],
                        released_samples=release_samples,
                        released_games=release_games,
                        error=f"{type(error).__name__}: {error}" if error is not None else "",
                    )
                except Exception as release_exc:
                    job.logger.error(
                        "dataset global budget release failed | reason=%s | matchup=%s | game=%s/%s | released_samples=%s | released_games=%s | error=%s: %s",
                        reason,
                        matchup_id,
                        pending["game_no"],
                        games,
                        release_samples,
                        release_games,
                        type(release_exc).__name__,
                        release_exc,
                    )
                    job.write_event(
                        "dataset_global_budget_release_failed",
                        reason=reason,
                        matchup=matchup_id,
                        game_id=pending["game_id"],
                        game_index=pending["game_no"],
                        released_samples=release_samples,
                        released_games=release_games,
                        release_error=f"{type(release_exc).__name__}: {release_exc}",
                    )

            def _resolve_admitted_samples(pending: dict[str, Any], requested_samples: int) -> int:
                requested = max(0, int(requested_samples))
                if requested <= 0:
                    return 0
                admitted = min(requested, _local_target_remaining())
                if admitted <= 0:
                    return 0
                if not global_budget_reservation_enabled:
                    if _global_target_reached():
                        return 0
                    return admitted
                admitted_samples, _admitted_games, _state = _reserve_global_generation_budget(
                    progress_path=global_progress_path,
                    lock_dir=global_progress_lock_dir,
                    global_target_id=global_target_id,
                    target_samples=global_target_samples,
                    job_id=job.job_id,
                    dataset_source_id=dataset_source_id,
                    requested_samples=admitted,
                    requested_games=1,
                    progress_backend=global_progress_backend,
                )
                return int(admitted_samples)

            def _make_sequential_progress_callback(base_games: int, base_samples: int) -> Callable[[dict[str, Any], int, int], None]:
                def _on_sequential_game_completed(pending: dict[str, Any], completed_games_so_far: int, completed_samples_so_far: int) -> None:
                    nonlocal matchup_game_count, matchup_sample_count, total_samples_after_run, last_completed_game_id
                    previous_games = int(matchup_game_count)
                    previous_samples = int(matchup_sample_count)
                    matchup_game_count = int(base_games) + int(completed_games_so_far)
                    matchup_sample_count = int(base_samples) + int(completed_samples_so_far)
                    games_inc = max(0, int(matchup_game_count) - previous_games)
                    samples_inc = max(0, int(matchup_sample_count) - previous_samples)
                    total_samples_after_run = initial_total_samples + sum(int(item.get("samples_added", 0)) for item in summaries) + matchup_sample_count
                    last_completed_game_id = str(pending["game_id"])
                    _accumulate_global_progress_deltas(games_inc=games_inc, samples_inc=samples_inc)
                    if matchup_game_count % progress_update_every_n_games == 0:
                        _update_benchmatch_progress(str(pending["game_id"]))

                return _on_sequential_game_completed

            def _on_sequential_game_failed(pending: dict[str, Any], exc: Exception) -> None:
                nonlocal matchup_failed_game_count
                matchup_failed_game_count += 1
                job.logger.warning(
                    "dataset game failed | matchup=%s | game=%s/%s | error=%s: %s | mode=sequential",
                    matchup_id,
                    pending["game_no"],
                    games,
                    type(exc).__name__,
                    exc,
                )
                job.write_event(
                    "dataset_game_failed",
                    matchup=matchup_id,
                    game_id=pending["game_id"],
                    game_index=pending["game_no"],
                    execution_mode="sequential",
                    error=f"{type(exc).__name__}: {exc}",
                )

            def _on_sequential_materialization_error(pending: dict[str, Any], reserved_samples: int, exc: Exception) -> None:
                reserved_games = 1 if (global_budget_reservation_enabled and int(reserved_samples) > 0) else 0
                _release_reserved_global_budget(
                    pending=pending,
                    reserved_samples=int(reserved_samples),
                    reserved_games=reserved_games,
                    reason="materialization_failed_sequential",
                    error=exc,
                )

            if num_workers <= 1:
                completed_games, completed_samples, _execution_mode = _run_pending_games_sequential(
                    job,
                    pending_games=pending_games,
                    matchup_a=matchup_a,
                    matchup_b=matchup_b,
                    matchup_id=matchup_id,
                    games=games,
                    sample_every_n_plies=sample_every_n_plies,
                    max_moves=max_moves,
                    models_root=job.paths.models_root,
                    model_agent_device=model_agent_device,
                    on_game_completed=_make_sequential_progress_callback(0, 0),
                    sample_cap_resolver=lambda pending, sample_count: _resolve_admitted_samples(pending, sample_count),
                    on_game_failed=_on_sequential_game_failed,
                    on_materialization_error=_on_sequential_materialization_error,
                )
                matchup_game_count = int(completed_games)
                matchup_sample_count = int(completed_samples)
                if last_completed_game_id:
                    _update_benchmatch_progress(last_completed_game_id)
                _flush_global_progress_deltas(force=True)
            else:
                job.logger.info(
                    "dataset matchup parallel execution | matchup=%s | workers=%s | pending_games=%s | max_pending=%s | start_method=%s | max_tasks_per_child=%s",
                    matchup_id,
                    num_workers,
                    len(pending_games),
                    max_pending_futures,
                    multiprocessing_start_method,
                    effective_max_tasks_per_child,
                )
                job.write_event(
                    "dataset_parallel_execution_started",
                    matchup=matchup_id,
                    workers=num_workers,
                    pending_games=len(pending_games),
                    max_pending_futures=max_pending_futures,
                    multiprocessing_start_method=multiprocessing_start_method,
                    max_tasks_per_child=effective_max_tasks_per_child,
                )
    
                future_map: dict[concurrent.futures.Future, dict[str, Any]] = {}
                pending_queue = list(pending_games)
                try:
                    mp_context = multiprocessing.get_context(multiprocessing_start_method)
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=num_workers,
                        mp_context=mp_context,
                        max_tasks_per_child=effective_max_tasks_per_child,
                    ) as executor:
                        while pending_queue or future_map:
                            if _global_target_reached():
                                global_stop_requested = True
                                pending_queue.clear()
                            while pending_queue and len(future_map) < max_pending_futures:
                                if _global_target_reached():
                                    global_stop_requested = True
                                    pending_queue.clear()
                                    break
                                pending = pending_queue.pop(0)
                                job.logger.info(
                                    "dataset game scheduled | matchup=%s | game=%s/%s | seed=%s | starter=%s | sample_offset=%s | mode=parallel",
                                    matchup_id,
                                    pending["game_no"],
                                    games,
                                    pending["seed"],
                                    pending["starter"],
                                    pending.get("sample_ply_offset", 0),
                                )
                                future = executor.submit(
                                    _play_and_sample_game_from_specs,
                                    matchup_a,
                                    matchup_b,
                                    matchup_id=matchup_id,
                                    game_id=str(pending["game_id"]),
                                    seed=int(pending["seed"]),
                                    starter=int(pending["starter"]),
                                    sample_every_n_plies=sample_every_n_plies,
                                    sample_ply_offset=int(pending.get("sample_ply_offset", 0)),
                                    max_moves=max_moves,
                                    models_root=str(job.paths.models_root),
                                    model_agent_device=model_agent_device,
                                )
                                future_map[future] = pending
    
                            done, _not_done = concurrent.futures.wait(
                                future_map.keys(),
                                return_when=concurrent.futures.FIRST_COMPLETED,
                            )
                            for future in done:
                                pending = future_map.pop(future)
                                try:
                                    raw_payload, samples = future.result()
                                except Exception as exc:
                                    matchup_failed_game_count += 1
                                    job.logger.warning(
                                        "dataset game failed | matchup=%s | game=%s/%s | error=%s: %s | mode=parallel",
                                        matchup_id,
                                        pending["game_no"],
                                        games,
                                        type(exc).__name__,
                                        exc,
                                    )
                                    job.write_event(
                                        "dataset_game_failed",
                                        matchup=matchup_id,
                                        game_id=pending["game_id"],
                                        game_index=pending["game_no"],
                                        execution_mode="parallel",
                                        error=f"{type(exc).__name__}: {exc}",
                                    )
                                    continue
                                admitted_samples = _resolve_admitted_samples(pending, len(samples))
                                try:
                                    games_inc, samples_inc = _materialize_completed_game(
                                        job,
                                        pending=pending,
                                        raw_payload=raw_payload,
                                        samples=samples,
                                        matchup_id=matchup_id,
                                        games=games,
                                        execution_mode="parallel",
                                        max_samples_to_write=admitted_samples,
                                    )
                                except Exception as exc:
                                    _release_reserved_global_budget(
                                        pending=pending,
                                        reserved_samples=int(admitted_samples),
                                        reserved_games=(1 if (global_budget_reservation_enabled and int(admitted_samples) > 0) else 0),
                                        reason="materialization_failed_parallel",
                                        error=exc,
                                    )
                                    matchup_failed_game_count += 1
                                    job.logger.warning(
                                        "dataset game failed at materialization | matchup=%s | game=%s/%s | error=%s: %s | mode=parallel",
                                        matchup_id,
                                        pending["game_no"],
                                        games,
                                        type(exc).__name__,
                                        exc,
                                    )
                                    job.write_event(
                                        "dataset_game_failed",
                                        matchup=matchup_id,
                                        game_id=pending["game_id"],
                                        game_index=pending["game_no"],
                                        execution_mode="parallel",
                                        error=f"{type(exc).__name__}: {exc}",
                                    )
                                    continue
                                matchup_game_count += games_inc
                                matchup_sample_count += samples_inc
                                total_samples_after_run += samples_inc
                                _accumulate_global_progress_deltas(games_inc=games_inc, samples_inc=samples_inc)
                                if games_inc > 0:
                                    last_completed_game_id = str(pending["game_id"])
                                if _global_target_reached():
                                    global_stop_requested = True
                                    pending_queue.clear()
                                if games_inc > 0 and matchup_game_count % progress_update_every_n_games == 0:
                                    _update_benchmatch_progress(str(pending["game_id"]))
                except concurrent.futures.process.BrokenProcessPool as exc:
                    failed_pending = list(future_map.values()) + list(pending_queue)
                    job.logger.warning(
                        "dataset parallel pool broken | matchup=%s | completed_games=%s | remaining_fallback=%s | error=%s",
                        matchup_id,
                        matchup_game_count,
                        len(failed_pending),
                        exc,
                    )
                    job.write_event(
                        "dataset_parallel_execution_broken",
                        matchup=matchup_id,
                        completed_games=matchup_game_count,
                        remaining_fallback=len(failed_pending),
                        error=str(exc),
                    )
                    completed_games, completed_samples, _execution_mode = _run_pending_games_sequential(
                        job,
                        pending_games=failed_pending,
                        matchup_a=matchup_a,
                        matchup_b=matchup_b,
                        matchup_id=matchup_id,
                        games=games,
                        sample_every_n_plies=sample_every_n_plies,
                        max_moves=max_moves,
                        models_root=job.paths.models_root,
                        model_agent_device=model_agent_device,
                        on_game_completed=_make_sequential_progress_callback(matchup_game_count, matchup_sample_count),
                        sample_cap_resolver=lambda pending, sample_count: _resolve_admitted_samples(pending, sample_count),
                        on_game_failed=_on_sequential_game_failed,
                        on_materialization_error=_on_sequential_materialization_error,
                    )
                    matchup_game_count = int(matchup_game_count)
                    matchup_sample_count = int(matchup_sample_count)
                    if failed_pending and last_completed_game_id:
                        job.write_state(
                            {
                                "current_matchup": matchup_id,
                                "completed_games": matchup_game_count,
                                "remaining_games": games - matchup_game_count,
                                "last_completed_game_id": last_completed_game_id,
                                "sample_count": total_samples_after_run,
                                "target_samples": target_samples,
                            }
                        )
                _flush_global_progress_deltas(force=True)
            if global_stop_requested:
                job.logger.info(
                    "dataset generation global target reached, stopping early | global_target_id=%s | target_samples=%s",
                    global_target_id,
                    global_target_samples,
                )

            summary_payload = {
                "matchup_id": matchup_id,
                "cycle_index": cycle_index,
                "player_a": matchup_a,
                "player_b": matchup_b,
                "existing_games": existing_game_count,
                "games_added": matchup_game_count,
                "games_failed": matchup_failed_game_count,
                "samples_added": matchup_sample_count,
                "sample_every_n_plies": sample_every_n_plies,
                "num_workers": num_workers,
            }
            _write_json(summary_path, summary_payload)
            _accumulate_matchup_summary(summaries, summary_payload)
            job.logger.info(
                "dataset matchup completed | cycle=%s | %s vs %s | games_added=%s | games_failed=%s | samples_added=%s | total_samples=%s",
                cycle_index,
                matchup_a,
                matchup_b,
                matchup_game_count,
                matchup_failed_game_count,
                matchup_sample_count,
                total_samples_after_run,
            )
            job.write_state(
                {
                    "current_matchup": matchup_id,
                    "completed_games": matchup_game_count,
                    "remaining_games": 0,
                    "last_completed_game_id": last_completed_game_id,
                    "sample_count": total_samples_after_run,
                    "target_samples": target_samples,
                }
            )
            job.write_metric({"metric_type": "dataset_matchup_completed", **summary_payload})
            job.write_event("dataset_matchup_completed", matchup=matchup_id, samples=matchup_sample_count, cycle_index=cycle_index)
            _write_benchmatch_progress_snapshot(
                job=job,
                dataset_dir=dataset_dir,
                dataset_source_id=dataset_source_id,
                raw_dir=raw_dir,
                sampled_dir=sampled_dir,
                target_samples=target_samples,
                games_per_matchup=games,
                sample_every_n_plies=sample_every_n_plies,
                matchups=[str(matchup) for matchup in matchups],
                source_dataset_id=source_dataset_id,
                source_dataset_ids=source_dataset_ids,
                derivation_strategy=derivation_strategy,
                derivation_params=derivation_params,
                summaries=summaries,
                total_samples=total_samples_after_run,
            )
            if global_stop_requested:
                cycle_stop_requested = True
                break
    
        if cycle_stop_requested:
            break
        if target_samples > 0 and total_samples_after_run >= target_samples:
            break
        if not (source_mode == "benchmatch" and cycle_matchups_until_target):
            break
        if max_matchup_cycles > 0 and cycle_index >= max_matchup_cycles:
            job.logger.info(
                "dataset generation stopping after max cycles | cycles=%s | max_matchup_cycles=%s | total_samples=%s | target_samples=%s",
                cycle_index,
                max_matchup_cycles,
                total_samples_after_run,
                target_samples,
            )
            break
        cycle_added_samples = int(total_samples_after_run) - int(cycle_samples_before)
        if cycle_added_samples <= 0:
            job.logger.warning(
                "dataset generation cycle produced no samples, stopping to avoid busy-loop | cycle=%s | total_samples=%s | target_samples=%s",
                cycle_index,
                total_samples_after_run,
                target_samples,
            )
            break
        job.logger.info(
            "dataset generation cycle completed, continuing | cycle=%s | cycle_added_samples=%s | total_samples=%s | target_samples=%s",
            cycle_index,
            cycle_added_samples,
            total_samples_after_run,
            target_samples,
        )

    final_total_samples = _count_total_jsonl_lines(sampled_dir)
    final_total_games = _count_jsonl_files(sampled_dir)
    added_games = max(0, int(final_total_games) - int(initial_total_games))
    failed_games = sum(int(item.get("games_failed", 0)) for item in summaries)
    added_samples = max(0, int(final_total_samples) - int(initial_total_samples))
    global_target_reached_final = False
    if global_target_enabled and source_mode == "benchmatch" and global_target_samples > 0:
        global_state = _read_global_generation_progress(
            global_progress_path,
            global_target_id=global_target_id,
            target_samples=global_target_samples,
            progress_backend=global_progress_backend,
            prefer_cache=False,
            telemetry=global_progress_read_telemetry,
        )
        global_target_reached_final = int(global_state.get("total_samples", 0)) >= int(global_target_samples)
    local_target_reached_final = target_samples > 0 and int(final_total_samples) >= int(target_samples)
    source_status = "completed" if (local_target_reached_final or global_target_reached_final) else "partial"
    summary = {
        "job_id": job.job_id,
        "dataset_source_id": dataset_source_id,
        "source_mode": source_mode,
        "matchups": summaries,
        "cycles_completed": cycle_index,
        "source_status": source_status,
        "existing_samples": initial_total_samples,
        "added_games": added_games,
        "failed_games": failed_games,
        "added_samples": added_samples,
        "total_samples": final_total_samples,
        "target_samples": target_samples,
    }
    _register_dataset_source(
        job,
        dataset_source_id=dataset_source_id,
        source_mode=source_mode,
        raw_dir=raw_dir,
        sampled_dir=sampled_dir,
        target_samples=target_samples,
        games_per_matchup=games,
        sample_every_n_plies=sample_every_n_plies,
        matchups=[str(matchup) for matchup in matchups],
        source_dataset_id=source_dataset_id,
        source_dataset_ids=source_dataset_ids,
        derivation_strategy=derivation_strategy,
        derivation_params=derivation_params,
        source_status=source_status,
        partial_summary={} if source_status == "completed" else {
            "cycles_completed": cycle_index,
            "added_games": added_games,
            "failed_games": failed_games,
            "added_samples": added_samples,
            "total_samples": final_total_samples,
            "target_samples": target_samples,
        },
        raw_files_override=_count_json_files(raw_dir),
        sampled_files_override=_count_jsonl_files(sampled_dir),
        sampled_positions_override=final_total_samples,
    )
    _write_json(dataset_dir / "dataset_generation_summary.json", summary)
    job.logger.info(
        "dataset generation completed | source_status=%s | cycles=%s | matchups=%s | added_games=%s | failed_games=%s | added_samples=%s | total_samples=%s | target_samples=%s",
        source_status,
        cycle_index,
        len(summaries),
        added_games,
        failed_games,
        added_samples,
        final_total_samples,
        target_samples,
    )
    job.write_state(
        {
            "current_matchup": None,
            "completed_games": added_games,
            "remaining_games": 0,
            "last_completed_game_id": last_completed_game_id,
            "sample_count": final_total_samples,
            "target_samples": target_samples,
        }
    )
    if global_target_enabled and source_mode == "benchmatch":
        job.write_metric(
            {
                "metric_type": "dataset_global_progress_read_telemetry",
                "redis_hit": int(global_progress_read_telemetry.get("redis_hit", 0)),
                "redis_miss": int(global_progress_read_telemetry.get("redis_miss", 0)),
                "redis_error": int(global_progress_read_telemetry.get("redis_error", 0)),
                "fallback_firestore": int(global_progress_read_telemetry.get("fallback_firestore", 0)),
                "firestore_read": int(global_progress_read_telemetry.get("firestore_read", 0)),
            }
        )
    return summary


def run_dataset_build(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("dataset_build", {})
    runtime_cfg = job.config.get("runtime", {})
    teacher_cfg = cfg.get("teacher", {})
    teacher_engine = str(teacher_cfg.get("engine", "minimax"))
    teacher_level = str(teacher_cfg.get("level", "hard"))
    dataset_id = str(cfg.get("dataset_id", "dataset_v1"))
    source_dataset_id = str(cfg.get("source_dataset_id", "")).strip()
    build_mode_requested = str(cfg.get("build_mode", "auto")).strip().lower() or "auto"
    if build_mode_requested not in {"auto", "teacher_label", "source_prelabeled"}:
        raise ValueError(
            f"Unsupported dataset_build.build_mode: {build_mode_requested} (attendu: auto|teacher_label|source_prelabeled)"
        )
    configured_follow_source_updates = _as_bool(cfg.get("follow_source_updates", False), default=False)
    source_poll_interval_seconds = max(1.0, float(cfg.get("source_poll_interval_seconds", 30.0)))
    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.8))
    validation_ratio = float(split_cfg.get("validation", 0.1))
    num_workers = max(1, int(cfg.get("num_workers", runtime_cfg.get("num_workers", 1))))
    max_pending_futures = max(1, int(cfg.get("max_pending_futures", num_workers * 2)))
    multiprocessing_start_method = str(runtime_cfg.get("multiprocessing_start_method", "spawn")).strip().lower() or "spawn"
    max_tasks_per_child = int(runtime_cfg.get("max_tasks_per_child", 25))
    include_tactical_analysis = _as_bool(cfg.get("include_tactical_analysis", True), default=True)
    value_target_mix_teacher_weight = max(
        0.0,
        min(
            1.0,
            float(
                cfg.get(
                    "value_target_mix_teacher_weight",
                    cfg.get("value_target_mix", 1.0),
                )
            ),
        ),
    )
    hard_examples_enabled = _as_bool(cfg.get("hard_examples_enabled", True), default=True)
    hard_examples_margin_threshold = max(1e-6, float(cfg.get("hard_examples_margin_threshold", 0.08)))
    hard_examples_outcome_focus = max(0.0, min(1.0, float(cfg.get("hard_examples_outcome_focus", 0.35))))
    hard_examples_weight_multiplier = max(1.0, float(cfg.get("hard_examples_weight_multiplier", 2.0)))
    dedupe_sample_ids = _as_bool(cfg.get("dedupe_sample_ids", True), default=True)
    export_partial_every_n_files = max(1, int(cfg.get("export_partial_every_n_files", 250)))
    missing_source_file_retry_attempts = max(1, int(cfg.get("missing_source_file_retry_attempts", 5)))
    missing_source_file_retry_delay_seconds = max(0.0, float(cfg.get("missing_source_file_retry_delay_seconds", 2.0)))
    progressive_global_merge_enabled = _as_bool(cfg.get("progressive_global_merge_enabled", False), default=False)
    progressive_global_merge_dataset_id = str(cfg.get("progressive_global_merge_dataset_id", "")).strip()
    progressive_global_merge_source_dataset_id_prefix = str(
        cfg.get("progressive_global_merge_source_dataset_id_prefix", "")
    ).strip()
    progressive_global_merge_include_partial = _as_bool(
        cfg.get("progressive_global_merge_include_partial", True),
        default=True,
    )
    progressive_global_merge_every_n_partial_exports = max(
        1,
        int(cfg.get("progressive_global_merge_every_n_partial_exports", 1)),
    )
    progressive_global_merge_min_interval_seconds = max(
        0.0,
        float(cfg.get("progressive_global_merge_min_interval_seconds", 300.0)),
    )
    progressive_global_merge_min_sources = max(1, int(cfg.get("progressive_global_merge_min_sources", 2)))
    progressive_global_merge_on_completion = _as_bool(
        cfg.get("progressive_global_merge_on_completion", True),
        default=True,
    )
    progressive_global_merge_dedupe_sample_ids = _as_bool(
        cfg.get("progressive_global_merge_dedupe_sample_ids", dedupe_sample_ids),
        default=dedupe_sample_ids,
    )
    progressive_global_merge_lock_ttl_seconds = max(
        30.0,
        float(cfg.get("progressive_global_merge_lock_ttl_seconds", 1800.0)),
    )
    progressive_global_merge_lock_wait_seconds = max(
        1.0,
        float(cfg.get("progressive_global_merge_lock_wait_seconds", 30.0)),
    )
    progressive_global_merge_async = _as_bool(
        cfg.get("progressive_global_merge_async", True),
        default=True,
    )
    progressive_global_merge_candidates_cache_ttl_seconds = max(
        5.0,
        float(cfg.get("progressive_global_merge_candidates_cache_ttl_seconds", 30.0)),
    )
    progressive_global_merge_require_data_delta = _as_bool(
        cfg.get("progressive_global_merge_require_data_delta", True),
        default=True,
    )
    progressive_global_merge_completion_wait_seconds = max(
        5.0,
        float(cfg.get("progressive_global_merge_completion_wait_seconds", 600.0)),
    )
    if progressive_global_merge_enabled and not progressive_global_merge_dataset_id:
        job.logger.warning(
            "dataset build progressive global merge disabled | reason=missing progressive_global_merge_dataset_id"
        )
        progressive_global_merge_enabled = False
    adaptive_source_polling = _as_bool(cfg.get("adaptive_source_polling", True), default=True)
    source_poll_interval_min_seconds = max(1.0, float(cfg.get("source_poll_interval_min_seconds", max(1.0, source_poll_interval_seconds / 2.0))))
    source_poll_interval_max_seconds = max(source_poll_interval_min_seconds, float(cfg.get("source_poll_interval_max_seconds", max(source_poll_interval_seconds * 4.0, source_poll_interval_min_seconds))))
    current_poll_interval_seconds = source_poll_interval_seconds
    stop_when_global_target_reached = _as_bool(cfg.get("stop_when_global_target_reached", True), default=True)
    global_target_progress_path = _resolve_storage_path(
        job.paths.drive_root,
        cfg.get("global_target_progress_path"),
        job.paths.data_root / "global_generation_progress" / "bench_models_20m_global.json",
    )
    global_target_samples_cfg = int(cfg.get("global_target_samples", 0))
    global_target_id = str(cfg.get("global_target_id", "")).strip() or str(Path(global_target_progress_path).stem)
    global_progress_backend = _resolve_global_progress_backend_config(
        cfg=cfg,
        global_target_id=global_target_id,
        target_samples=global_target_samples_cfg,
    )
    global_target_stabilization_polls = max(1, int(cfg.get("global_target_stabilization_polls", 3)))
    global_target_reached_polls = 0
    global_progress_read_telemetry: dict[str, int] = {}
    effective_max_tasks_per_child = _resolve_pool_max_tasks_per_child(
        start_method=multiprocessing_start_method,
        configured_value=max_tasks_per_child,
        logger=job.logger,
        scope="dataset build pool configuration",
    )

    def _write_global_progress_read_telemetry_metric() -> None:
        job.write_metric(
            {
                "metric_type": "dataset_build_global_progress_read_telemetry",
                "redis_hit": int(global_progress_read_telemetry.get("redis_hit", 0)),
                "redis_miss": int(global_progress_read_telemetry.get("redis_miss", 0)),
                "redis_error": int(global_progress_read_telemetry.get("redis_error", 0)),
                "fallback_firestore": int(global_progress_read_telemetry.get("fallback_firestore", 0)),
                "firestore_read": int(global_progress_read_telemetry.get("firestore_read", 0)),
            }
        )

    job.logger.info(
        "dataset build startup | dataset=%s | source_dataset_id=%s | configured_input_sampled_dir=%s | build_mode_requested=%s | workers=%s | max_pending_futures=%s | include_tactical_analysis=%s | value_target_mix_teacher_weight=%.3f | hard_examples_enabled=%s | hard_examples_margin_threshold=%.4f | hard_examples_outcome_focus=%.3f | hard_examples_weight_multiplier=%.3f | dedupe_sample_ids=%s | export_partial_every_n_files=%s | follow_source_updates=%s | source_poll_interval_seconds=%.1f | adaptive_source_polling=%s | global_target_backend=%s",
        dataset_id,
        source_dataset_id or "<auto>",
        str(cfg.get("input_sampled_dir", "")),
        build_mode_requested,
        num_workers,
        max_pending_futures,
        include_tactical_analysis,
        value_target_mix_teacher_weight,
        hard_examples_enabled,
        hard_examples_margin_threshold,
        hard_examples_outcome_focus,
        hard_examples_weight_multiplier,
        dedupe_sample_ids,
        export_partial_every_n_files,
        configured_follow_source_updates,
        source_poll_interval_seconds,
        adaptive_source_polling,
        str(global_progress_backend.get("backend", "file")),
    )
    global_firestore_diag = _firestore_backend_diagnostics(global_progress_backend)
    job.logger.info(
        "dataset build global target backend config | backend=%s | project_id=%s | collection=%s | document=%s | auth_mode=%s | credentials_path_exists=%s | api_key_set=%s",
        str(global_firestore_diag.get("backend", "file")),
        str(global_firestore_diag.get("project_id", "")) or "<empty>",
        str(global_firestore_diag.get("collection", "")) or "<empty>",
        str(global_firestore_diag.get("document", "")) or "<empty>",
        str(global_firestore_diag.get("auth_mode", "")) or "<unknown>",
        bool(global_firestore_diag.get("credentials_path_exists", False)),
        bool(global_firestore_diag.get("api_key_set", False)),
    )
    job.logger.info(
        "dataset build progressive global merge config | enabled=%s | target_dataset_id=%s | source_prefix=%s | include_partial=%s | every_n_partial_exports=%s | min_interval_seconds=%.1f | min_sources=%s | on_completion=%s | dedupe_sample_ids=%s",
        progressive_global_merge_enabled,
        progressive_global_merge_dataset_id or "<empty>",
        progressive_global_merge_source_dataset_id_prefix or "<none>",
        progressive_global_merge_include_partial,
        progressive_global_merge_every_n_partial_exports,
        progressive_global_merge_min_interval_seconds,
        progressive_global_merge_min_sources,
        progressive_global_merge_on_completion,
        progressive_global_merge_dedupe_sample_ids,
    )
    job.logger.info(
        "dataset build progressive global merge runtime tuning | async=%s | require_data_delta=%s | candidates_cache_ttl_seconds=%.1f | completion_wait_seconds=%.1f",
        progressive_global_merge_async,
        progressive_global_merge_require_data_delta,
        progressive_global_merge_candidates_cache_ttl_seconds,
        progressive_global_merge_completion_wait_seconds,
    )

    dataset_dir = job.job_dir / "dataset_build"
    source_mode_hint = ""
    if source_dataset_id:
        source_entry = None
        while source_entry is None:
            try:
                source_entry = _resolve_dataset_source(job, source_dataset_id)
            except FileNotFoundError:
                if not configured_follow_source_updates:
                    raise
                job.logger.info(
                    "dataset build waiting for source registration | source_dataset_id=%s | poll_interval_seconds=%.1f",
                    source_dataset_id,
                    current_poll_interval_seconds,
                )
                job.write_event(
                    "dataset_build_waiting_for_source_registration",
                    source_dataset_id=source_dataset_id,
                    poll_interval_seconds=current_poll_interval_seconds,
                )
                time.sleep(current_poll_interval_seconds)
                if adaptive_source_polling:
                    current_poll_interval_seconds = min(source_poll_interval_max_seconds, current_poll_interval_seconds * 1.5)
        source_mode_hint = str(source_entry.get("source_mode", "")).strip().lower() if isinstance(source_entry, dict) else ""
        sampled_root = Path(str(source_entry["sampled_dir"]))
    else:
        sampled_root = _resolve_storage_path(job.paths.drive_root, cfg.get("input_sampled_dir"), dataset_dir.parent / "dataset_generation" / "sampled_positions")
        source_dataset_id = Path(sampled_root).name
    if adaptive_source_polling:
        current_poll_interval_seconds = source_poll_interval_min_seconds
    follow_source_updates = bool(configured_follow_source_updates and source_dataset_id)
    configured_label_cache_dir = cfg.get("label_cache_dir")
    if configured_label_cache_dir:
        configured_label_cache_path = Path(str(configured_label_cache_dir))
        if configured_label_cache_path.name != dataset_id:
            job.logger.info(
                "dataset build label cache isolation enabled | dataset=%s | configured_label_cache_dir=%s | isolated_label_cache_dir=%s",
                dataset_id,
                configured_label_cache_path,
                job.paths.data_root / "label_cache" / dataset_id,
            )
            configured_label_cache_dir = None
    label_cache_dir = _resolve_storage_path(
        job.paths.drive_root,
        configured_label_cache_dir,
        job.paths.data_root / "label_cache" / dataset_id,
    )
    labeled_root = label_cache_dir / "labeled_positions"
    configured_output_dir = cfg.get("output_dir")
    if configured_output_dir:
        configured_output_path = Path(str(configured_output_dir))
        if configured_output_path.name != dataset_id:
            job.logger.info(
                "dataset build output isolation enabled | dataset=%s | configured_output_dir=%s | isolated_output_dir=%s",
                dataset_id,
                configured_output_path,
                job.paths.data_root / "datasets" / dataset_id,
            )
            configured_output_dir = None
    output_root = _resolve_storage_path(job.paths.drive_root, configured_output_dir, job.paths.data_root / "datasets" / dataset_id)
    labeled_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    target_labeled_samples = int(cfg.get("target_labeled_samples", 0))
    label_cache_metadata_path = label_cache_dir / "metadata.json"
    _write_json(
        label_cache_metadata_path,
        {
            "dataset_id": dataset_id,
            "source_dataset_id": source_dataset_id,
            "teacher_engine": teacher_engine,
            "teacher_level": teacher_level,
            "label_cache_dir": str(label_cache_dir),
        },
    )

    def _refresh_source_inventory() -> tuple[list[Path], str]:
        source_status = "completed"
        if source_dataset_id:
            try:
                source_entry_latest = _resolve_dataset_source(job, source_dataset_id)
                source_status = str(source_entry_latest.get("source_status", "completed")).strip().lower() or "completed"
            except FileNotFoundError:
                source_status = "completed"
        return sorted(sampled_root.rglob("*.jsonl")), source_status

    def _global_generation_target_reached_for_build() -> bool:
        if not stop_when_global_target_reached:
            return False
        payload = _read_global_generation_progress(
            global_target_progress_path,
            global_target_id=global_target_id,
            target_samples=int(global_target_samples_cfg),
            progress_backend=global_progress_backend,
            prefer_cache=False,
            telemetry=global_progress_read_telemetry,
        )
        total = int(payload.get("total_samples", 0))
        target = int(payload.get("target_samples") or global_target_samples_cfg or 0)
        return target > 0 and total >= target

    sampled_files, latest_source_status = _refresh_source_inventory()
    while not sampled_files and follow_source_updates and latest_source_status != "completed":
        job.logger.info(
            "dataset build waiting for first sampled files | source_dataset_id=%s | source_status=%s | poll_interval_seconds=%.1f",
            source_dataset_id,
            latest_source_status,
            current_poll_interval_seconds,
        )
        job.write_event(
            "dataset_build_waiting_for_first_sampled_files",
            source_dataset_id=source_dataset_id,
            source_status=latest_source_status,
            poll_interval_seconds=current_poll_interval_seconds,
        )
        if _global_generation_target_reached_for_build():
            global_target_reached_polls += 1
            if global_target_reached_polls >= global_target_stabilization_polls:
                break
        else:
            global_target_reached_polls = 0
        time.sleep(current_poll_interval_seconds)
        if adaptive_source_polling:
            current_poll_interval_seconds = min(source_poll_interval_max_seconds, current_poll_interval_seconds * 1.5)
        sampled_files, latest_source_status = _refresh_source_inventory()
    if sampled_files:
        global_target_reached_polls = 0
    if not sampled_files:
        if _global_generation_target_reached_for_build():
            summary = {
                "job_id": job.job_id,
                "dataset_id": dataset_id,
                "teacher_engine": teacher_engine,
                "teacher_level": teacher_level,
                "source_dataset_id": source_dataset_id,
                "label_cache_dir": str(label_cache_dir),
                "splits": {
                    "train": {"games": 0, "samples": 0, "duplicate_samples_removed": 0},
                    "validation": {"games": 0, "samples": 0, "duplicate_samples_removed": 0},
                    "test": {"games": 0, "samples": 0, "duplicate_samples_removed": 0},
                },
                "output_dir": str(output_root),
                "labeled_samples": 0,
                "target_labeled_samples": target_labeled_samples,
                "input_dim": 17,
                "feature_schema_version": "policy_value_tactical_v3",
                "build_mode": ("source_prelabeled" if build_mode_requested == "source_prelabeled" else "teacher_label"),
                "dedupe_sample_ids": dedupe_sample_ids,
                "duplicate_samples_removed": 0,
                "skipped_terminal_samples": 0,
                "skipped_no_legal_samples": 0,
                "skipped_invalid_samples": 0,
                "build_status": "completed_empty_global_target_reached",
            }
            _write_json(dataset_dir / "dataset_build_summary.json", summary)
            job.logger.info(
                "dataset build completed empty | dataset=%s | source_dataset_id=%s | reason=global_target_reached_before_first_source_file",
                dataset_id,
                source_dataset_id,
            )
            _write_global_progress_read_telemetry_metric()
            return summary
        raise FileNotFoundError(f"Aucun fichier sampled_positions trouve dans {sampled_root}")

    if build_mode_requested == "teacher_label":
        effective_build_mode = "teacher_label"
    elif build_mode_requested == "source_prelabeled":
        effective_build_mode = "source_prelabeled"
    else:
        if source_mode_hint in {"self_play_puct"}:
            effective_build_mode = "source_prelabeled"
        else:
            effective_build_mode = "teacher_label"
            for candidate_file in sampled_files[: min(8, len(sampled_files))]:
                if _detect_source_samples_are_prelabeled(candidate_file):
                    effective_build_mode = "source_prelabeled"
                    break
    job.logger.info(
        "dataset build mode resolved | requested=%s | effective=%s | source_mode_hint=%s",
        build_mode_requested,
        effective_build_mode,
        source_mode_hint or "<none>",
    )

    state = job.read_state()
    completed_files = set(state.get("completed_files", []))
    file_sample_counts: dict[str, int] = {}
    sampled_relative_names = [str(path.relative_to(sampled_root)) for path in sampled_files]
    known_relative_names = set(sampled_relative_names)
    contiguous_completed_prefix = 0
    for relative_name in sampled_relative_names:
        if relative_name not in completed_files:
            break
        if not (labeled_root / relative_name).exists():
            break
        contiguous_completed_prefix += 1

    processed_count = contiguous_completed_prefix
    labeled_samples_total = int(state.get("labeled_samples", 0)) if contiguous_completed_prefix > 0 else 0
    skipped_terminal_samples = 0
    skipped_no_legal_samples = 0
    skipped_invalid_samples = 0
    log_every_n_files = max(1, int(cfg.get("log_every_n_files", 1)))
    last_logged_progress_count = -1
    build_started_monotonic = time.monotonic()
    last_partial_export_processed = 0
    progressive_global_merge_exports_count = max(
        0,
        _safe_int(state.get("progressive_global_merge_exports_count", 0), 0),
    )
    last_progressive_global_merge_monotonic = 0.0
    last_progressive_global_merge_signature = str(state.get("progressive_global_merge_last_signature", "")).strip()
    last_progressive_global_merge_source_labeled_samples = max(
        0,
        _safe_int(state.get("progressive_global_merge_last_source_labeled_samples", 0), 0),
    )
    last_progressive_global_merge_last_export_labeled_samples = max(
        0,
        _safe_int(state.get("progressive_global_merge_last_export_labeled_samples", labeled_samples_total), labeled_samples_total),
    )
    progressive_global_merge_executor: concurrent.futures.ThreadPoolExecutor | None = None
    progressive_global_merge_future: concurrent.futures.Future[dict[str, Any]] | None = None
    progressive_global_merge_future_context: dict[str, Any] | None = None
    if progressive_global_merge_enabled and progressive_global_merge_async:
        progressive_global_merge_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dataset-build-merge",
        )
    progressive_global_merge_candidates_cache_initialized = False
    progressive_global_merge_candidates_cache_next_refresh_monotonic = 0.0
    progressive_global_merge_candidates_cache_registry_signature = ""
    progressive_global_merge_candidates_cache_source_signature = ""
    progressive_global_merge_candidates_cache_source_labeled_total = 0
    progressive_global_merge_candidates_cache_source_ids: list[str] = []
    last_completed_file_name = sampled_relative_names[contiguous_completed_prefix - 1] if contiguous_completed_prefix > 0 else ""

    def _estimate_remaining_seconds() -> float | None:
        if processed_count <= 0:
            return None
        elapsed = max(0.001, time.monotonic() - build_started_monotonic)
        files_per_second = processed_count / elapsed
        if files_per_second <= 0:
            return None
        remaining_files = len(sampled_files) - processed_count
        return remaining_files / files_per_second

    def _throughput_metrics() -> tuple[float | None, float | None]:
        if processed_count <= 0:
            return None, None
        elapsed = max(0.001, time.monotonic() - build_started_monotonic)
        files_per_second = processed_count / elapsed
        samples_per_second = labeled_samples_total / elapsed
        return files_per_second, samples_per_second

    def _write_build_state() -> None:
        job.write_state(
            {
                "completed_files": sorted(completed_files),
                "processed_files": processed_count,
                "remaining_files": len(sampled_files) - processed_count,
                "labeled_samples": labeled_samples_total,
                "skipped_terminal_samples": skipped_terminal_samples,
                "skipped_no_legal_samples": skipped_no_legal_samples,
                "skipped_invalid_samples": skipped_invalid_samples,
                "target_labeled_samples": target_labeled_samples,
                "include_tactical_analysis": include_tactical_analysis,
                "contiguous_completed_prefix": contiguous_completed_prefix,
                "follow_source_updates": follow_source_updates,
                "source_poll_interval_seconds": current_poll_interval_seconds,
                "last_completed_file": last_completed_file_name,
                "progressive_global_merge_last_signature": last_progressive_global_merge_signature,
                "progressive_global_merge_last_source_labeled_samples": last_progressive_global_merge_source_labeled_samples,
                "progressive_global_merge_exports_count": progressive_global_merge_exports_count,
                "progressive_global_merge_last_export_labeled_samples": last_progressive_global_merge_last_export_labeled_samples,
            }
        )

    def _hash_progressive_global_merge_payload(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def _collect_progressive_global_merge_source_dataset_snapshot(
        *,
        force_refresh: bool = False,
    ) -> tuple[list[str], str, int]:
        nonlocal progressive_global_merge_candidates_cache_initialized
        nonlocal progressive_global_merge_candidates_cache_next_refresh_monotonic
        nonlocal progressive_global_merge_candidates_cache_registry_signature
        nonlocal progressive_global_merge_candidates_cache_source_signature
        nonlocal progressive_global_merge_candidates_cache_source_labeled_total
        nonlocal progressive_global_merge_candidates_cache_source_ids
        if not progressive_global_merge_enabled:
            return [], "", 0
        now_monotonic = time.monotonic()
        if (
            progressive_global_merge_candidates_cache_initialized
            and not force_refresh
            and now_monotonic < progressive_global_merge_candidates_cache_next_refresh_monotonic
        ):
            return (
                list(progressive_global_merge_candidates_cache_source_ids),
                str(progressive_global_merge_candidates_cache_source_signature),
                int(progressive_global_merge_candidates_cache_source_labeled_total),
            )
        registry = _read_dataset_registry(job)
        built_entries = registry.get("built_datasets", [])
        if not isinstance(built_entries, list):
            built_entries = []
        registry_signature_items: list[tuple[str, int, str, str]] = []
        for entry in built_entries:
            if not isinstance(entry, dict):
                continue
            candidate_id = str(entry.get("dataset_id", "")).strip()
            if not candidate_id:
                continue
            registry_signature_items.append(
                (
                    candidate_id,
                    max(0, _safe_int(entry.get("labeled_samples", 0), 0)),
                    str(entry.get("build_status", "")).strip().lower(),
                    str(entry.get("output_dir", "")).strip(),
                )
            )
        registry_signature_items.sort(key=lambda item: item[0])
        registry_signature = _hash_progressive_global_merge_payload(registry_signature_items)
        if (
            progressive_global_merge_candidates_cache_initialized
            and registry_signature == progressive_global_merge_candidates_cache_registry_signature
        ):
            progressive_global_merge_candidates_cache_next_refresh_monotonic = (
                now_monotonic + progressive_global_merge_candidates_cache_ttl_seconds
            )
            return (
                list(progressive_global_merge_candidates_cache_source_ids),
                str(progressive_global_merge_candidates_cache_source_signature),
                int(progressive_global_merge_candidates_cache_source_labeled_total),
            )

        candidates: list[str] = []
        seen: set[str] = set()
        source_signature_items: list[tuple[str, int, str]] = []
        source_labeled_total = 0
        for entry in built_entries:
            if not isinstance(entry, dict):
                continue
            candidate_id = str(entry.get("dataset_id", "")).strip()
            if not candidate_id or candidate_id in seen:
                continue
            if candidate_id == progressive_global_merge_dataset_id:
                continue
            if (
                progressive_global_merge_source_dataset_id_prefix
                and not candidate_id.startswith(progressive_global_merge_source_dataset_id_prefix)
            ):
                continue
            build_status = str(entry.get("build_status", "")).strip().lower()
            if not progressive_global_merge_include_partial and build_status != "completed":
                continue
            output_dir_text = str(entry.get("output_dir", "")).strip()
            if not output_dir_text:
                continue
            output_dir = Path(output_dir_text)
            if not (
                (output_dir / "train.npz").exists()
                and (output_dir / "validation.npz").exists()
                and (output_dir / "test.npz").exists()
            ):
                continue
            labeled_samples = max(0, _safe_int(entry.get("labeled_samples", 0), 0))
            seen.add(candidate_id)
            candidates.append(candidate_id)
            source_labeled_total += labeled_samples
            source_signature_items.append((candidate_id, labeled_samples, build_status))

        source_signature_items.sort(key=lambda item: item[0])
        source_signature = _hash_progressive_global_merge_payload(source_signature_items)
        progressive_global_merge_candidates_cache_initialized = True
        progressive_global_merge_candidates_cache_registry_signature = registry_signature
        progressive_global_merge_candidates_cache_source_ids = list(candidates)
        progressive_global_merge_candidates_cache_source_signature = source_signature
        progressive_global_merge_candidates_cache_source_labeled_total = int(source_labeled_total)
        progressive_global_merge_candidates_cache_next_refresh_monotonic = (
            now_monotonic + progressive_global_merge_candidates_cache_ttl_seconds
        )
        return candidates, source_signature, source_labeled_total

    def _run_progressive_global_merge_once(
        *,
        merge_cfg: dict[str, Any],
        trigger: str,
        source_dataset_ids_for_merge: list[str],
        source_signature: str,
        source_labeled_total: int,
    ) -> dict[str, Any]:
        started_monotonic = time.monotonic()
        merge_summary = run_dataset_merge_final(job, cfg_override=merge_cfg)
        return {
            "trigger": trigger,
            "target_dataset_id": progressive_global_merge_dataset_id,
            "source_dataset_count": len(source_dataset_ids_for_merge),
            "source_signature": source_signature,
            "source_labeled_total": int(source_labeled_total),
            "merge_summary": merge_summary if isinstance(merge_summary, dict) else {},
            "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
        }

    def _emit_progressive_global_merge_completed(result: dict[str, Any], *, async_mode: bool) -> None:
        nonlocal last_progressive_global_merge_monotonic
        nonlocal last_progressive_global_merge_signature
        nonlocal last_progressive_global_merge_source_labeled_samples
        merged_summary = result.get("merge_summary", {}) if isinstance(result, dict) else {}
        merged_labeled_samples = _safe_int(
            merged_summary.get("labeled_samples", 0) if isinstance(merged_summary, dict) else 0,
            0,
        )
        last_progressive_global_merge_monotonic = time.monotonic()
        last_progressive_global_merge_signature = str(result.get("source_signature", "")).strip()
        last_progressive_global_merge_source_labeled_samples = max(
            0,
            _safe_int(result.get("source_labeled_total", 0), 0),
        )
        job.logger.info(
            "dataset build progressive global merge completed | trigger=%s | target_dataset_id=%s | source_datasets=%s | source_labeled_total=%s | labeled_samples=%s | async=%s | duration_seconds=%.2f",
            str(result.get("trigger", "<none>")),
            str(result.get("target_dataset_id", progressive_global_merge_dataset_id)),
            _safe_int(result.get("source_dataset_count", 0), 0),
            _safe_int(result.get("source_labeled_total", 0), 0),
            merged_labeled_samples,
            async_mode,
            float(result.get("duration_seconds", 0.0) or 0.0),
        )
        job.write_event(
            "dataset_build_progressive_global_merge_completed",
            trigger=str(result.get("trigger", "")),
            target_dataset_id=str(result.get("target_dataset_id", progressive_global_merge_dataset_id)),
            source_dataset_count=_safe_int(result.get("source_dataset_count", 0), 0),
            source_labeled_total=_safe_int(result.get("source_labeled_total", 0), 0),
            source_signature=str(result.get("source_signature", "")),
            labeled_samples=merged_labeled_samples,
            async_mode=bool(async_mode),
            duration_seconds=float(result.get("duration_seconds", 0.0) or 0.0),
        )
        _write_build_state()

    def _emit_progressive_global_merge_failed(
        *,
        trigger: str,
        source_dataset_count: int,
        source_labeled_total: int,
        source_signature: str,
        exc: Exception,
        async_mode: bool,
    ) -> None:
        job.logger.warning(
            "dataset build progressive global merge failed | trigger=%s | target_dataset_id=%s | source_datasets=%s | source_labeled_total=%s | async=%s | error=%s",
            trigger,
            progressive_global_merge_dataset_id,
            source_dataset_count,
            source_labeled_total,
            async_mode,
            exc,
        )
        job.write_event(
            "dataset_build_progressive_global_merge_failed",
            trigger=trigger,
            target_dataset_id=progressive_global_merge_dataset_id,
            source_dataset_count=source_dataset_count,
            source_labeled_total=source_labeled_total,
            source_signature=source_signature,
            async_mode=bool(async_mode),
            error=f"{type(exc).__name__}: {exc}",
        )

    def _drain_progressive_global_merge_future(
        *,
        block: bool = False,
        timeout_seconds: float | None = None,
    ) -> bool:
        nonlocal progressive_global_merge_future, progressive_global_merge_future_context
        if progressive_global_merge_future is None:
            return True
        if not block and not progressive_global_merge_future.done():
            return False
        try:
            if block:
                result = progressive_global_merge_future.result(timeout=timeout_seconds)
            else:
                result = progressive_global_merge_future.result()
        except concurrent.futures.TimeoutError:
            return False
        except Exception as exc:
            context = progressive_global_merge_future_context or {}
            _emit_progressive_global_merge_failed(
                trigger=str(context.get("trigger", "partial_snapshot")),
                source_dataset_count=_safe_int(context.get("source_dataset_count", 0), 0),
                source_labeled_total=_safe_int(context.get("source_labeled_total", 0), 0),
                source_signature=str(context.get("source_signature", "")),
                exc=exc,
                async_mode=True,
            )
        else:
            if not isinstance(result, dict):
                result = {}
            _emit_progressive_global_merge_completed(result, async_mode=True)
        finally:
            progressive_global_merge_future = None
            progressive_global_merge_future_context = None
        return True

    def _maybe_progressive_global_merge(*, force: bool = False, trigger: str = "partial_snapshot") -> None:
        nonlocal last_progressive_global_merge_monotonic
        nonlocal progressive_global_merge_future
        nonlocal progressive_global_merge_future_context
        if not progressive_global_merge_enabled:
            return
        _drain_progressive_global_merge_future(block=False)
        now_monotonic = time.monotonic()
        if not force:
            if (
                progressive_global_merge_exports_count % progressive_global_merge_every_n_partial_exports
            ) != 0:
                return
            if (
                progressive_global_merge_min_interval_seconds > 0
                and (now_monotonic - last_progressive_global_merge_monotonic)
                < progressive_global_merge_min_interval_seconds
            ):
                return
        source_dataset_ids_for_merge, source_signature, source_labeled_total = _collect_progressive_global_merge_source_dataset_snapshot()
        if len(source_dataset_ids_for_merge) < progressive_global_merge_min_sources:
            job.logger.info(
                "dataset build progressive global merge skipped | trigger=%s | reason=insufficient_sources | sources=%s | min_sources=%s",
                trigger,
                len(source_dataset_ids_for_merge),
                progressive_global_merge_min_sources,
            )
            return
        if (
            progressive_global_merge_require_data_delta
            and source_signature
            and source_signature == last_progressive_global_merge_signature
        ):
            job.logger.info(
                "dataset build progressive global merge skipped | trigger=%s | reason=no_source_delta | source_signature=%s | source_labeled_total=%s",
                trigger,
                source_signature,
                source_labeled_total,
            )
            return
        merge_cfg = {
            "dataset_id": progressive_global_merge_dataset_id,
            "source_dataset_ids": source_dataset_ids_for_merge,
            "include_all_built": False,
            "dedupe_sample_ids": progressive_global_merge_dedupe_sample_ids,
            "merge_lock_ttl_seconds": progressive_global_merge_lock_ttl_seconds,
            "merge_lock_wait_seconds": progressive_global_merge_lock_wait_seconds,
        }
        if progressive_global_merge_executor is not None and not force:
            if progressive_global_merge_future is not None and not progressive_global_merge_future.done():
                job.logger.info(
                    "dataset build progressive global merge deferred | trigger=%s | reason=merge_inflight",
                    trigger,
                )
                return
            progressive_global_merge_future_context = {
                "trigger": trigger,
                "source_dataset_count": len(source_dataset_ids_for_merge),
                "source_labeled_total": int(source_labeled_total),
                "source_signature": source_signature,
            }
            progressive_global_merge_future = progressive_global_merge_executor.submit(
                _run_progressive_global_merge_once,
                merge_cfg=merge_cfg,
                trigger=trigger,
                source_dataset_ids_for_merge=list(source_dataset_ids_for_merge),
                source_signature=source_signature,
                source_labeled_total=int(source_labeled_total),
            )
            job.write_event(
                "dataset_build_progressive_global_merge_queued",
                trigger=trigger,
                target_dataset_id=progressive_global_merge_dataset_id,
                source_dataset_count=len(source_dataset_ids_for_merge),
                source_labeled_total=int(source_labeled_total),
                source_signature=source_signature,
                async_mode=True,
            )
            return
        if progressive_global_merge_executor is not None and force:
            drained = _drain_progressive_global_merge_future(
                block=True,
                timeout_seconds=progressive_global_merge_completion_wait_seconds,
            )
            if not drained:
                job.logger.warning(
                    "dataset build progressive global merge skipped | trigger=%s | reason=merge_inflight_timeout | wait_seconds=%.1f",
                    trigger,
                    progressive_global_merge_completion_wait_seconds,
                )
                return
        try:
            result = _run_progressive_global_merge_once(
                merge_cfg=merge_cfg,
                trigger=trigger,
                source_dataset_ids_for_merge=list(source_dataset_ids_for_merge),
                source_signature=source_signature,
                source_labeled_total=int(source_labeled_total),
            )
        except Exception as exc:
            _emit_progressive_global_merge_failed(
                trigger=trigger,
                source_dataset_count=len(source_dataset_ids_for_merge),
                source_labeled_total=int(source_labeled_total),
                source_signature=source_signature,
                exc=exc,
                async_mode=False,
            )
            return
        _emit_progressive_global_merge_completed(result, async_mode=False)

    def _log_build_progress_if_needed() -> None:
        nonlocal last_logged_progress_count
        if processed_count == last_logged_progress_count:
            return
        if processed_count % log_every_n_files != 0 and processed_count != len(sampled_files):
            return
        files_per_second, samples_per_second = _throughput_metrics()
        job.logger.info(
            "dataset build progress | files=%s/%s | labeled_samples=%s | skipped_terminal=%s | skipped_no_legal=%s | skipped_invalid=%s | files_per_sec=%.2f | samples_per_sec=%.2f | eta=%s",
            processed_count,
            len(sampled_files),
            labeled_samples_total,
            skipped_terminal_samples,
            skipped_no_legal_samples,
            skipped_invalid_samples,
            files_per_second or 0.0,
            samples_per_second or 0.0,
            _format_eta_seconds(_estimate_remaining_seconds()),
        )
        last_logged_progress_count = processed_count

    def _target_reached() -> bool:
        return target_labeled_samples > 0 and labeled_samples_total >= target_labeled_samples

    def _export_partial_snapshot_if_needed(*, force: bool = False) -> None:
        nonlocal last_partial_export_processed, progressive_global_merge_exports_count
        nonlocal last_progressive_global_merge_last_export_labeled_samples
        if processed_count <= 0:
            return
        if not force and (processed_count - last_partial_export_processed) < export_partial_every_n_files:
            return
        split_summary, exported_input_dim, duplicate_samples_removed = _export_built_dataset_snapshot(
            job=job,
            dataset_id=dataset_id,
            source_dataset_id=source_dataset_id,
            source_dataset_ids=[source_dataset_id] if source_dataset_id else [],
            sampled_root=sampled_root,
            labeled_root=labeled_root,
            output_root=output_root,
            label_cache_dir=label_cache_dir,
            teacher_engine=teacher_engine,
            teacher_level=teacher_level,
            sampled_relative_names=sampled_relative_names,
            completed_files=completed_files,
            file_sample_counts=file_sample_counts,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            labeled_samples_total=labeled_samples_total,
            target_labeled_samples=target_labeled_samples,
            build_status="partial",
            build_mode=effective_build_mode,
            dedupe_sample_ids=dedupe_sample_ids,
        )
        last_partial_export_processed = processed_count
        job.logger.info(
            "dataset build partial snapshot exported | dataset=%s | processed_files=%s/%s | labeled_samples=%s | duplicate_samples_removed=%s | input_dim=%s | train=%s | validation=%s | test=%s | output_dir=%s",
            dataset_id,
            processed_count,
            len(sampled_files),
            labeled_samples_total,
            duplicate_samples_removed,
            exported_input_dim,
            split_summary.get("train", {}).get("samples", 0),
            split_summary.get("validation", {}).get("samples", 0),
            split_summary.get("test", {}).get("samples", 0),
            output_root,
        )
        job.write_event(
            "dataset_build_partial_snapshot_exported",
            dataset_id=dataset_id,
            processed_files=processed_count,
            total_files=len(sampled_files),
            labeled_samples=labeled_samples_total,
            output_dir=str(output_root),
            split_summary=split_summary,
            duplicate_samples_removed=duplicate_samples_removed,
            dedupe_sample_ids=dedupe_sample_ids,
            input_dim=exported_input_dim,
        )
        if labeled_samples_total > last_progressive_global_merge_last_export_labeled_samples:
            progressive_global_merge_exports_count += 1
            last_progressive_global_merge_last_export_labeled_samples = labeled_samples_total
        _maybe_progressive_global_merge(force=False, trigger="partial_snapshot")

    job.logger.info(
        "dataset build started | dataset=%s | source_dataset_id=%s | build_mode=%s | teacher=%s:%s | sampled_root=%s | label_cache=%s | output_dir=%s | files=%s | target_labeled_samples=%s | workers=%s | include_tactical_analysis=%s | dedupe_sample_ids=%s",
        dataset_id,
        source_dataset_id,
        effective_build_mode,
        teacher_engine,
        teacher_level,
        sampled_root,
        label_cache_dir,
        output_root,
        len(sampled_files),
        target_labeled_samples,
        num_workers,
        include_tactical_analysis,
        dedupe_sample_ids,
    )
    job.set_phase("dataset_build")
    job.write_event(
        "dataset_build_started",
        dataset_id=dataset_id,
        source_dataset_id=source_dataset_id,
        build_mode_requested=build_mode_requested,
        build_mode=effective_build_mode,
        teacher_engine=teacher_engine,
        teacher_level=teacher_level,
        sampled_root=str(sampled_root),
        label_cache_dir=str(label_cache_dir),
        sampled_files=len(sampled_files),
        target_labeled_samples=target_labeled_samples,
        num_workers=num_workers,
        include_tactical_analysis=include_tactical_analysis,
        value_target_mix_teacher_weight=value_target_mix_teacher_weight,
        hard_examples_enabled=hard_examples_enabled,
        hard_examples_margin_threshold=hard_examples_margin_threshold,
        hard_examples_outcome_focus=hard_examples_outcome_focus,
        hard_examples_weight_multiplier=hard_examples_weight_multiplier,
        dedupe_sample_ids=dedupe_sample_ids,
        adaptive_source_polling=adaptive_source_polling,
    )
    job.write_metric({"metric_type": "dataset_build_started"})

    if contiguous_completed_prefix > 0:
        job.logger.info(
            "dataset build resume shortcut | contiguous_completed_prefix=%s | skipped_rescan_files=%s | labeled_samples=%s",
            contiguous_completed_prefix,
            contiguous_completed_prefix,
            labeled_samples_total,
        )
        job.write_event(
            "dataset_build_resume_shortcut",
            contiguous_completed_prefix=contiguous_completed_prefix,
            skipped_rescan_files=contiguous_completed_prefix,
            labeled_samples=labeled_samples_total,
        )

    pending_files: list[dict[str, Any]] = []
    for file_index, sampled_file in enumerate(sampled_files[contiguous_completed_prefix:], start=contiguous_completed_prefix + 1):
        relative_name = str(sampled_file.relative_to(sampled_root))
        output_labeled = labeled_root / relative_name
        output_labeled.parent.mkdir(parents=True, exist_ok=True)

        if output_labeled.exists():
            source_count = _count_jsonl_lines(output_labeled)
            job.logger.info(
                "dataset build file reused | %s/%s | file=%s | labeled_samples=%s",
                file_index,
                len(sampled_files),
                relative_name,
                source_count,
            )
            completed_files.add(relative_name)
            file_sample_counts[relative_name] = source_count
            processed_count += 1
            labeled_samples_total += source_count
            last_completed_file_name = relative_name
        else:
            pending_files.append(
                {
                    "file_index": file_index,
                    "relative_name": relative_name,
                    "sampled_file": sampled_file,
                    "output_labeled": output_labeled,
                }
            )

        _log_build_progress_if_needed()
        _write_build_state()
        if _target_reached():
            job.logger.info(
                "dataset build target reached during scan | labeled_samples=%s | target_labeled_samples=%s | processed_files=%s",
                labeled_samples_total,
                target_labeled_samples,
                processed_count,
            )
            break

    reused_count = processed_count
    pending_count = len(pending_files)
    files_per_second, samples_per_second = _throughput_metrics()
    job.logger.info(
        "dataset build scan completed | reused_files=%s | pending_files=%s | total_files=%s | labeled_samples=%s | files_per_sec=%.2f | samples_per_sec=%.2f | eta=%s",
        reused_count,
        pending_count,
        len(sampled_files),
        labeled_samples_total,
        files_per_second or 0.0,
        samples_per_second or 0.0,
        _format_eta_seconds(_estimate_remaining_seconds()),
    )
    job.write_event(
        "dataset_build_scan_completed",
        reused_files=reused_count,
        pending_files=pending_count,
        total_files=len(sampled_files),
        labeled_samples=labeled_samples_total,
        source_status=latest_source_status,
    )

    def _extract_samples_from_file(
        sampled_file: Path,
    ) -> tuple[int, list[dict[str, Any]], int, int, int]:
        if effective_build_mode == "source_prelabeled":
            source_count, file_samples, skipped_terminal, skipped_no_legal, skipped_invalid = _prepare_prelabeled_samples_from_file(
                str(sampled_file),
                include_tactical_analysis=include_tactical_analysis,
                hard_examples_enabled=hard_examples_enabled,
                hard_examples_outcome_focus=hard_examples_outcome_focus,
                hard_examples_weight_multiplier=hard_examples_weight_multiplier,
            )
            return source_count, file_samples, skipped_terminal, skipped_no_legal, skipped_invalid
        source_count, file_samples, skipped_terminal, skipped_no_legal = _label_samples_from_file(
            str(sampled_file),
            teacher_engine=teacher_engine,
            teacher_level=teacher_level,
            include_tactical_analysis=include_tactical_analysis,
            value_target_mix_teacher_weight=value_target_mix_teacher_weight,
            hard_examples_enabled=hard_examples_enabled,
            hard_examples_margin_threshold=hard_examples_margin_threshold,
            hard_examples_outcome_focus=hard_examples_outcome_focus,
            hard_examples_weight_multiplier=hard_examples_weight_multiplier,
        )
        return source_count, file_samples, skipped_terminal, skipped_no_legal, 0

    def _materialize_labeled_file(
        file_item: dict[str, Any],
        source_count: int,
        file_samples: list[dict[str, Any]],
        skipped_terminal: int,
        skipped_no_legal: int,
        skipped_invalid: int,
        *,
        mode: str,
    ) -> None:
        nonlocal processed_count, skipped_terminal_samples, skipped_no_legal_samples, skipped_invalid_samples, labeled_samples_total, last_completed_file_name
        output_labeled = Path(file_item["output_labeled"])
        output_labeled.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(output_labeled, file_samples, ensure_ascii=True)

        skipped_terminal_samples += skipped_terminal
        skipped_no_legal_samples += skipped_no_legal
        skipped_invalid_samples += skipped_invalid
        completed_files.add(str(file_item["relative_name"]))
        file_sample_counts[str(file_item["relative_name"])] = len(file_samples)
        processed_count += 1
        labeled_samples_total += len(file_samples)
        last_completed_file_name = str(file_item["relative_name"])
        job.logger.info(
            "dataset build file completed | %s/%s | file=%s | mode=%s | source_samples=%s | labeled_samples=%s | skipped_terminal=%s | skipped_no_legal=%s | skipped_invalid=%s",
            file_item["file_index"],
            len(sampled_files),
            file_item["relative_name"],
            mode,
            source_count,
            len(file_samples),
            skipped_terminal_samples,
            skipped_no_legal_samples,
            skipped_invalid_samples,
        )
        job.write_event("dataset_labeled_file_completed", file=str(file_item["relative_name"]), samples=len(file_samples))
        job.write_metric({"metric_type": "dataset_labeled_file_completed", "file": str(file_item["relative_name"]), "samples": len(file_samples)})
        _log_build_progress_if_needed()
        _write_build_state()
        _export_partial_snapshot_if_needed()

    def _process_pending_files_batch(pending_batch: list[dict[str, Any]]) -> None:
        if not pending_batch or _target_reached():
            return
        def _handle_missing_source_file(
            file_item: dict[str, Any],
            *,
            mode: str,
            pending_queue: list[dict[str, Any]] | None,
        ) -> None:
            relative_name = str(file_item.get("relative_name", ""))
            sampled_file_path = Path(file_item.get("sampled_file", ""))
            retry_count = int(file_item.get("__missing_source_retries", 0) or 0) + 1
            file_item["__missing_source_retries"] = retry_count
            job.logger.warning(
                "dataset build source file missing | file=%s | mode=%s | retry=%s/%s | path=%s",
                relative_name,
                mode,
                retry_count,
                missing_source_file_retry_attempts,
                sampled_file_path,
            )
            job.write_event(
                "dataset_build_source_file_missing",
                file=relative_name,
                mode=mode,
                retry_count=retry_count,
                retry_limit=missing_source_file_retry_attempts,
                sampled_file_path=str(sampled_file_path),
            )
            if retry_count < missing_source_file_retry_attempts:
                if pending_queue is not None:
                    pending_queue.append(file_item)
                if missing_source_file_retry_delay_seconds > 0.0:
                    time.sleep(missing_source_file_retry_delay_seconds)
                return
            if follow_source_updates:
                known_relative_names.discard(relative_name)
                job.logger.warning(
                    "dataset build source file deferred after retries | file=%s | mode=%s | action=wait_next_refresh",
                    relative_name,
                    mode,
                )
                job.write_event(
                    "dataset_build_source_file_deferred",
                    file=relative_name,
                    mode=mode,
                    action="wait_next_refresh",
                )
                return
            raise FileNotFoundError(f"Fichier source introuvable apres retries: {sampled_file_path}")
        if num_workers <= 1:
            pending_queue = list(pending_batch)
            while pending_queue:
                file_item = pending_queue.pop(0)
                job.logger.info(
                    "dataset build file started | %s/%s | file=%s | mode=sequential",
                    file_item["file_index"],
                    len(sampled_files),
                    file_item["relative_name"],
                )
                try:
                    source_count, file_samples, skipped_terminal, skipped_no_legal, skipped_invalid = _extract_samples_from_file(
                        Path(file_item["sampled_file"])
                    )
                except FileNotFoundError:
                    _handle_missing_source_file(
                        file_item,
                        mode="sequential",
                        pending_queue=pending_queue,
                    )
                    continue
                _materialize_labeled_file(
                    file_item,
                    source_count,
                    file_samples,
                    skipped_terminal,
                    skipped_no_legal,
                    skipped_invalid,
                    mode="sequential",
                )
                if _target_reached():
                    job.logger.info(
                        "dataset build target reached | labeled_samples=%s | target_labeled_samples=%s | processed_files=%s",
                        labeled_samples_total,
                        target_labeled_samples,
                        processed_count,
                    )
                    break
        else:
            job.logger.info(
                "dataset build parallel execution | workers=%s | pending_files=%s | max_pending=%s | start_method=%s | max_tasks_per_child=%s",
                num_workers,
                len(pending_batch),
                max_pending_futures,
                multiprocessing_start_method,
                effective_max_tasks_per_child,
            )
            job.write_event(
                "dataset_build_parallel_execution_started",
                workers=num_workers,
                pending_files=len(pending_batch),
                max_pending_futures=max_pending_futures,
                multiprocessing_start_method=multiprocessing_start_method,
                max_tasks_per_child=effective_max_tasks_per_child,
            )
            future_map: dict[concurrent.futures.Future, dict[str, Any]] = {}
            pending_queue = list(pending_batch)
            target_reached = False
            if effective_build_mode == "source_prelabeled":
                worker_function: Callable[..., Any] = _prepare_prelabeled_samples_from_file
                worker_kwargs = {
                    "include_tactical_analysis": include_tactical_analysis,
                    "hard_examples_enabled": hard_examples_enabled,
                    "hard_examples_outcome_focus": hard_examples_outcome_focus,
                    "hard_examples_weight_multiplier": hard_examples_weight_multiplier,
                }
            else:
                worker_function = _label_samples_from_file
                worker_kwargs = {
                    "teacher_engine": teacher_engine,
                    "teacher_level": teacher_level,
                    "include_tactical_analysis": include_tactical_analysis,
                    "value_target_mix_teacher_weight": value_target_mix_teacher_weight,
                    "hard_examples_enabled": hard_examples_enabled,
                    "hard_examples_margin_threshold": hard_examples_margin_threshold,
                    "hard_examples_outcome_focus": hard_examples_outcome_focus,
                    "hard_examples_weight_multiplier": hard_examples_weight_multiplier,
                }
            try:
                mp_context = multiprocessing.get_context(multiprocessing_start_method)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=mp_context,
                    max_tasks_per_child=effective_max_tasks_per_child,
                ) as executor:
                    while pending_queue or future_map:
                        while pending_queue and len(future_map) < max_pending_futures and not target_reached:
                            file_item = pending_queue.pop(0)
                            job.logger.info(
                                "dataset build file started | %s/%s | file=%s | mode=parallel",
                                file_item["file_index"],
                                len(sampled_files),
                                file_item["relative_name"],
                            )
                            future = executor.submit(
                                worker_function,
                                str(file_item["sampled_file"]),
                                **worker_kwargs,
                            )
                            future_map[future] = file_item

                        if not future_map:
                            break
                        done, _not_done = concurrent.futures.wait(
                            future_map.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            file_item = future_map.pop(future)
                            try:
                                result_payload = future.result()
                            except FileNotFoundError:
                                _handle_missing_source_file(
                                    file_item,
                                    mode="parallel",
                                    pending_queue=pending_queue,
                                )
                                continue
                            if effective_build_mode == "source_prelabeled":
                                source_count, file_samples, skipped_terminal, skipped_no_legal, skipped_invalid = result_payload
                            else:
                                source_count, file_samples, skipped_terminal, skipped_no_legal = result_payload
                                skipped_invalid = 0
                            _materialize_labeled_file(
                                file_item,
                                source_count,
                                file_samples,
                                skipped_terminal,
                                skipped_no_legal,
                                skipped_invalid,
                                mode="parallel",
                            )
                            if _target_reached() and not target_reached:
                                target_reached = True
                                pending_queue.clear()
                                job.logger.info(
                                    "dataset build target reached | labeled_samples=%s | target_labeled_samples=%s | processed_files=%s | allowing_inflight_workers_to_finish=%s",
                                    labeled_samples_total,
                                    target_labeled_samples,
                                    processed_count,
                                    len(future_map),
                                )
            except concurrent.futures.process.BrokenProcessPool as exc:
                failed_pending = list(future_map.values()) + list(pending_queue)
                job.logger.warning(
                    "dataset build parallel pool broken | completed_files=%s | remaining_fallback=%s | error=%s",
                    processed_count,
                    len(failed_pending),
                    exc,
                )
                job.write_event(
                    "dataset_build_parallel_execution_broken",
                    completed_files=processed_count,
                    remaining_fallback=len(failed_pending),
                    error=str(exc),
                )
                fallback_queue = list(failed_pending)
                while fallback_queue:
                    file_item = fallback_queue.pop(0)
                    job.logger.info(
                        "dataset build file started | %s/%s | file=%s | mode=sequential_fallback",
                        file_item["file_index"],
                        len(sampled_files),
                        file_item["relative_name"],
                    )
                    try:
                        source_count, file_samples, skipped_terminal, skipped_no_legal, skipped_invalid = _extract_samples_from_file(
                            Path(file_item["sampled_file"])
                        )
                    except FileNotFoundError:
                        _handle_missing_source_file(
                            file_item,
                            mode="sequential_fallback",
                            pending_queue=fallback_queue,
                        )
                        continue
                    _materialize_labeled_file(
                        file_item,
                        source_count,
                        file_samples,
                        skipped_terminal,
                        skipped_no_legal,
                        skipped_invalid,
                        mode="sequential_fallback",
                    )
                    if _target_reached():
                        job.logger.info(
                            "dataset build target reached | labeled_samples=%s | target_labeled_samples=%s | processed_files=%s",
                            labeled_samples_total,
                            target_labeled_samples,
                            processed_count,
                        )
                        break

    _process_pending_files_batch(pending_files)

    def _discover_new_sampled_files() -> tuple[list[dict[str, Any]], int, str]:
        nonlocal sampled_files, sampled_relative_names, processed_count, labeled_samples_total, last_completed_file_name, latest_source_status
        new_pending_files: list[dict[str, Any]] = []
        reused_new_files = 0
        sampled_files, latest_source_status = _refresh_source_inventory()
        sampled_relative_names = [str(path.relative_to(sampled_root)) for path in sampled_files]
        for file_index, sampled_file in enumerate(sampled_files, start=1):
            relative_name = str(sampled_file.relative_to(sampled_root))
            if relative_name in known_relative_names:
                continue
            known_relative_names.add(relative_name)
            output_labeled = labeled_root / relative_name
            output_labeled.parent.mkdir(parents=True, exist_ok=True)
            if output_labeled.exists():
                source_count = _count_jsonl_lines(output_labeled)
                job.logger.info(
                    "dataset build file reused | %s/%s | file=%s | labeled_samples=%s | discovery=source_refresh",
                    file_index,
                    len(sampled_files),
                    relative_name,
                    source_count,
                )
                completed_files.add(relative_name)
                file_sample_counts[relative_name] = source_count
                processed_count += 1
                labeled_samples_total += source_count
                last_completed_file_name = relative_name
                reused_new_files += 1
                _log_build_progress_if_needed()
                _write_build_state()
                if _target_reached():
                    break
                continue
            new_pending_files.append(
                {
                    "file_index": file_index,
                    "relative_name": relative_name,
                    "sampled_file": sampled_file,
                    "output_labeled": output_labeled,
                }
            )
        return new_pending_files, reused_new_files, latest_source_status

    while follow_source_updates and not _target_reached():
        new_pending_files, reused_new_files, latest_source_status = _discover_new_sampled_files()
        if reused_new_files > 0:
            _export_partial_snapshot_if_needed(force=True)
        if new_pending_files:
            global_target_reached_polls = 0
            if adaptive_source_polling:
                current_poll_interval_seconds = source_poll_interval_min_seconds
            job.logger.info(
                "dataset build source refresh detected new files | dataset=%s | source_dataset_id=%s | source_status=%s | new_pending_files=%s | reused_new_files=%s | total_known_files=%s",
                dataset_id,
                source_dataset_id,
                latest_source_status,
                len(new_pending_files),
                reused_new_files,
                len(sampled_files),
            )
            job.write_event(
                "dataset_build_source_refreshed",
                dataset_id=dataset_id,
                source_dataset_id=source_dataset_id,
                source_status=latest_source_status,
                new_pending_files=len(new_pending_files),
                reused_new_files=reused_new_files,
                total_known_files=len(sampled_files),
            )
            _process_pending_files_batch(new_pending_files)
            continue
        if latest_source_status == "completed":
            break
        if _global_generation_target_reached_for_build():
            global_target_reached_polls += 1
            if global_target_reached_polls >= global_target_stabilization_polls:
                job.logger.info(
                    "dataset build global target reached with no new files, finalizing | dataset=%s | source_dataset_id=%s | polls=%s",
                    dataset_id,
                    source_dataset_id,
                    global_target_reached_polls,
                )
                break
        else:
            global_target_reached_polls = 0
        job.logger.info(
            "dataset build waiting for source updates | dataset=%s | source_dataset_id=%s | source_status=%s | processed_files=%s | known_files=%s | poll_interval_seconds=%.1f",
            dataset_id,
            source_dataset_id,
            latest_source_status,
            processed_count,
            len(sampled_files),
            current_poll_interval_seconds,
        )
        job.write_event(
            "dataset_build_waiting_for_source_updates",
            dataset_id=dataset_id,
            source_dataset_id=source_dataset_id,
            source_status=latest_source_status,
            processed_files=processed_count,
            known_files=len(sampled_files),
            poll_interval_seconds=current_poll_interval_seconds,
        )
        time.sleep(current_poll_interval_seconds)
        if adaptive_source_polling:
            current_poll_interval_seconds = min(source_poll_interval_max_seconds, current_poll_interval_seconds * 1.5)

    split_summary, exported_input_dim, duplicate_samples_removed = _export_built_dataset_snapshot(
        job=job,
        dataset_id=dataset_id,
        source_dataset_id=source_dataset_id,
        source_dataset_ids=[source_dataset_id] if source_dataset_id else [],
        sampled_root=sampled_root,
        labeled_root=labeled_root,
        output_root=output_root,
        label_cache_dir=label_cache_dir,
        teacher_engine=teacher_engine,
        teacher_level=teacher_level,
        sampled_relative_names=sampled_relative_names,
        completed_files=completed_files,
        file_sample_counts=file_sample_counts,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        labeled_samples_total=labeled_samples_total,
        target_labeled_samples=target_labeled_samples,
        build_status="completed",
        build_mode=effective_build_mode,
        dedupe_sample_ids=dedupe_sample_ids,
    )

    summary = {
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "build_mode": effective_build_mode,
        "value_target_mix_teacher_weight": value_target_mix_teacher_weight,
        "hard_examples_enabled": hard_examples_enabled,
        "hard_examples_margin_threshold": hard_examples_margin_threshold,
        "hard_examples_outcome_focus": hard_examples_outcome_focus,
        "hard_examples_weight_multiplier": hard_examples_weight_multiplier,
        "teacher_engine": teacher_engine,
        "teacher_level": teacher_level,
        "source_dataset_id": source_dataset_id,
        "label_cache_dir": str(label_cache_dir),
        "splits": split_summary,
        "output_dir": str(output_root),
        "labeled_samples": labeled_samples_total,
        "target_labeled_samples": target_labeled_samples,
        "input_dim": exported_input_dim,
        "feature_schema_version": "policy_value_tactical_v3",
        "dedupe_sample_ids": dedupe_sample_ids,
        "duplicate_samples_removed": duplicate_samples_removed,
        "skipped_terminal_samples": skipped_terminal_samples,
        "skipped_no_legal_samples": skipped_no_legal_samples,
        "skipped_invalid_samples": skipped_invalid_samples,
    }
    _write_json(dataset_dir / "dataset_build_summary.json", summary)
    job.logger.info(
        "dataset build completed | dataset=%s | build_mode=%s | source_dataset_id=%s | train=%s | validation=%s | test=%s | duplicate_samples_removed=%s | skipped_terminal=%s | skipped_no_legal=%s | skipped_invalid=%s | output_dir=%s",
        dataset_id,
        effective_build_mode,
        source_dataset_id,
        split_summary.get("train", {}).get("samples", 0),
        split_summary.get("validation", {}).get("samples", 0),
        split_summary.get("test", {}).get("samples", 0),
        duplicate_samples_removed,
        skipped_terminal_samples,
        skipped_no_legal_samples,
        skipped_invalid_samples,
        output_root,
    )
    job.write_metric(
        {
            "metric_type": "dataset_build_completed",
            "dataset_id": dataset_id,
            "build_mode": effective_build_mode,
            "duplicate_samples_removed": duplicate_samples_removed,
            "dedupe_sample_ids": dedupe_sample_ids,
        }
    )
    job.write_event(
        "dataset_build_completed",
        dataset_id=dataset_id,
        build_mode=effective_build_mode,
        duplicate_samples_removed=duplicate_samples_removed,
        dedupe_sample_ids=dedupe_sample_ids,
    )
    if progressive_global_merge_enabled and progressive_global_merge_on_completion:
        _maybe_progressive_global_merge(force=True, trigger="build_completed")
    if progressive_global_merge_executor is not None:
        drained = _drain_progressive_global_merge_future(
            block=True,
            timeout_seconds=progressive_global_merge_completion_wait_seconds,
        )
        if not drained:
            job.logger.warning(
                "dataset build progressive global merge shutdown timeout | wait_seconds=%.1f",
                progressive_global_merge_completion_wait_seconds,
            )
        progressive_global_merge_executor.shutdown(wait=False)
    _write_global_progress_read_telemetry_metric()
    return summary


def run_dataset_merge_final(job: JobContext, *, cfg_override: dict[str, Any] | None = None) -> dict[str, object]:
    cfg = cfg_override if isinstance(cfg_override, dict) else job.config.get("dataset_merge_final", {})
    dataset_id = str(cfg.get("dataset_id", "dataset_merged_final")).strip()
    if not dataset_id:
        raise ValueError("`dataset_id` est requis pour `dataset_merge_final`")

    source_dataset_ids = [str(value).strip() for value in cfg.get("source_dataset_ids", []) if str(value).strip()]
    include_all_built = bool(cfg.get("include_all_built", False))
    source_dataset_id_prefix = str(cfg.get("source_dataset_id_prefix", "")).strip()
    dedupe_sample_ids = bool(cfg.get("dedupe_sample_ids", True))
    merge_lock_ttl_seconds = max(30.0, float(cfg.get("merge_lock_ttl_seconds", 1800.0)))
    merge_lock_wait_seconds = max(1.0, float(cfg.get("merge_lock_wait_seconds", 180.0)))

    lock_handle = _acquire_dataset_merge_final_lock(
        job,
        dataset_id=dataset_id,
        lock_ttl_seconds=merge_lock_ttl_seconds,
        lock_wait_seconds=merge_lock_wait_seconds,
        poll_seconds=1.0,
    )
    job.logger.info(
        "dataset final merge lock acquired | dataset=%s | backend=%s | owner_job_id=%s",
        dataset_id,
        str(lock_handle.get("backend", "")),
        str(lock_handle.get("owner_job_id", job.job_id)),
    )
    job.write_event(
        "dataset_merge_final_lock_acquired",
        dataset_id=dataset_id,
        lock_backend=str(lock_handle.get("backend", "")),
        lock_owner_job_id=str(lock_handle.get("owner_job_id", job.job_id)),
    )

    try:
        registry = _read_dataset_registry(job)
        built_entries = registry.get("built_datasets", [])
        if include_all_built:
            source_dataset_ids = [str(entry.get("dataset_id", "")).strip() for entry in built_entries if str(entry.get("dataset_id", "")).strip()]
        if source_dataset_id_prefix:
            source_dataset_ids = [item for item in source_dataset_ids if item.startswith(source_dataset_id_prefix)]
        source_dataset_ids = [item for item in source_dataset_ids if item and item != dataset_id]
        source_dataset_ids = list(dict.fromkeys(source_dataset_ids))
        if not source_dataset_ids:
            raise ValueError("Aucun `source_dataset_ids` fourni pour `dataset_merge_final`")

        source_entries = [_resolve_built_dataset(job, source_dataset_id) for source_dataset_id in source_dataset_ids]
        teacher_pairs = {(str(entry.get("teacher_engine", "")), str(entry.get("teacher_level", ""))) for entry in source_entries}
        if len(teacher_pairs) != 1:
            raise ValueError("Tous les datasets finaux a fusionner doivent partager le meme teacher")
        teacher_engine, teacher_level = next(iter(teacher_pairs))

        output_root = _resolve_storage_path(
            job.paths.drive_root,
            cfg.get("output_dir"),
            job.paths.data_root / "datasets" / dataset_id,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        merge_dir = job.job_dir / "dataset_merge_final"
        merge_dir.mkdir(parents=True, exist_ok=True)

        job.logger.info(
            "dataset final merge started | dataset=%s | source_datasets=%s | teacher=%s:%s | dedupe_sample_ids=%s | output_dir=%s",
            dataset_id,
            len(source_entries),
            teacher_engine,
            teacher_level,
            dedupe_sample_ids,
            output_root,
        )
        job.write_event(
            "dataset_merge_final_started",
            dataset_id=dataset_id,
            source_dataset_ids=source_dataset_ids,
            teacher_engine=teacher_engine,
            teacher_level=teacher_level,
            dedupe_sample_ids=dedupe_sample_ids,
            output_dir=str(output_root),
        )

        split_summary: dict[str, dict[str, int]] = {}
        merge_breakdown: dict[str, dict[str, int]] = {}
        source_breakdown: dict[str, dict[str, dict[str, int]]] = {}
        total_labeled_samples = 0
        source_sampled_roots = [str(entry.get("sampled_root", "")) for entry in source_entries if str(entry.get("sampled_root", "")).strip()]
        label_cache_dirs = [str(entry.get("label_cache_dir", "")) for entry in source_entries if str(entry.get("label_cache_dir", "")).strip()]

        for split_name in ("train", "validation", "test"):
            split_items: list[tuple[str, Path]] = []
            for entry in source_entries:
                output_dir = Path(str(entry["output_dir"]))
                split_path = output_dir / f"{split_name}.npz"
                if not split_path.exists():
                    raise FileNotFoundError(f"Split introuvable pour {entry['dataset_id']}: {split_path}")
                split_items.append((str(entry["dataset_id"]), split_path))

            job.logger.info(
                "dataset final merge split started | split=%s | source_files=%s",
                split_name,
                len(split_items),
            )
            merged_arrays, split_metrics, split_source_breakdown = _merge_npz_splits_with_source_breakdown(
                split_items,
                dedupe_sample_ids=dedupe_sample_ids,
            )
            _write_npz_compressed(output_root / f"{split_name}.npz", **merged_arrays)
            split_summary[split_name] = {
                "games": int(split_metrics["unique_games"]),
                "samples": int(split_metrics["kept_samples"]),
            }
            merge_breakdown[split_name] = split_metrics
            source_breakdown[split_name] = split_source_breakdown
            total_labeled_samples += int(split_metrics["kept_samples"])
            job.logger.info(
                "dataset final merge split completed | split=%s | kept_samples=%s | duplicate_samples=%s | unique_games=%s | path=%s",
                split_name,
                split_metrics["kept_samples"],
                split_metrics["duplicate_samples"],
                split_metrics["unique_games"],
                output_root / f"{split_name}.npz",
            )
            for source_dataset_id, stats in split_source_breakdown.items():
                job.logger.info(
                    "dataset final merge source breakdown | split=%s | source_dataset_id=%s | input_samples=%s | kept_samples=%s | duplicate_samples=%s | unique_games=%s",
                    split_name,
                    source_dataset_id,
                    stats["input_samples"],
                    stats["kept_samples"],
                    stats["duplicate_samples"],
                    stats["unique_games"],
                )

        metadata = _register_built_dataset(
            job,
            dataset_id=dataset_id,
            source_dataset_id=source_dataset_ids[0],
            source_dataset_ids=source_dataset_ids,
            sampled_root=Path(source_sampled_roots[0]) if source_sampled_roots else output_root,
            output_root=output_root,
            label_cache_dir=Path(label_cache_dirs[0]) if label_cache_dirs else output_root,
            teacher_engine=teacher_engine,
            teacher_level=teacher_level,
            split_summary=split_summary,
            labeled_samples=total_labeled_samples,
            target_labeled_samples=total_labeled_samples,
            build_mode="merged_final",
            parent_dataset_ids=source_dataset_ids,
        )
        metadata["merge_breakdown"] = merge_breakdown
        metadata["source_breakdown"] = source_breakdown
        metadata["dedupe_sample_ids"] = dedupe_sample_ids
        metadata["source_dataset_ids"] = source_dataset_ids
        _write_json(output_root / "dataset_metadata.json", metadata)

        summary = {
            "job_id": job.job_id,
            "dataset_id": dataset_id,
            "build_mode": "merged_final",
            "source_dataset_ids": source_dataset_ids,
            "teacher_engine": teacher_engine,
            "teacher_level": teacher_level,
            "dedupe_sample_ids": dedupe_sample_ids,
            "splits": split_summary,
            "merge_breakdown": merge_breakdown,
            "source_breakdown": source_breakdown,
            "output_dir": str(output_root),
            "labeled_samples": total_labeled_samples,
        }
        _write_json(merge_dir / "dataset_merge_final_summary.json", summary)
        job.logger.info(
            "dataset final merge completed | dataset=%s | sources=%s | train=%s | validation=%s | test=%s | labeled_samples=%s",
            dataset_id,
            len(source_dataset_ids),
            split_summary.get("train", {}).get("samples", 0),
            split_summary.get("validation", {}).get("samples", 0),
            split_summary.get("test", {}).get("samples", 0),
            total_labeled_samples,
        )
        job.write_event(
            "dataset_merge_final_completed",
            dataset_id=dataset_id,
            source_dataset_ids=source_dataset_ids,
            labeled_samples=total_labeled_samples,
            source_breakdown=source_breakdown,
        )
        job.write_metric({"metric_type": "dataset_merge_final_completed", "dataset_id": dataset_id, "labeled_samples": total_labeled_samples})
        return summary
    finally:
        _release_dataset_merge_final_lock(job, lock_handle)
        job.write_event(
            "dataset_merge_final_lock_released",
            dataset_id=dataset_id,
            lock_backend=str(lock_handle.get("backend", "")),
        )
