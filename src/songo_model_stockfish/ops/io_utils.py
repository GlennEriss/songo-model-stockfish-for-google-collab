from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


def read_json_dict(
    path: Path,
    *,
    default: Mapping[str, Any] | None = None,
    retries: int = 6,
    wait_seconds: float = 0.05,
) -> dict[str, Any]:
    fallback = dict(default or {})
    if not path.exists():
        return fallback
    attempts = max(1, int(retries))
    for attempt in range(attempts):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            if attempt + 1 >= attempts:
                return fallback
            time.sleep(max(0.01, float(wait_seconds) * (attempt + 1)))
            continue
        if not isinstance(payload, dict):
            return fallback
        return payload
    return fallback


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: OSError | None = None
    for attempt in range(3):
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
        try:
            with tmp_path.open("w", encoding=encoding) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            return
        except FileNotFoundError as exc:
            # Google Drive FUSE peut parfois echouer sur le rename atomique.
            # On retente avant fallback.
            last_exc = exc
            path.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(0.02 * (attempt + 1))
        except OSError as exc:
            last_exc = exc
            path.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(0.02 * (attempt + 1))
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    # Fallback robuste: ecriture directe sous lock appelant.
    try:
        with path.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return
    except OSError:
        if last_exc is not None:
            raise last_exc
        raise


def write_json_atomic(path: Path, payload: Mapping[str, Any], *, ensure_ascii: bool = True, indent: int = 2) -> None:
    write_text_atomic(path, json.dumps(dict(payload), indent=indent, ensure_ascii=ensure_ascii), encoding="utf-8")


def write_jsonl_atomic(path: Path, payloads: list[dict[str, Any]], *, ensure_ascii: bool = True) -> None:
    lines = [json.dumps(payload, ensure_ascii=ensure_ascii) for payload in payloads]
    text = "\n".join(lines)
    if text:
        text += "\n"
    write_text_atomic(path, text, encoding="utf-8")


def acquire_lock_dir(
    lock_dir: Path,
    *,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.1,
    stale_after_seconds: float = 120.0,
) -> bool:
    deadline = time.time() + max(1.0, float(timeout_seconds))
    while time.time() < deadline:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            return True
        except FileExistsError:
            # Nettoyage defensif des verrous stale (runtime tue, kernel reset, etc.).
            try:
                stat = lock_dir.stat()
                age_seconds = time.time() - float(stat.st_mtime)
                if age_seconds >= max(10.0, float(stale_after_seconds)):
                    lock_dir.rmdir()
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                # Verrou legitime (ou non vide): on attend le prochain poll.
                pass
            time.sleep(max(0.01, float(poll_seconds)))
    return False


def release_lock_dir(lock_dir: Path) -> None:
    try:
        lock_dir.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return
