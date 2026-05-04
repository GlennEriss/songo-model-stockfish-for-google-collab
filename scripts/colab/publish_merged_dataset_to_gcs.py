from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REQUIRED_DATASET_FILES = (
    "train.npz",
    "validation.npz",
    "test.npz",
    "dataset_metadata.json",
)


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else dict(default)
    except Exception:
        return dict(default)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _run_live(cmd: list[str], *, cwd: Path, env: dict[str, str], heartbeat_s: int = 30) -> None:
    print("RUN:", cmd, flush=True)
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    last_hb = started
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
        now = time.time()
        if (now - last_hb) >= max(10, int(heartbeat_s)):
            print(f"[heartbeat] elapsed={int(now-started)}s | process_running=True", flush=True)
            last_hb = now
    rc = proc.wait()
    print(f"[exit] returncode={rc} | elapsed={int(time.time()-started)}s", flush=True)
    if rc != 0:
        raise RuntimeError(f"Commande en echec (rc={rc}): {cmd}")


def _normalize_bucket(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("gs://"):
        text = text[len("gs://") :]
    text = text.strip().strip("/")
    if not text:
        raise ValueError("Bucket GCS vide. Configure --gcs-bucket ou SONGO_VERTEX_GCS_BUCKET.")
    if "/" in text:
        raise ValueError(f"Bucket GCS invalide (attendu nom bucket seul): {text}")
    return text


def _normalize_prefix(value: str) -> str:
    return str(value or "").strip().strip("/")


def _build_gs_uri(bucket: str, *segments: str, prefix: str = "") -> str:
    parts = [f"gs://{bucket}"]
    if prefix:
        parts.append(prefix)
    for segment in segments:
        seg = str(segment or "").strip().strip("/")
        if seg:
            parts.append(seg)
    return "/".join(parts)


def _build_vertex_path(bucket: str, *segments: str, prefix: str = "") -> str:
    parts = ["/gcs", bucket]
    if prefix:
        parts.append(prefix)
    for segment in segments:
        seg = str(segment or "").strip().strip("/")
        if seg:
            parts.append(seg)
    return "/".join(parts)


def _resolve_merged_output_dir(
    *,
    drive_root: Path,
    merged_dataset_id: str,
    merge_summary_path: Path | None,
) -> tuple[Path, dict[str, Any]]:
    if merge_summary_path is not None:
        payload = _load_json(merge_summary_path, {})
        merged_output_dir = str(payload.get("merged_output_dir", "")).strip()
        if merged_output_dir:
            return Path(merged_output_dir).resolve(strict=False), payload
    return (drive_root / "data" / "datasets" / merged_dataset_id).resolve(strict=False), {}


def _validate_merged_dataset_dir(merged_output_dir: Path) -> None:
    if not merged_output_dir.exists():
        raise FileNotFoundError(f"Dataset fusionne introuvable: {merged_output_dir}")
    missing = [name for name in _REQUIRED_DATASET_FILES if not (merged_output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset fusionne incomplet. Fichiers manquants: "
            f"{missing} | output_dir={merged_output_dir}"
        )


def _stat_dataset_bytes(merged_output_dir: Path) -> int:
    total = 0
    for name in _REQUIRED_DATASET_FILES:
        path = merged_output_dir / name
        if path.exists() and path.is_file():
            total += int(path.stat().st_size)
    return total


def _read_dataset_split_counts(merged_output_dir: Path) -> dict[str, int]:
    metadata = _load_json(merged_output_dir / "dataset_metadata.json", {})
    splits = metadata.get("splits", {}) if isinstance(metadata, dict) else {}
    train_samples = int(((splits.get("train", {}) or {}).get("samples", 0) or 0))
    validation_samples = int(((splits.get("validation", {}) or {}).get("samples", 0) or 0))
    test_samples = int(((splits.get("test", {}) or {}).get("samples", 0) or 0))
    labeled_samples = int(metadata.get("labeled_samples", 0) or 0) if isinstance(metadata, dict) else 0
    if labeled_samples <= 0:
        labeled_samples = train_samples + validation_samples + test_samples
    return {
        "labeled_samples": int(labeled_samples),
        "train_samples": int(train_samples),
        "validation_samples": int(validation_samples),
        "test_samples": int(test_samples),
    }


def run_publish_merged_dataset_to_gcs(
    *,
    worktree: Path,
    drive_root: Path,
    merged_dataset_id: str,
    merge_summary_path: Path | None,
    gcs_bucket: str,
    gcs_prefix: str,
    sync_models: bool,
    heartbeat_seconds: int,
) -> dict[str, Any]:
    bucket = _normalize_bucket(gcs_bucket)
    prefix = _normalize_prefix(gcs_prefix)
    merged_dataset_id = str(merged_dataset_id or "").strip()
    if not merged_dataset_id:
        raise ValueError("merged_dataset_id vide")

    merged_output_dir, merge_summary_payload = _resolve_merged_output_dir(
        drive_root=drive_root,
        merged_dataset_id=merged_dataset_id,
        merge_summary_path=merge_summary_path,
    )
    _validate_merged_dataset_dir(merged_output_dir)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    dataset_destination_uri = _build_gs_uri(
        bucket,
        "data",
        "datasets",
        merged_dataset_id,
        prefix=prefix,
    )
    _run_live(
        [
            "gcloud",
            "storage",
            "rsync",
            str(merged_output_dir),
            dataset_destination_uri,
            "--recursive",
            "--delete-unmatched-destination-objects",
        ],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )

    dataset_registry_local_path = drive_root / "data" / "dataset_registry.json"
    dataset_registry_destination_uri = _build_gs_uri(
        bucket,
        "data",
        "dataset_registry.json",
        prefix=prefix,
    )
    dataset_registry_synced = False
    if dataset_registry_local_path.exists():
        _run_live(
            [
                "gcloud",
                "storage",
                "cp",
                str(dataset_registry_local_path),
                dataset_registry_destination_uri,
            ],
            cwd=worktree,
            env=env,
            heartbeat_s=int(heartbeat_seconds),
        )
        dataset_registry_synced = True
    else:
        print(
            "dataset_registry.json absent localement; sync ignoree "
            f"| path={dataset_registry_local_path}",
            flush=True,
        )

    models_source_dir = drive_root / "models"
    models_destination_uri = _build_gs_uri(bucket, "models", prefix=prefix)
    models_sync_performed = False
    if sync_models:
        if models_source_dir.exists():
            _run_live(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    str(models_source_dir),
                    models_destination_uri,
                    "--recursive",
                ],
                cwd=worktree,
                env=env,
                heartbeat_s=int(heartbeat_seconds),
            )
            models_sync_performed = True
        else:
            print(f"models/ absent; sync models ignoree | path={models_source_dir}", flush=True)

    split_counts = _read_dataset_split_counts(merged_output_dir)
    dataset_disk_bytes = _stat_dataset_bytes(merged_output_dir)

    train_npz_vertex_path = _build_vertex_path(
        bucket,
        "data",
        "datasets",
        merged_dataset_id,
        "train.npz",
        prefix=prefix,
    )
    validation_npz_vertex_path = _build_vertex_path(
        bucket,
        "data",
        "datasets",
        merged_dataset_id,
        "validation.npz",
        prefix=prefix,
    )
    test_npz_vertex_path = _build_vertex_path(
        bucket,
        "data",
        "datasets",
        merged_dataset_id,
        "test.npz",
        prefix=prefix,
    )

    latest_payload = {
        "dataset_id": merged_dataset_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "source_merged_output_dir": str(merged_output_dir),
        "source_merge_summary": merge_summary_payload,
        "gcs_bucket": bucket,
        "gcs_prefix": prefix,
        "gcs_dataset_dir": dataset_destination_uri,
        "vertex_dataset_dir": _build_vertex_path(
            bucket,
            "data",
            "datasets",
            merged_dataset_id,
            prefix=prefix,
        ),
        "train_npz": {
            "gcs": _build_gs_uri(bucket, "data", "datasets", merged_dataset_id, "train.npz", prefix=prefix),
            "vertex": train_npz_vertex_path,
        },
        "validation_npz": {
            "gcs": _build_gs_uri(bucket, "data", "datasets", merged_dataset_id, "validation.npz", prefix=prefix),
            "vertex": validation_npz_vertex_path,
        },
        "test_npz": {
            "gcs": _build_gs_uri(bucket, "data", "datasets", merged_dataset_id, "test.npz", prefix=prefix),
            "vertex": test_npz_vertex_path,
        },
        "dataset_registry": {
            "local_path": str(dataset_registry_local_path),
            "gcs_path": dataset_registry_destination_uri,
            "synced": bool(dataset_registry_synced),
        },
        "stats": {
            **split_counts,
            "dataset_disk_bytes": int(dataset_disk_bytes),
        },
        "models": {
            "sync_requested": bool(sync_models),
            "sync_performed": bool(models_sync_performed),
            "local_dir": str(models_source_dir),
            "gcs_dir": models_destination_uri,
        },
    }

    latest_local_path = worktree / "config" / "generated" / "vertex" / "merged_dataset.latest.json"
    _write_json(latest_local_path, latest_payload)

    latest_destination_uri = _build_gs_uri(
        bucket,
        "data",
        "datasets",
        "merged",
        "latest.json",
        prefix=prefix,
    )
    _run_live(
        [
            "gcloud",
            "storage",
            "cp",
            str(latest_local_path),
            latest_destination_uri,
        ],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )

    summary = {
        "merged_dataset_id": merged_dataset_id,
        "merged_output_dir": str(merged_output_dir),
        "gcs_bucket": bucket,
        "gcs_prefix": prefix,
        "gcs_dataset_dir": dataset_destination_uri,
        "latest_pointer_gcs_path": latest_destination_uri,
        "latest_pointer_local_path": str(latest_local_path),
        "dataset_registry_synced": bool(dataset_registry_synced),
        "models_sync_performed": bool(models_sync_performed),
        "stats": latest_payload["stats"],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    parser.add_argument(
        "--drive-root",
        default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"),
    )
    parser.add_argument("--merged-dataset-id", default="dataset_full_matrix_merged_all_colabs")
    parser.add_argument("--merge-summary-path", default="")
    parser.add_argument("--gcs-bucket", default=(str(os.environ.get("SONGO_VERTEX_GCS_BUCKET", "")).strip()))
    parser.add_argument("--gcs-prefix", default=(str(os.environ.get("SONGO_VERTEX_GCS_PREFIX", "songo-stockfish")).strip()))
    parser.add_argument("--sync-models", dest="sync_models", action="store_true", default=True)
    parser.add_argument("--skip-sync-models", dest="sync_models", action="store_false")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    summary = run_publish_merged_dataset_to_gcs(
        worktree=Path(str(args.worktree)),
        drive_root=Path(str(args.drive_root)),
        merged_dataset_id=str(args.merged_dataset_id),
        merge_summary_path=(Path(str(args.merge_summary_path)) if str(args.merge_summary_path).strip() else None),
        gcs_bucket=str(args.gcs_bucket),
        gcs_prefix=str(args.gcs_prefix),
        sync_models=bool(args.sync_models),
        heartbeat_seconds=int(args.heartbeat_seconds),
    )

    summary_path = str(args.summary_path or "").strip()
    if summary_path:
        Path(summary_path).write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    if bool(args.print_json):
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
