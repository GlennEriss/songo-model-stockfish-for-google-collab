from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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


def _valid_split_triplet(dataset_output_dir: Path) -> bool:
    return (
        (dataset_output_dir / "train.npz").exists()
        and (dataset_output_dir / "validation.npz").exists()
        and (dataset_output_dir / "test.npz").exists()
    )


def _collect_workspace_built_entries(
    *,
    drive_root: Path,
    source_dataset_id_prefix: str,
) -> list[dict[str, Any]]:
    collected: dict[str, dict[str, Any]] = {}

    for workspace_dir in sorted(drive_root.iterdir(), key=lambda item: item.name):
        if not workspace_dir.is_dir():
            continue
        if not (workspace_dir.name.startswith("colab_") or workspace_dir.name == "unknown_drive_identity"):
            continue

        registry_path = workspace_dir / "data" / "dataset_registry.json"
        registry = _load_json(registry_path, {"dataset_sources": [], "built_datasets": []})
        built_entries = registry.get("built_datasets", [])
        if not isinstance(built_entries, list):
            continue

        for entry in built_entries:
            if not isinstance(entry, dict):
                continue
            dataset_id = str(entry.get("dataset_id", "")).strip()
            if not dataset_id:
                continue
            if source_dataset_id_prefix and not dataset_id.startswith(source_dataset_id_prefix):
                continue

            output_dir_text = str(entry.get("output_dir", "")).strip()
            if not output_dir_text:
                continue
            output_dir = Path(output_dir_text)
            if not output_dir.is_absolute():
                output_dir = drive_root / output_dir
            output_dir = output_dir.resolve(strict=False)
            if not _valid_split_triplet(output_dir):
                continue

            normalized = dict(entry)
            normalized["dataset_id"] = dataset_id
            normalized["output_dir"] = str(output_dir)
            normalized["workspace"] = workspace_dir.name
            normalized["labeled_samples"] = int(normalized.get("labeled_samples", 0) or 0)

            previous = collected.get(dataset_id)
            if previous is None:
                collected[dataset_id] = normalized
                continue

            prev_updated = str(previous.get("updated_at", ""))
            new_updated = str(normalized.get("updated_at", ""))
            if new_updated >= prev_updated:
                collected[dataset_id] = normalized

    rows = list(collected.values())
    rows.sort(
        key=lambda item: (
            int(item.get("labeled_samples", 0) or 0),
            str(item.get("updated_at", "")),
            str(item.get("dataset_id", "")),
        ),
        reverse=True,
    )
    return rows


def _upsert_global_registry_built_entries(
    *,
    global_registry_path: Path,
    source_entries: list[dict[str, Any]],
    merged_dataset_id: str,
) -> dict[str, Any]:
    registry = _load_json(global_registry_path, {"dataset_sources": [], "built_datasets": []})
    dataset_sources = registry.get("dataset_sources", [])
    if not isinstance(dataset_sources, list):
        dataset_sources = []
    built_datasets = registry.get("built_datasets", [])
    if not isinstance(built_datasets, list):
        built_datasets = []

    by_dataset_id: dict[str, dict[str, Any]] = {}
    for item in built_datasets:
        if not isinstance(item, dict):
            continue
        dataset_id = str(item.get("dataset_id", "")).strip()
        if dataset_id:
            by_dataset_id[dataset_id] = dict(item)

    # Force overwrite behavior for merged dataset id on every run.
    by_dataset_id.pop(str(merged_dataset_id).strip(), None)

    for entry in source_entries:
        dataset_id = str(entry.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        by_dataset_id[dataset_id] = dict(entry)

    merged_list = list(by_dataset_id.values())
    merged_list.sort(
        key=lambda item: (
            int(item.get("labeled_samples", 0) or 0),
            str(item.get("updated_at", "")),
            str(item.get("dataset_id", "")),
        ),
        reverse=True,
    )

    payload = {
        "dataset_sources": dataset_sources,
        "built_datasets": merged_list,
    }
    _write_json(global_registry_path, payload)
    return payload


def _build_merge_config_payload(
    *,
    base_cfg: dict[str, Any],
    identity: str,
    drive_root: Path,
    merged_dataset_id: str,
    source_dataset_ids: list[str],
    dedupe_sample_ids: bool,
) -> dict[str, Any]:
    payload = dict(base_cfg)
    storage_cfg = dict(payload.get("storage", {}) or {})
    storage_cfg["drive_root"] = str(drive_root)
    storage_cfg["jobs_root"] = f"{identity}/jobs"
    storage_cfg["jobs_backup_root"] = f"{identity}/runtime_backup/jobs"
    storage_cfg["logs_root"] = f"{identity}/logs"
    storage_cfg["reports_root"] = f"{identity}/reports"
    storage_cfg["models_root"] = "models"
    # Important: registry global pour pouvoir fusionner tous les builds colabs.
    storage_cfg["data_root"] = "data"
    storage_cfg["runtime_state_backup_enabled"] = True
    payload["storage"] = storage_cfg

    job_cfg = dict(payload.get("job", {}) or {})
    job_cfg["run_type"] = "dataset_merge_final"
    job_cfg["resume"] = True
    job_cfg["job_id"] = f"dataset_merge_final_{identity}"
    payload["job"] = job_cfg

    merge_cfg = dict(payload.get("dataset_merge_final", {}) or {})
    merge_cfg["dataset_id"] = merged_dataset_id
    merge_cfg["source_dataset_ids"] = list(source_dataset_ids)
    merge_cfg["include_all_built"] = False
    merge_cfg["dedupe_sample_ids"] = bool(dedupe_sample_ids)
    merge_cfg["output_dir"] = f"data/datasets/{merged_dataset_id}"
    payload["dataset_merge_final"] = merge_cfg

    return payload


def _patch_train_eval_active_configs(
    *,
    worktree: Path,
    identity: str,
    merged_dataset_id: str,
) -> dict[str, str]:
    train_path = worktree / "config" / "generated" / f"train.{identity}.active.yaml"
    eval_path = worktree / "config" / "generated" / f"evaluation.{identity}.active.yaml"

    if not train_path.exists():
        raise FileNotFoundError(f"Config train active introuvable: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Config evaluation active introuvable: {eval_path}")

    train_payload = _load_yaml(train_path)
    train_payload.setdefault("storage", {})
    train_payload["storage"]["data_root"] = "data"
    train_payload.setdefault("train", {})
    train_payload["train"]["dataset_selection_mode"] = "configured"
    train_payload["train"]["dataset_id"] = merged_dataset_id
    train_payload["train"]["dataset_path"] = ""
    train_payload["train"]["validation_path"] = ""
    _save_yaml(train_path, train_payload)

    eval_payload = _load_yaml(eval_path)
    eval_payload.setdefault("storage", {})
    eval_payload["storage"]["data_root"] = "data"
    eval_payload.setdefault("evaluation", {})
    eval_payload["evaluation"]["dataset_selection_mode"] = "configured"
    eval_payload["evaluation"]["dataset_id"] = merged_dataset_id
    eval_payload["evaluation"]["test_dataset_path"] = ""
    _save_yaml(eval_path, eval_payload)

    return {
        "train_config": str(train_path),
        "evaluation_config": str(eval_path),
    }


def _format_bytes(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024.0 and idx < (len(units) - 1):
        size /= 1024.0
        idx += 1
    return f"{size:.2f}{units[idx]}"


def _compute_dataset_disk_bytes(output_dir: Path) -> int:
    total = 0
    for name in ("train.npz", "validation.npz", "test.npz", "dataset_metadata.json"):
        path = output_dir / name
        if path.exists() and path.is_file():
            total += int(path.stat().st_size)
    return total


def _read_merged_dataset_stats(*, merged_dataset_id: str, merged_output_dir: Path, global_registry_path: Path) -> dict[str, Any]:
    train_samples = 0
    validation_samples = 0
    test_samples = 0
    labeled_samples = 0

    metadata_path = merged_output_dir / "dataset_metadata.json"
    metadata = _load_json(metadata_path, {})
    if metadata:
        splits = metadata.get("splits", {})
        if isinstance(splits, dict):
            train_samples = int(((splits.get("train", {}) or {}).get("samples", 0) or 0))
            validation_samples = int(((splits.get("validation", {}) or {}).get("samples", 0) or 0))
            test_samples = int(((splits.get("test", {}) or {}).get("samples", 0) or 0))
        labeled_samples = int(metadata.get("labeled_samples", 0) or 0)

    if labeled_samples <= 0:
        registry = _load_json(global_registry_path, {"dataset_sources": [], "built_datasets": []})
        built_entries = registry.get("built_datasets", [])
        if isinstance(built_entries, list):
            for entry in built_entries:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("dataset_id", "")).strip() != str(merged_dataset_id).strip():
                    continue
                labeled_samples = int(entry.get("labeled_samples", 0) or 0)
                splits = entry.get("splits", {})
                if isinstance(splits, dict):
                    train_samples = int(((splits.get("train", {}) or {}).get("samples", train_samples) or train_samples))
                    validation_samples = int(((splits.get("validation", {}) or {}).get("samples", validation_samples) or validation_samples))
                    test_samples = int(((splits.get("test", {}) or {}).get("samples", test_samples) or test_samples))
                break

    if labeled_samples <= 0:
        labeled_samples = int(train_samples + validation_samples + test_samples)

    disk_bytes = _compute_dataset_disk_bytes(merged_output_dir)
    return {
        "labeled_samples": int(labeled_samples),
        "train_samples": int(train_samples),
        "validation_samples": int(validation_samples),
        "test_samples": int(test_samples),
        "disk_bytes": int(disk_bytes),
        "disk_human": _format_bytes(disk_bytes),
    }


def run_merge_built_datasets(
    *,
    python_bin: str,
    worktree: Path,
    drive_root: Path,
    identity: str,
    merged_dataset_id: str,
    source_dataset_id_prefix: str,
    dedupe_sample_ids: bool,
    heartbeat_seconds: int,
) -> dict[str, Any]:
    identity_key = str(identity or "").strip() or "unknown_drive_identity"
    merged_dataset_id = str(merged_dataset_id or "").strip()
    if not merged_dataset_id:
        raise ValueError("merged_dataset_id vide")

    source_entries = _collect_workspace_built_entries(
        drive_root=drive_root,
        source_dataset_id_prefix=str(source_dataset_id_prefix or "").strip(),
    )
    if len(source_entries) < 1:
        raise RuntimeError(
            "Fusion impossible: aucun dataset build valide detecte "
            f"(trouves={len(source_entries)})."
        )

    source_dataset_ids = [str(item["dataset_id"]) for item in source_entries]

    print(
        "merge-built-datasets source selection "
        f"| dataset_count={len(source_dataset_ids)} | source_prefix={source_dataset_id_prefix}",
        flush=True,
    )
    if len(source_dataset_ids) == 1:
        print(
            "merge-built-datasets info | un seul dataset source detecte: "
            "une fusion finale mono-source sera generee pour garder un dataset cible unique.",
            flush=True,
        )
    for item in source_entries:
        print(
            " - "
            f"dataset_id={item.get('dataset_id')} "
            f"| workspace={item.get('workspace')} "
            f"| labeled_samples={int(item.get('labeled_samples', 0) or 0)} "
            f"| output_dir={item.get('output_dir')}",
            flush=True,
        )

    global_registry_path = drive_root / "data" / "dataset_registry.json"
    global_registry_payload = _upsert_global_registry_built_entries(
        global_registry_path=global_registry_path,
        source_entries=source_entries,
        merged_dataset_id=merged_dataset_id,
    )
    print("global registry path =", global_registry_path, flush=True)
    print(
        "global built datasets in registry =",
        len(list(global_registry_payload.get("built_datasets", []))),
        flush=True,
    )

    merged_output_dir = drive_root / "data" / "datasets" / merged_dataset_id
    if merged_output_dir.exists():
        print("Suppression ancienne fusion dataset:", merged_output_dir, flush=True)
        shutil.rmtree(merged_output_dir)

    base_cfg_path = worktree / "config" / "dataset_merge_final.colab_pro.yaml"
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Config merge base introuvable: {base_cfg_path}")

    base_cfg = _load_yaml(base_cfg_path)
    merge_cfg_payload = _build_merge_config_payload(
        base_cfg=base_cfg,
        identity=identity_key,
        drive_root=drive_root,
        merged_dataset_id=merged_dataset_id,
        source_dataset_ids=source_dataset_ids,
        dedupe_sample_ids=bool(dedupe_sample_ids),
    )

    generated_cfg_path = worktree / "config" / "generated" / f"dataset_merge_final.{identity_key}.active.yaml"
    _save_yaml(generated_cfg_path, merge_cfg_payload)
    print("merge config active =", generated_cfg_path, flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    env["PYTHONUNBUFFERED"] = "1"

    _run_live(
        [
            str(python_bin),
            "-u",
            "-m",
            "songo_model_stockfish.cli.main",
            "dataset-merge-final",
            "--config",
            str(generated_cfg_path),
            "--dataset-id",
            merged_dataset_id,
            "--dedupe-sample-ids",
        ],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )

    merged_stats = _read_merged_dataset_stats(
        merged_dataset_id=merged_dataset_id,
        merged_output_dir=merged_output_dir,
        global_registry_path=global_registry_path,
    )
    print(
        "merged dataset size "
        f"| dataset_id={merged_dataset_id} "
        f"| labeled_samples={int(merged_stats.get('labeled_samples', 0))} "
        f"| split_samples(train/val/test)="
        f"{int(merged_stats.get('train_samples', 0))}/"
        f"{int(merged_stats.get('validation_samples', 0))}/"
        f"{int(merged_stats.get('test_samples', 0))} "
        f"| disk={merged_stats.get('disk_human', '0B')} "
        f"| output_dir={merged_output_dir}",
        flush=True,
    )

    patched = _patch_train_eval_active_configs(
        worktree=worktree,
        identity=identity_key,
        merged_dataset_id=merged_dataset_id,
    )

    return {
        "identity": identity_key,
        "merged_dataset_id": merged_dataset_id,
        "source_dataset_ids": source_dataset_ids,
        "source_dataset_count": len(source_dataset_ids),
        "dedupe_sample_ids": bool(dedupe_sample_ids),
        "global_registry_path": str(global_registry_path),
        "merge_config_path": str(generated_cfg_path),
        "merged_output_dir": str(merged_output_dir),
        "merged_dataset_stats": merged_stats,
        "patched_active_configs": patched,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    parser.add_argument(
        "--drive-root",
        default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"),
    )
    parser.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    parser.add_argument("--python-bin", default=(sys.executable or os.environ.get("PYTHON_BIN", "python3")))
    parser.add_argument("--merged-dataset-id", default="dataset_full_matrix_merged_all_colabs")
    parser.add_argument("--source-dataset-id-prefix", default="dataset_full_matrix_colab_")
    parser.add_argument("--dedupe-sample-ids", dest="dedupe_sample_ids", action="store_true", default=True)
    parser.add_argument("--keep-duplicate-sample-ids", dest="dedupe_sample_ids", action="store_false")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    summary = run_merge_built_datasets(
        python_bin=str(args.python_bin),
        worktree=Path(str(args.worktree)),
        drive_root=Path(str(args.drive_root)),
        identity=str(args.identity),
        merged_dataset_id=str(args.merged_dataset_id),
        source_dataset_id_prefix=str(args.source_dataset_id_prefix),
        dedupe_sample_ids=bool(args.dedupe_sample_ids),
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
