from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_CONFIG_STEMS = {
    "dataset-generate": "dataset_generation",
    "dataset-build": "dataset_build",
    "train": "train",
    "evaluate": "evaluation",
    "benchmark": "benchmark",
}


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else dict(default)
    except Exception:
        return dict(default)


def _resolve_storage_root(*, drive_root: Path, configured: Any, fallback_relative: str) -> Path:
    text = str(configured if configured is not None else "").strip()
    rel_or_abs = Path(text) if text else Path(fallback_relative)
    if rel_or_abs.is_absolute():
        return rel_or_abs
    return drive_root / rel_or_abs


def _normalize_registry_output_dir(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return None
    return candidate.resolve(strict=False)


def _select_largest_built_dataset_entry(registry: dict[str, Any]) -> dict[str, Any] | None:
    built_entries = registry.get("built_datasets", [])
    if not isinstance(built_entries, list):
        return None
    candidates: list[dict[str, Any]] = []
    for entry in built_entries:
        if not isinstance(entry, dict):
            continue
        output_dir = _normalize_registry_output_dir(entry.get("output_dir"))
        if output_dir is None:
            continue
        if not (output_dir / "train.npz").exists():
            continue
        if not (output_dir / "validation.npz").exists():
            continue
        normalized = dict(entry)
        normalized["output_dir"] = str(output_dir)
        candidates.append(normalized)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            int(item.get("labeled_samples", 0) or 0),
            str(item.get("updated_at", "")),
        ),
        reverse=True,
    )
    return candidates[0]


def _resolve_dataset_entry_by_id(registry: dict[str, Any], dataset_id: str) -> dict[str, Any] | None:
    built_entries = registry.get("built_datasets", [])
    if not isinstance(built_entries, list):
        return None
    requested = str(dataset_id).strip()
    if not requested:
        return None
    for entry in built_entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("dataset_id", "")).strip() == requested:
            output_dir = _normalize_registry_output_dir(entry.get("output_dir"))
            normalized = dict(entry)
            if output_dir is not None:
                normalized["output_dir"] = str(output_dir)
            return normalized
    return None


def _print_training_dataset_preflight(*, train_cfg_path: Path, drive_root: Path, identity: str) -> None:
    import yaml

    payload = yaml.safe_load(train_cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        payload = {}
    train_cfg = dict(payload.get("train", {}) or {})
    storage_cfg = dict(payload.get("storage", {}) or {})

    identity_key = str(identity or "").strip() or "unknown_drive_identity"
    data_root = _resolve_storage_root(
        drive_root=drive_root,
        configured=storage_cfg.get("data_root"),
        fallback_relative=f"{identity_key}/data",
    )
    registry_path = data_root / "dataset_registry.json"
    registry = _read_json(registry_path, {"dataset_sources": [], "built_datasets": []})

    selection_mode = str(train_cfg.get("dataset_selection_mode", "configured")).strip().lower() or "configured"
    requested_dataset_id = str(train_cfg.get("dataset_id", "")).strip()
    configured_dataset_path = str(train_cfg.get("dataset_path", "")).strip()
    configured_validation_path = str(train_cfg.get("validation_path", "")).strip()
    planned_epochs = int(train_cfg.get("epochs", 0) or 0)
    batch_size = int(train_cfg.get("batch_size", 0) or 0)

    selected_entry: dict[str, Any] | None = None
    resolved_reason = "unknown"
    if selection_mode == "largest_built":
        selected_entry = _select_largest_built_dataset_entry(registry)
        resolved_reason = "largest_built"
    elif requested_dataset_id and requested_dataset_id not in {"auto"} and not configured_dataset_path and not configured_validation_path:
        selected_entry = _resolve_dataset_entry_by_id(registry, requested_dataset_id)
        resolved_reason = "configured_dataset_id"
    elif configured_dataset_path or configured_validation_path:
        resolved_reason = "configured_paths"
    else:
        resolved_reason = "configured_fallback"

    print(
        "[preflight][train] "
        f"selection_mode={selection_mode} | planned_epochs={planned_epochs} | batch_size={batch_size}",
        flush=True,
    )
    print(
        "[preflight][train] "
        f"registry_path={registry_path}",
        flush=True,
    )

    if selected_entry is not None:
        dataset_id = str(selected_entry.get("dataset_id", "")).strip() or "<unknown>"
        labeled_samples = int(selected_entry.get("labeled_samples", 0) or 0)
        target_labeled_samples = int(selected_entry.get("target_labeled_samples", 0) or 0)
        build_status = str(selected_entry.get("build_status", "")).strip() or "<unknown>"
        output_dir = str(selected_entry.get("output_dir", "")).strip() or "<unknown>"
        splits = selected_entry.get("splits", {})
        train_samples = int(((splits or {}).get("train", {}) or {}).get("samples", 0) or 0)
        validation_samples = int(((splits or {}).get("validation", {}) or {}).get("samples", 0) or 0)
        test_samples = int(((splits or {}).get("test", {}) or {}).get("samples", 0) or 0)
        print(
            "[preflight][train] "
            f"resolved_via={resolved_reason} | dataset_id={dataset_id} | "
            f"labeled_samples={labeled_samples} | target_labeled_samples={target_labeled_samples} | "
            f"build_status={build_status}",
            flush=True,
        )
        print(
            "[preflight][train] "
            f"split_samples(train/val/test)={train_samples}/{validation_samples}/{test_samples} | "
            f"output_dir={output_dir}",
            flush=True,
        )
    else:
        print(
            "[preflight][train] "
            f"resolved_via={resolved_reason} | requested_dataset_id={requested_dataset_id or '<empty>'} | "
            f"dataset_path={configured_dataset_path or '<auto>'} | "
            f"validation_path={configured_validation_path or '<auto>'}",
            flush=True,
        )


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


def _extract_return_code_from_exception(exc: Exception) -> int | None:
    text = str(exc or "")
    match = re.search(r"rc=(-?\d+)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _detect_colab_compute_mode() -> str:
    override = str(os.environ.get("SONGO_COLAB_COMPUTE_MODE", "")).strip().lower()
    if override in {"cpu", "tpu"}:
        return override
    tpu_addr = str(os.environ.get("COLAB_TPU_ADDR", "")).strip()
    if tpu_addr:
        return "tpu"
    return "cpu"


def _apply_benchmark_safe_mode(payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    out.setdefault("benchmark", {})
    benchmark_cfg = dict(out.get("benchmark", {}) or {})
    benchmark_cfg["parallel_enabled"] = False
    benchmark_cfg["parallel_backend"] = "sequential"
    benchmark_cfg["parallel_workers"] = 1
    out["benchmark"] = benchmark_cfg
    return out


def _resolve_benchmark_cpu_worker_cap() -> int:
    raw = str(os.environ.get("SONGO_BENCHMARK_CPU_WORKER_CAP", "2")).strip()
    try:
        value = int(raw)
    except Exception:
        value = 2
    return max(1, value)


def _apply_benchmark_compute_tuning(payload: dict[str, Any], *, compute_mode: str) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    out.setdefault("benchmark", {})
    benchmark_cfg = dict(out.get("benchmark", {}) or {})

    if compute_mode == "cpu":
        cap = _resolve_benchmark_cpu_worker_cap()
        configured_workers = int(benchmark_cfg.get("parallel_workers", cap) or cap)
        tuned_workers = max(1, min(configured_workers, cap))
        benchmark_cfg["parallel_workers"] = tuned_workers
        if tuned_workers <= 1:
            benchmark_cfg["parallel_enabled"] = False
            benchmark_cfg["parallel_backend"] = "sequential"

    out["benchmark"] = benchmark_cfg
    return out


def _resolve_active_config_path(*, worktree: Path, identity: str, command: str) -> Path:
    stem = _CONFIG_STEMS[str(command)]
    return worktree / "config" / "generated" / f"{stem}.{identity}.active.yaml"


def _run_single_command(
    *,
    python_bin: str,
    worktree: Path,
    identity: str,
    command: str,
    config_path: Path | None,
    heartbeat_seconds: int,
) -> None:
    cfg = config_path or _resolve_active_config_path(
        worktree=worktree,
        identity=identity,
        command=command,
    )
    if not cfg.exists():
        raise FileNotFoundError(f"Config introuvable: {cfg}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    env["PYTHONUNBUFFERED"] = "1"
    _run_live(
        [
            python_bin,
            "-u",
            "-m",
            "songo_model_stockfish.cli.main",
            command,
            "--config",
            str(cfg),
        ],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )


def _run_train_eval(
    *,
    python_bin: str,
    worktree: Path,
    identity: str,
    heartbeat_seconds: int,
    train_config: Path | None,
    eval_config: Path | None,
    drive_root: Path,
) -> dict[str, Any]:
    import yaml

    train_cfg = train_config or _resolve_active_config_path(
        worktree=worktree,
        identity=identity,
        command="train",
    )
    eval_cfg = eval_config or _resolve_active_config_path(
        worktree=worktree,
        identity=identity,
        command="evaluate",
    )
    for cfg in [train_cfg, eval_cfg]:
        if not cfg.exists():
            raise FileNotFoundError(f"Config introuvable: {cfg}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    env["PYTHONUNBUFFERED"] = "1"

    _print_training_dataset_preflight(
        train_cfg_path=train_cfg,
        drive_root=drive_root,
        identity=identity,
    )

    _run_live(
        [python_bin, "-u", "-m", "songo_model_stockfish.cli.main", "train", "--config", str(train_cfg)],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )

    registry_path = drive_root / "models" / "model_registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {"models": []}
    )
    models = list(registry.get("models", []))
    if not models:
        raise RuntimeError("Aucun modele trouve dans model_registry apres train.")
    latest = max(models, key=lambda item: float(item.get("sort_ts", 0.0)))
    model_id = str(latest.get("model_id", "")).strip()
    if not model_id:
        raise RuntimeError("model_id vide dans le registre.")
    print("model_id entrainé =", model_id, flush=True)

    eval_payload = yaml.safe_load(eval_cfg.read_text(encoding="utf-8")) or {}
    eval_payload.setdefault("evaluation", {})
    eval_payload["evaluation"]["model_id"] = model_id
    eval_runtime = eval_cfg.with_name(eval_cfg.stem + ".runtime.yaml")
    eval_runtime.write_text(yaml.safe_dump(eval_payload, sort_keys=False), encoding="utf-8")

    _run_live(
        [python_bin, "-u", "-m", "songo_model_stockfish.cli.main", "evaluate", "--config", str(eval_runtime)],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )

    return {
        "train_config": str(train_cfg),
        "eval_runtime_config": str(eval_runtime),
        "model_id": model_id,
    }


def _run_train_eval_benchmark(
    *,
    python_bin: str,
    worktree: Path,
    identity: str,
    heartbeat_seconds: int,
    train_config: Path | None,
    eval_config: Path | None,
    benchmark_config: Path | None,
    drive_root: Path,
) -> dict[str, Any]:
    import yaml

    train_eval_summary = _run_train_eval(
        python_bin=python_bin,
        worktree=worktree,
        identity=identity,
        heartbeat_seconds=heartbeat_seconds,
        train_config=train_config,
        eval_config=eval_config,
        drive_root=drive_root,
    )
    model_id = str(train_eval_summary["model_id"])

    bench_cfg = benchmark_config or _resolve_active_config_path(
        worktree=worktree,
        identity=identity,
        command="benchmark",
    )
    if not bench_cfg.exists():
        raise FileNotFoundError(f"Config introuvable: {bench_cfg}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    env["PYTHONUNBUFFERED"] = "1"

    bench_payload = yaml.safe_load(bench_cfg.read_text(encoding="utf-8")) or {}
    bench_payload.setdefault("benchmark", {})
    bench_payload["benchmark"]["target"] = model_id
    compute_mode = _detect_colab_compute_mode()
    bench_payload = _apply_benchmark_compute_tuning(bench_payload, compute_mode=compute_mode)
    bench_runtime = bench_cfg.with_name(bench_cfg.stem + ".runtime.yaml")
    bench_runtime.write_text(yaml.safe_dump(bench_payload, sort_keys=False), encoding="utf-8")

    runtime_bench_cfg = dict(bench_payload.get("benchmark", {}) or {})
    print(
        "benchmark runtime config | "
        f"compute_mode={compute_mode} | "
        f"parallel_enabled={runtime_bench_cfg.get('parallel_enabled', True)} | "
        f"parallel_backend={runtime_bench_cfg.get('parallel_backend', 'thread')} | "
        f"parallel_workers={runtime_bench_cfg.get('parallel_workers', '<auto>')}",
        flush=True,
    )

    benchmark_runtime_used = bench_runtime
    benchmark_safe_fallback_used = False
    benchmark_safe_fallback_reason = ""
    benchmark_cmd = [python_bin, "-u", "-m", "songo_model_stockfish.cli.main", "benchmark", "--config", str(bench_runtime)]
    try:
        _run_live(
            benchmark_cmd,
            cwd=worktree,
            env=env,
            heartbeat_s=int(heartbeat_seconds),
        )
    except RuntimeError as exc:
        rc = _extract_return_code_from_exception(exc)
        # -9 and 137 are common SIGKILL exits on Colab (often OOM).
        if rc not in {-9, 137}:
            raise
        benchmark_safe_fallback_used = True
        benchmark_safe_fallback_reason = f"benchmark_primary_failed_rc_{rc}"
        print(
            "benchmark primary run killed; fallback safe-mode activated | "
            f"rc={rc} | mode=sequential",
            flush=True,
        )
        safe_payload = _apply_benchmark_safe_mode(bench_payload)
        safe_runtime = bench_cfg.with_name(bench_cfg.stem + ".runtime.safe.yaml")
        safe_runtime.write_text(yaml.safe_dump(safe_payload, sort_keys=False), encoding="utf-8")
        safe_bench_cfg = dict(safe_payload.get("benchmark", {}) or {})
        print(
            "benchmark fallback config | "
            f"parallel_enabled={safe_bench_cfg.get('parallel_enabled', False)} | "
            f"parallel_backend={safe_bench_cfg.get('parallel_backend', 'sequential')} | "
            f"parallel_workers={safe_bench_cfg.get('parallel_workers', 1)}",
            flush=True,
        )
        _run_live(
            [python_bin, "-u", "-m", "songo_model_stockfish.cli.main", "benchmark", "--config", str(safe_runtime)],
            cwd=worktree,
            env=env,
            heartbeat_s=int(heartbeat_seconds),
        )
        benchmark_runtime_used = safe_runtime

    promoted_meta_path = drive_root / "models" / "promoted" / "best" / "metadata.json"
    promoted_meta = (
        json.loads(promoted_meta_path.read_text(encoding="utf-8"))
        if promoted_meta_path.exists()
        else {}
    )
    if promoted_meta:
        print("promoted_model_id =", promoted_meta.get("model_id", "<none>"), flush=True)
        print(
            "promoted_checkpoint =",
            promoted_meta.get("promoted_checkpoint_path", "<none>"),
            flush=True,
        )
    else:
        print("Aucun metadata de promotion trouvé.", flush=True)

    return {
        "train_config": str(train_eval_summary["train_config"]),
        "eval_runtime_config": str(train_eval_summary["eval_runtime_config"]),
        "benchmark_runtime_config": str(benchmark_runtime_used),
        "benchmark_safe_fallback_used": benchmark_safe_fallback_used,
        "benchmark_safe_fallback_reason": benchmark_safe_fallback_reason,
        "model_id": model_id,
        "promoted_meta_path": str(promoted_meta_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "dataset-generate",
            "dataset-build",
            "train",
            "evaluate",
            "benchmark",
            "train-eval",
            "train-eval-benchmark",
        ],
    )
    parser.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    parser.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    parser.add_argument("--python-bin", default=(sys.executable or os.environ.get("PYTHON_BIN", "python3")))
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--config", default="")
    parser.add_argument("--train-config", default="")
    parser.add_argument("--eval-config", default="")
    parser.add_argument("--benchmark-config", default="")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/songo-stockfish")
    args = parser.parse_args()

    worktree = Path(str(args.worktree))
    command = str(args.command)
    identity = str(args.identity)
    python_bin = str(args.python_bin)
    heartbeat_seconds = int(args.heartbeat_seconds)

    if command == "train-eval":
        _run_train_eval(
            python_bin=python_bin,
            worktree=worktree,
            identity=identity,
            heartbeat_seconds=heartbeat_seconds,
            train_config=(Path(str(args.train_config)) if str(args.train_config).strip() else None),
            eval_config=(Path(str(args.eval_config)) if str(args.eval_config).strip() else None),
            drive_root=Path(str(args.drive_root)),
        )
        return 0

    if command == "train-eval-benchmark":
        _run_train_eval_benchmark(
            python_bin=python_bin,
            worktree=worktree,
            identity=identity,
            heartbeat_seconds=heartbeat_seconds,
            train_config=(Path(str(args.train_config)) if str(args.train_config).strip() else None),
            eval_config=(Path(str(args.eval_config)) if str(args.eval_config).strip() else None),
            benchmark_config=(Path(str(args.benchmark_config)) if str(args.benchmark_config).strip() else None),
            drive_root=Path(str(args.drive_root)),
        )
        return 0

    _run_single_command(
        python_bin=python_bin,
        worktree=worktree,
        identity=identity,
        command=command,
        config_path=(Path(str(args.config)) if str(args.config).strip() else None),
        heartbeat_seconds=heartbeat_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
