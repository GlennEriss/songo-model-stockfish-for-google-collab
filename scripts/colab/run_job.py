from __future__ import annotations

import argparse
import json
import os
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
    bench_cfg = benchmark_config or _resolve_active_config_path(
        worktree=worktree,
        identity=identity,
        command="benchmark",
    )

    for cfg in [train_cfg, eval_cfg, bench_cfg]:
        if not cfg.exists():
            raise FileNotFoundError(f"Config introuvable: {cfg}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")
    env["PYTHONUNBUFFERED"] = "1"

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

    bench_payload = yaml.safe_load(bench_cfg.read_text(encoding="utf-8")) or {}
    bench_payload.setdefault("benchmark", {})
    bench_payload["benchmark"]["target"] = model_id
    bench_runtime = bench_cfg.with_name(bench_cfg.stem + ".runtime.yaml")
    bench_runtime.write_text(yaml.safe_dump(bench_payload, sort_keys=False), encoding="utf-8")

    _run_live(
        [python_bin, "-u", "-m", "songo_model_stockfish.cli.main", "benchmark", "--config", str(bench_runtime)],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )

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
        "train_config": str(train_cfg),
        "eval_runtime_config": str(eval_runtime),
        "benchmark_runtime_config": str(bench_runtime),
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
