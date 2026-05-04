from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _to_local_path(value: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Chemin config vide.")
    if text.startswith("gs://"):
        suffix = text[len("gs://") :].strip().strip("/")
        if not suffix:
            raise ValueError(f"URI GCS invalide: {text}")
        return Path("/gcs") / suffix
    return Path(text)


def _run_live(cmd: list[str], *, heartbeat_seconds: int) -> None:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print("RUN:", cmd, flush=True)
    started = time.time()
    proc = subprocess.Popen(
        cmd,
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
        if (now - last_hb) >= max(10, int(heartbeat_seconds)):
            print(f"[heartbeat] elapsed={int(now-started)}s | process_running=True", flush=True)
            last_hb = now

    rc = proc.wait()
    print(f"[exit] returncode={rc} | elapsed={int(time.time()-started)}s", flush=True)
    if rc != 0:
        raise RuntimeError(f"Commande en echec (rc={rc}): {cmd}")


def _run_cli_command(run_type: str, config_path: Path, *, heartbeat_seconds: int) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")
    py = str(sys.executable or "python3")
    cmd = [py, "-u", "-m", "songo_model_stockfish.cli.main", run_type, "--config", str(config_path)]
    _run_live(cmd, heartbeat_seconds=heartbeat_seconds)


def _run_train_eval(train_config: Path, eval_config: Path, *, heartbeat_seconds: int) -> None:
    print(f"vertex entrypoint | train_config={train_config}", flush=True)
    print(f"vertex entrypoint | eval_config={eval_config}", flush=True)
    _run_cli_command("train", train_config, heartbeat_seconds=heartbeat_seconds)
    _run_cli_command("evaluate", eval_config, heartbeat_seconds=heartbeat_seconds)


def _run_benchmark(config_path: Path, *, heartbeat_seconds: int) -> None:
    print(f"vertex entrypoint | benchmark_config={config_path}", flush=True)
    _run_cli_command("benchmark", config_path, heartbeat_seconds=heartbeat_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train-eval", "benchmark"])
    parser.add_argument("--train-config", default="")
    parser.add_argument("--eval-config", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args(argv)

    heartbeat_seconds = max(10, int(args.heartbeat_seconds))
    command = str(args.command)

    if command == "train-eval":
        train_path = _to_local_path(str(args.train_config))
        eval_path = _to_local_path(str(args.eval_config))
        _run_train_eval(train_path, eval_path, heartbeat_seconds=heartbeat_seconds)
        return 0

    if command == "benchmark":
        cfg_path = _to_local_path(str(args.config))
        _run_benchmark(cfg_path, heartbeat_seconds=heartbeat_seconds)
        return 0

    raise ValueError(f"Commande non supportee: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
