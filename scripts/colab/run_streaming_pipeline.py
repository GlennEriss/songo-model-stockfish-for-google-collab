from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CONFIG_STEMS = {
    "dataset-generate": "dataset_generation",
    "dataset-build": "dataset_build",
}


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
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


def _resolve_active_config_path(*, worktree: Path, identity: str, command: str) -> Path:
    stem = _CONFIG_STEMS[command]
    return worktree / "config" / "generated" / f"{stem}.{identity}.active.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_storage_root(*, drive_root: Path, configured: Any, fallback_relative: str) -> Path:
    text = str(configured if configured is not None else "").strip()
    rel_or_abs = Path(text) if text else Path(fallback_relative)
    if rel_or_abs.is_absolute():
        return rel_or_abs
    return drive_root / rel_or_abs


def _read_dataset_progress(*, registry_path: Path, dataset_id: str) -> dict[str, Any]:
    registry = _read_json(registry_path, {"dataset_sources": [], "built_datasets": []})
    built_entries = registry.get("built_datasets", [])
    if not isinstance(built_entries, list):
        built_entries = []
    for entry in built_entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("dataset_id", "")).strip() != dataset_id:
            continue
        return {
            "dataset_id": dataset_id,
            "labeled_samples": int(entry.get("labeled_samples", 0) or 0),
            "target_labeled_samples": int(entry.get("target_labeled_samples", 0) or 0),
            "build_status": str(entry.get("build_status", "")).strip().lower(),
            "updated_at": str(entry.get("updated_at", "")).strip(),
        }
    return {
        "dataset_id": dataset_id,
        "labeled_samples": 0,
        "target_labeled_samples": 0,
        "build_status": "",
        "updated_at": "",
    }


def _stream_output(prefix: str, stream, output_q: queue.Queue[tuple[str, str]]) -> None:
    for raw_line in stream:
        output_q.put((prefix, raw_line.rstrip("\n")))


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    cwd: Path
    env: dict[str, str]
    output_q: queue.Queue[tuple[str, str]]
    started_at: float = 0.0
    popen: subprocess.Popen[str] | None = None
    _thread: threading.Thread | None = None
    _exit_reported: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        print(f"RUN[{self.name}]: {self.command}")
        self.started_at = time.time()
        self.popen = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert self.popen.stdout is not None
        self._thread = threading.Thread(
            target=_stream_output,
            args=(self.name, self.popen.stdout, self.output_q),
            daemon=True,
        )
        self._thread.start()

    def poll(self) -> int | None:
        return self.popen.poll() if self.popen is not None else None

    def is_running(self) -> bool:
        return self.poll() is None

    def report_exit_once(self) -> int | None:
        rc = self.poll()
        if rc is None:
            return None
        if not self._exit_reported:
            self._exit_reported = True
            elapsed = int(max(0.0, time.time() - self.started_at))
            print(f"[exit:{self.name}] returncode={rc} | elapsed={elapsed}s")
        return rc

    def terminate(self, *, grace_seconds: float = 10.0) -> None:
        if self.popen is None:
            return
        if self.poll() is not None:
            return
        self.popen.terminate()
        deadline = time.time() + max(1.0, float(grace_seconds))
        while time.time() < deadline:
            if self.poll() is not None:
                return
            time.sleep(0.2)
        if self.poll() is None:
            self.popen.kill()


def _drain_output(output_q: queue.Queue[tuple[str, str]]) -> None:
    while True:
        try:
            prefix, line = output_q.get_nowait()
        except queue.Empty:
            return
        print(f"[{prefix}] {line}")


def _should_start_training(
    *,
    labeled_samples: int,
    last_trained_labeled_samples: int,
    train_proc: ManagedProcess | None,
    min_samples: int,
    min_delta_samples: int,
    train_runs: int,
    max_train_runs: int,
) -> bool:
    if train_proc is not None and train_proc.is_running():
        return False
    if max_train_runs > 0 and train_runs >= max_train_runs:
        return False
    if labeled_samples < min_samples:
        return False
    if last_trained_labeled_samples <= 0:
        return True
    return (labeled_samples - last_trained_labeled_samples) >= min_delta_samples


def _build_job_command(
    *,
    python_bin: str,
    worktree: Path,
    identity: str,
    command: str,
    heartbeat_seconds: int,
    drive_root: Path,
) -> list[str]:
    cmd = [
        python_bin,
        str(worktree / "scripts" / "colab" / "run_job.py"),
        command,
        "--worktree",
        str(worktree),
        "--identity",
        identity,
        "--heartbeat-seconds",
        str(int(heartbeat_seconds)),
    ]
    if command == "train-eval-benchmark":
        cmd.extend(["--drive-root", str(drive_root)])
    return cmd


def run_streaming_pipeline(
    *,
    python_bin: str,
    worktree: Path,
    identity: str,
    drive_root: Path,
    heartbeat_seconds: int,
    poll_seconds: float,
    train_min_samples: int,
    train_min_delta_samples: int,
    max_train_runs: int,
    disable_auto_train: bool,
    continue_on_train_error: bool,
    skip_generate: bool,
    skip_build: bool,
    state_path: Path | None,
) -> dict[str, Any]:
    identity_key = str(identity or "").strip() or "unknown_drive_identity"
    build_cfg_path = _resolve_active_config_path(
        worktree=worktree,
        identity=identity_key,
        command="dataset-build",
    )
    if not build_cfg_path.exists():
        raise FileNotFoundError(f"Config introuvable: {build_cfg_path}")
    build_cfg = _load_yaml(build_cfg_path)
    dataset_build_cfg = dict(build_cfg.get("dataset_build", {}) or {})
    storage_cfg = dict(build_cfg.get("storage", {}) or {})
    dataset_id = str(dataset_build_cfg.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("dataset_build.dataset_id introuvable dans la config active.")

    data_root = _resolve_storage_root(
        drive_root=drive_root,
        configured=storage_cfg.get("data_root"),
        fallback_relative=f"{identity_key}/data",
    )
    jobs_root = _resolve_storage_root(
        drive_root=drive_root,
        configured=storage_cfg.get("jobs_root"),
        fallback_relative=f"{identity_key}/jobs",
    )
    registry_path = data_root / "dataset_registry.json"
    pipeline_state_path = state_path or (jobs_root / "pipeline_streaming_state.json")
    pipeline_state = _read_json(
        pipeline_state_path,
        {
            "last_trained_labeled_samples": 0,
            "train_runs": 0,
            "last_train_model_id": "",
            "updated_at": "",
        },
    )
    last_trained_labeled_samples = int(pipeline_state.get("last_trained_labeled_samples", 0) or 0)
    train_runs = int(pipeline_state.get("train_runs", 0) or 0)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(worktree / "src")

    output_q: queue.Queue[tuple[str, str]] = queue.Queue()
    base_processes: list[ManagedProcess] = []

    if not skip_generate:
        gen_proc = ManagedProcess(
            name="dataset-generate",
            command=_build_job_command(
                python_bin=python_bin,
                worktree=worktree,
                identity=identity_key,
                command="dataset-generate",
                heartbeat_seconds=heartbeat_seconds,
                drive_root=drive_root,
            ),
            cwd=worktree,
            env=env,
            output_q=output_q,
        )
        gen_proc.start()
        base_processes.append(gen_proc)

    if not skip_build:
        build_proc = ManagedProcess(
            name="dataset-build",
            command=_build_job_command(
                python_bin=python_bin,
                worktree=worktree,
                identity=identity_key,
                command="dataset-build",
                heartbeat_seconds=heartbeat_seconds,
                drive_root=drive_root,
            ),
            cwd=worktree,
            env=env,
            output_q=output_q,
        )
        build_proc.start()
        base_processes.append(build_proc)

    train_proc: ManagedProcess | None = None
    last_heartbeat = 0.0
    started = time.time()

    print(
        "streaming pipeline started | "
        f"identity={identity_key} | dataset_id={dataset_id} | "
        f"disable_auto_train={bool(disable_auto_train)} | "
        f"train_min_samples={int(train_min_samples)} | "
        f"train_min_delta_samples={int(train_min_delta_samples)} | "
        f"max_train_runs={int(max_train_runs)}"
    )
    print(f"registry_path={registry_path}")
    print(f"state_path={pipeline_state_path}")

    try:
        while True:
            _drain_output(output_q)

            for proc in base_processes:
                rc = proc.report_exit_once()
                if rc is not None and rc != 0:
                    if train_proc is not None:
                        train_proc.terminate()
                    for other in base_processes:
                        if other is not proc:
                            other.terminate()
                    raise RuntimeError(f"Processus en echec: {proc.name} (rc={rc})")

            progress = _read_dataset_progress(registry_path=registry_path, dataset_id=dataset_id)
            labeled_samples = int(progress.get("labeled_samples", 0) or 0)
            build_status = str(progress.get("build_status", "")).strip().lower()

            if (not disable_auto_train) and _should_start_training(
                labeled_samples=labeled_samples,
                last_trained_labeled_samples=last_trained_labeled_samples,
                train_proc=train_proc,
                min_samples=max(0, int(train_min_samples)),
                min_delta_samples=max(1, int(train_min_delta_samples)),
                train_runs=train_runs,
                max_train_runs=max(0, int(max_train_runs)),
            ):
                train_proc = ManagedProcess(
                    name="train-eval-benchmark",
                    command=_build_job_command(
                        python_bin=python_bin,
                        worktree=worktree,
                        identity=identity_key,
                        command="train-eval-benchmark",
                        heartbeat_seconds=heartbeat_seconds,
                        drive_root=drive_root,
                    ),
                    cwd=worktree,
                    env=env,
                    output_q=output_q,
                    metadata={"started_labeled_samples": labeled_samples},
                )
                train_proc.start()
                print(
                    "training cycle started | "
                    f"run_index={train_runs + 1} | "
                    f"started_labeled_samples={labeled_samples}"
                )

            if train_proc is not None:
                train_rc = train_proc.report_exit_once()
                if train_rc is not None:
                    if train_rc == 0:
                        started_labeled_samples = int(train_proc.metadata.get("started_labeled_samples", 0) or 0)
                        last_trained_labeled_samples = max(last_trained_labeled_samples, started_labeled_samples)
                        train_runs += 1
                        latest_progress = _read_dataset_progress(registry_path=registry_path, dataset_id=dataset_id)
                        latest_labeled = int(latest_progress.get("labeled_samples", 0) or 0)
                        print(
                            "training cycle completed | "
                            f"run_index={train_runs} | "
                            f"started_labeled_samples={started_labeled_samples} | "
                            f"current_labeled_samples={latest_labeled}"
                        )
                        _write_json(
                            pipeline_state_path,
                            {
                                "last_trained_labeled_samples": int(last_trained_labeled_samples),
                                "train_runs": int(train_runs),
                                "last_train_model_id": "",
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            },
                        )
                    else:
                        message = f"train-eval-benchmark en echec (rc={train_rc})"
                        if not continue_on_train_error:
                            for proc in base_processes:
                                proc.terminate()
                            raise RuntimeError(message)
                        print(f"WARNING: {message} | continue_on_train_error=True")
                    train_proc = None

            now = time.time()
            if (now - last_heartbeat) >= max(10.0, float(heartbeat_seconds)):
                base_status = []
                for proc in base_processes:
                    rc = proc.poll()
                    if rc is None:
                        base_status.append(f"{proc.name}:running")
                    else:
                        base_status.append(f"{proc.name}:exit({rc})")
                train_state = "idle"
                if train_proc is not None:
                    rc = train_proc.poll()
                    train_state = "running" if rc is None else f"exit({rc})"
                print(
                    "[orchestrator-heartbeat] "
                    f"elapsed={int(now-started)}s | "
                    f"labeled_samples={labeled_samples} | "
                    f"build_status={build_status or '<none>'} | "
                    f"last_trained_labeled_samples={last_trained_labeled_samples} | "
                    f"train_runs={train_runs} | "
                    f"train_state={train_state} | "
                    f"base={' | '.join(base_status) if base_status else '<none>'}"
                )
                last_heartbeat = now

            base_done = all(proc.poll() is not None for proc in base_processes) if base_processes else True
            train_running = train_proc is not None and train_proc.is_running()
            can_start_more_training = False
            if not disable_auto_train:
                can_start_more_training = _should_start_training(
                    labeled_samples=labeled_samples,
                    last_trained_labeled_samples=last_trained_labeled_samples,
                    train_proc=train_proc,
                    min_samples=max(0, int(train_min_samples)),
                    min_delta_samples=max(1, int(train_min_delta_samples)),
                    train_runs=train_runs,
                    max_train_runs=max(0, int(max_train_runs)),
                )
            if base_done and (not train_running) and (not can_start_more_training):
                break

            time.sleep(max(1.0, float(poll_seconds)))
    except KeyboardInterrupt:
        print("interrupt received | stopping child processes...")
        if train_proc is not None:
            train_proc.terminate()
        for proc in base_processes:
            proc.terminate()
        raise
    finally:
        _drain_output(output_q)

    elapsed = int(max(0.0, time.time() - started))
    final_progress = _read_dataset_progress(registry_path=registry_path, dataset_id=dataset_id)
    summary = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "identity": identity_key,
        "dataset_id": dataset_id,
        "registry_path": str(registry_path),
        "state_path": str(pipeline_state_path),
        "final_labeled_samples": int(final_progress.get("labeled_samples", 0) or 0),
        "final_build_status": str(final_progress.get("build_status", "")).strip(),
        "disable_auto_train": bool(disable_auto_train),
        "last_trained_labeled_samples": int(last_trained_labeled_samples),
        "train_runs": int(train_runs),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    parser.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    parser.add_argument("--python-bin", default=(sys.executable or os.environ.get("PYTHON_BIN", "python3")))
    parser.add_argument("--drive-root", default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"))
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--train-min-samples", type=int, default=int(os.environ.get("SONGO_STREAM_TRAIN_MIN_SAMPLES", "50000")))
    parser.add_argument(
        "--train-min-delta-samples",
        type=int,
        default=int(os.environ.get("SONGO_STREAM_TRAIN_MIN_DELTA_SAMPLES", "50000")),
    )
    parser.add_argument("--max-train-runs", type=int, default=int(os.environ.get("SONGO_STREAM_MAX_TRAIN_RUNS", "0")))
    parser.add_argument("--disable-auto-train", action="store_true")
    parser.add_argument("--continue-on-train-error", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--state-path", default="")
    parser.add_argument("--summary-path", default="")
    args = parser.parse_args()

    summary = run_streaming_pipeline(
        python_bin=str(args.python_bin),
        worktree=Path(str(args.worktree)),
        identity=str(args.identity),
        drive_root=Path(str(args.drive_root)),
        heartbeat_seconds=int(args.heartbeat_seconds),
        poll_seconds=float(args.poll_seconds),
        train_min_samples=int(args.train_min_samples),
        train_min_delta_samples=int(args.train_min_delta_samples),
        max_train_runs=int(args.max_train_runs),
        disable_auto_train=bool(args.disable_auto_train),
        continue_on_train_error=bool(args.continue_on_train_error),
        skip_generate=bool(args.skip_generate),
        skip_build=bool(args.skip_build),
        state_path=(Path(str(args.state_path)) if str(args.state_path).strip() else None),
    )

    if str(args.summary_path or "").strip():
        _write_json(Path(str(args.summary_path)), summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
