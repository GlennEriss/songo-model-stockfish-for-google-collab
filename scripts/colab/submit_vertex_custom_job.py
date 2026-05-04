from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


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


def _build_vertex_drive_root(bucket: str, prefix: str) -> str:
    if prefix:
        return f"/gcs/{bucket}/{prefix}"
    return f"/gcs/{bucket}"


def _run_live_capture(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    heartbeat_s: int = 30,
) -> str:
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
    lines: list[str] = []
    last_hb = started
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
        lines.append(line)
        now = time.time()
        if (now - last_hb) >= max(10, int(heartbeat_s)):
            print(f"[heartbeat] elapsed={int(now-started)}s | process_running=True", flush=True)
            last_hb = now
    rc = proc.wait()
    print(f"[exit] returncode={rc} | elapsed={int(time.time()-started)}s", flush=True)
    if rc != 0:
        raise RuntimeError(f"Commande en echec (rc={rc}): {cmd}")
    return "".join(lines)


def _discover_dataset_id_from_gcs_pointer(
    *,
    pointer_uri: str,
    worktree: Path,
    heartbeat_seconds: int,
) -> str:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    output = _run_live_capture(
        ["gcloud", "storage", "cat", pointer_uri],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )
    raw_text = str(output or "").strip()
    try:
        payload = json.loads(raw_text)
    except Exception:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(raw_text[start : end + 1])
        else:
            raise RuntimeError(
                "Lecture du pointeur dataset GCS invalide (JSON non parseable) "
                f"| pointer_uri={pointer_uri}"
            )
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise RuntimeError(f"dataset_id absent dans le pointeur dataset: {pointer_uri}")
    return dataset_id


def _sanitize_label_value(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def _apply_vertex_storage(payload: dict[str, Any], *, vertex_drive_root: str) -> dict[str, Any]:
    out = dict(payload)
    storage_cfg = dict(out.get("storage", {}) or {})
    storage_cfg["drive_root"] = vertex_drive_root
    storage_cfg["jobs_root"] = "jobs"
    storage_cfg["jobs_backup_root"] = "runtime_backup/jobs"
    storage_cfg["logs_root"] = "logs"
    storage_cfg["reports_root"] = "reports"
    storage_cfg["models_root"] = "models"
    storage_cfg["data_root"] = "data"
    storage_cfg["runtime_state_backup_enabled"] = True
    out["storage"] = storage_cfg
    return out


def _ensure_runtime_device(payload: dict[str, Any], *, use_gpu: bool) -> dict[str, Any]:
    out = dict(payload)
    runtime_cfg = dict(out.get("runtime", {}) or {})
    runtime_cfg["device"] = "cuda" if use_gpu else "cpu"
    if not use_gpu:
        runtime_cfg["mixed_precision"] = False
        runtime_cfg["pin_memory"] = False
    out["runtime"] = runtime_cfg
    return out


def _prepare_train_eval_runtime_configs(
    *,
    worktree: Path,
    identity: str,
    dataset_id: str,
    vertex_drive_root: str,
    use_gpu: bool,
) -> tuple[Path, Path]:
    identity_key = str(identity or "").strip() or "unknown_drive_identity"
    train_active = worktree / "config" / "generated" / f"train.{identity_key}.active.yaml"
    eval_active = worktree / "config" / "generated" / f"evaluation.{identity_key}.active.yaml"
    if not train_active.exists():
        raise FileNotFoundError(f"Config train active introuvable: {train_active}")
    if not eval_active.exists():
        raise FileNotFoundError(f"Config evaluation active introuvable: {eval_active}")

    ts = int(time.time())
    out_dir = worktree / "config" / "generated" / "vertex"
    train_runtime_path = out_dir / f"train.{identity_key}.vertex.{ts}.runtime.yaml"
    eval_runtime_path = out_dir / f"evaluation.{identity_key}.vertex.{ts}.runtime.yaml"

    train_payload = _load_yaml(train_active)
    train_payload = _apply_vertex_storage(train_payload, vertex_drive_root=vertex_drive_root)
    train_payload = _ensure_runtime_device(train_payload, use_gpu=use_gpu)
    train_payload.setdefault("job", {})
    train_payload["job"]["run_type"] = "train"
    train_payload["job"]["resume"] = True
    train_payload["job"]["job_id"] = f"train_vertex_{identity_key}_{ts}"
    train_payload.setdefault("train", {})
    train_payload["train"]["dataset_selection_mode"] = "configured"
    train_payload["train"]["dataset_id"] = dataset_id
    train_payload["train"]["dataset_path"] = f"data/datasets/{dataset_id}/train.npz"
    train_payload["train"]["validation_path"] = f"data/datasets/{dataset_id}/validation.npz"
    _save_yaml(train_runtime_path, train_payload)

    eval_payload = _load_yaml(eval_active)
    eval_payload = _apply_vertex_storage(eval_payload, vertex_drive_root=vertex_drive_root)
    eval_payload = _ensure_runtime_device(eval_payload, use_gpu=use_gpu)
    eval_payload.setdefault("job", {})
    eval_payload["job"]["run_type"] = "evaluation"
    eval_payload["job"]["resume"] = True
    eval_payload["job"]["job_id"] = f"eval_vertex_{identity_key}_{ts}"
    eval_payload.setdefault("evaluation", {})
    eval_payload["evaluation"]["dataset_selection_mode"] = "configured"
    eval_payload["evaluation"]["dataset_id"] = dataset_id
    eval_payload["evaluation"]["test_dataset_path"] = f"data/datasets/{dataset_id}/test.npz"
    eval_payload["evaluation"]["model_id"] = "auto_latest"
    _save_yaml(eval_runtime_path, eval_payload)

    return train_runtime_path, eval_runtime_path


def _prepare_benchmark_runtime_config(
    *,
    worktree: Path,
    identity: str,
    vertex_drive_root: str,
    use_gpu: bool,
    benchmark_target: str,
) -> Path:
    identity_key = str(identity or "").strip() or "unknown_drive_identity"
    benchmark_active = worktree / "config" / "generated" / f"benchmark.{identity_key}.active.yaml"
    if not benchmark_active.exists():
        raise FileNotFoundError(f"Config benchmark active introuvable: {benchmark_active}")

    ts = int(time.time())
    out_dir = worktree / "config" / "generated" / "vertex"
    bench_runtime_path = out_dir / f"benchmark.{identity_key}.vertex.{ts}.runtime.yaml"

    bench_payload = _load_yaml(benchmark_active)
    bench_payload = _apply_vertex_storage(bench_payload, vertex_drive_root=vertex_drive_root)
    bench_payload = _ensure_runtime_device(bench_payload, use_gpu=use_gpu)
    bench_payload.setdefault("job", {})
    bench_payload["job"]["run_type"] = "benchmark"
    bench_payload["job"]["resume"] = True
    bench_payload["job"]["job_id"] = f"benchmark_vertex_{identity_key}_{ts}"
    bench_payload.setdefault("benchmark", {})
    bench_payload["benchmark"]["target"] = str(benchmark_target or "auto_latest")
    _save_yaml(bench_runtime_path, bench_payload)

    return bench_runtime_path


def _build_worker_pool_spec(
    *,
    worktree: Path,
    machine_type: str,
    executor_image_uri: str,
    accelerator_type: str,
    accelerator_count: int,
) -> str:
    fields = [
        f"machine-type={machine_type}",
        "replica-count=1",
        f"executor-image-uri={executor_image_uri}",
        f"local-package-path={worktree}",
        "script=scripts/colab/notebook_step.py",
    ]
    accel_type = str(accelerator_type or "").strip()
    accel_count = int(accelerator_count)
    if accel_type and accel_count > 0:
        fields.append(f"accelerator-type={accel_type}")
        fields.append(f"accelerator-count={accel_count}")
    return ",".join(fields)


def _extract_custom_job_id(create_output: str) -> str:
    lines = [line.strip() for line in str(create_output or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Impossible d'extraire l'id du custom job Vertex (sortie vide).")
    candidate = lines[-1]
    if "/" in candidate:
        candidate = candidate.split("/")[-1]
    return candidate


def _to_rel_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root.resolve(strict=False)))
    except Exception:
        return str(path)


def _submit_custom_job(
    *,
    worktree: Path,
    project_id: str,
    region: str,
    display_name: str,
    worker_pool_spec: str,
    args_list: list[str],
    service_account: str,
    labels: dict[str, str],
    heartbeat_seconds: int,
) -> str:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    labels_arg = ",".join([f"{k}={v}" for k, v in labels.items() if k and v])
    args_csv = ",".join(args_list)

    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--project={project_id}",
        f"--region={region}",
        f"--display-name={display_name}",
        f"--worker-pool-spec={worker_pool_spec}",
        f"--args={args_csv}",
        "--format=value(name)",
    ]
    if labels_arg:
        cmd.append(f"--labels={labels_arg}")
    if str(service_account or "").strip():
        cmd.append(f"--service-account={service_account}")

    output = _run_live_capture(
        cmd,
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )
    job_id = _extract_custom_job_id(output)
    print(f"Vertex custom job submitted | display_name={display_name} | job_id={job_id}", flush=True)
    return job_id


def _stream_custom_job_logs(
    *,
    worktree: Path,
    project_id: str,
    region: str,
    job_id: str,
    heartbeat_seconds: int,
) -> None:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    _run_live_capture(
        [
            "gcloud",
            "ai",
            "custom-jobs",
            "stream-logs",
            str(job_id),
            f"--project={project_id}",
            f"--region={region}",
            "--polling-interval=30",
        ],
        cwd=worktree,
        env=env,
        heartbeat_s=max(30, int(heartbeat_seconds)),
    )


def run_submit_vertex_custom_job(
    *,
    command: str,
    worktree: Path,
    identity: str,
    project_id: str,
    region: str,
    gcs_bucket: str,
    gcs_prefix: str,
    dataset_id: str,
    dataset_pointer_uri: str,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    executor_image_uri: str,
    service_account: str,
    benchmark_target: str,
    stream_logs: bool,
    heartbeat_seconds: int,
) -> dict[str, Any]:
    if not str(project_id or "").strip():
        raise ValueError("project_id vide. Configure --project-id ou SONGO_VERTEX_PROJECT_ID.")

    bucket = _normalize_bucket(gcs_bucket)
    prefix = _normalize_prefix(gcs_prefix)
    command_name = str(command or "").strip()
    identity_key = str(identity or "").strip() or "unknown_drive_identity"

    resolved_dataset_id = str(dataset_id or "").strip()
    if not resolved_dataset_id:
        pointer_uri = str(dataset_pointer_uri or "").strip()
        if not pointer_uri:
            pointer_uri = _build_gs_uri(bucket, "data", "datasets", "merged", "latest.json", prefix=prefix)
        resolved_dataset_id = _discover_dataset_id_from_gcs_pointer(
            pointer_uri=pointer_uri,
            worktree=worktree,
            heartbeat_seconds=int(heartbeat_seconds),
        )

    use_gpu = bool(str(accelerator_type or "").strip()) and int(accelerator_count) > 0
    vertex_drive_root = _build_vertex_drive_root(bucket, prefix)

    worker_pool_spec = _build_worker_pool_spec(
        worktree=worktree.resolve(strict=False),
        machine_type=str(machine_type),
        executor_image_uri=str(executor_image_uri),
        accelerator_type=str(accelerator_type),
        accelerator_count=int(accelerator_count),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    labels = {
        "pipeline": "songo",
        "job_kind": _sanitize_label_value(command_name),
        "identity": _sanitize_label_value(identity_key),
    }

    if command_name == "train-eval":
        train_runtime_path, eval_runtime_path = _prepare_train_eval_runtime_configs(
            worktree=worktree,
            identity=identity_key,
            dataset_id=resolved_dataset_id,
            vertex_drive_root=vertex_drive_root,
            use_gpu=use_gpu,
        )
        args_list = [
            "run-job",
            "train-eval",
            "--worktree",
            ".",
            "--identity",
            identity_key,
            "--python-bin",
            "python3",
            "--heartbeat-seconds",
            str(int(heartbeat_seconds)),
            "--drive-root",
            vertex_drive_root,
            "--train-config",
            _to_rel_path(train_runtime_path, root=worktree),
            "--eval-config",
            _to_rel_path(eval_runtime_path, root=worktree),
        ]
        display_name = f"songo-train-eval-{identity_key}-{ts}"
        custom_job_id = _submit_custom_job(
            worktree=worktree,
            project_id=str(project_id),
            region=str(region),
            display_name=display_name,
            worker_pool_spec=worker_pool_spec,
            args_list=args_list,
            service_account=str(service_account),
            labels=labels,
            heartbeat_seconds=int(heartbeat_seconds),
        )
        if bool(stream_logs):
            _stream_custom_job_logs(
                worktree=worktree,
                project_id=str(project_id),
                region=str(region),
                job_id=str(custom_job_id),
                heartbeat_seconds=int(heartbeat_seconds),
            )
        return {
            "command": command_name,
            "custom_job_id": custom_job_id,
            "display_name": display_name,
            "project_id": str(project_id),
            "region": str(region),
            "dataset_id": resolved_dataset_id,
            "vertex_drive_root": vertex_drive_root,
            "runtime_configs": {
                "train": str(train_runtime_path),
                "evaluation": str(eval_runtime_path),
            },
            "worker_pool_spec": worker_pool_spec,
            "stream_logs": bool(stream_logs),
        }

    if command_name == "benchmark":
        benchmark_runtime_path = _prepare_benchmark_runtime_config(
            worktree=worktree,
            identity=identity_key,
            vertex_drive_root=vertex_drive_root,
            use_gpu=use_gpu,
            benchmark_target=str(benchmark_target or "auto_latest"),
        )
        args_list = [
            "run-job",
            "benchmark",
            "--worktree",
            ".",
            "--identity",
            identity_key,
            "--python-bin",
            "python3",
            "--heartbeat-seconds",
            str(int(heartbeat_seconds)),
            "--drive-root",
            vertex_drive_root,
            "--config",
            _to_rel_path(benchmark_runtime_path, root=worktree),
        ]
        display_name = f"songo-benchmark-{identity_key}-{ts}"
        custom_job_id = _submit_custom_job(
            worktree=worktree,
            project_id=str(project_id),
            region=str(region),
            display_name=display_name,
            worker_pool_spec=worker_pool_spec,
            args_list=args_list,
            service_account=str(service_account),
            labels=labels,
            heartbeat_seconds=int(heartbeat_seconds),
        )
        if bool(stream_logs):
            _stream_custom_job_logs(
                worktree=worktree,
                project_id=str(project_id),
                region=str(region),
                job_id=str(custom_job_id),
                heartbeat_seconds=int(heartbeat_seconds),
            )
        return {
            "command": command_name,
            "custom_job_id": custom_job_id,
            "display_name": display_name,
            "project_id": str(project_id),
            "region": str(region),
            "dataset_id": resolved_dataset_id,
            "vertex_drive_root": vertex_drive_root,
            "runtime_configs": {
                "benchmark": str(benchmark_runtime_path),
            },
            "worker_pool_spec": worker_pool_spec,
            "stream_logs": bool(stream_logs),
        }

    raise ValueError(f"Commande Vertex non supportee: {command_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train-eval", "benchmark"])
    parser.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    parser.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    parser.add_argument("--project-id", default=(str(os.environ.get("SONGO_VERTEX_PROJECT_ID", "")).strip()))
    parser.add_argument("--region", default=(str(os.environ.get("SONGO_VERTEX_REGION", "us-central1")).strip()))
    parser.add_argument("--gcs-bucket", default=(str(os.environ.get("SONGO_VERTEX_GCS_BUCKET", "")).strip()))
    parser.add_argument("--gcs-prefix", default=(str(os.environ.get("SONGO_VERTEX_GCS_PREFIX", "songo-stockfish")).strip()))
    parser.add_argument("--dataset-id", default=(str(os.environ.get("SONGO_VERTEX_DATASET_ID", "")).strip()))
    parser.add_argument("--dataset-pointer-uri", default=(str(os.environ.get("SONGO_VERTEX_DATASET_POINTER_URI", "")).strip()))
    parser.add_argument("--machine-type", default=(str(os.environ.get("SONGO_VERTEX_MACHINE_TYPE", "n1-standard-8")).strip()))
    parser.add_argument(
        "--accelerator-type",
        default=(str(os.environ.get("SONGO_VERTEX_ACCELERATOR_TYPE", "NVIDIA_TESLA_T4")).strip()),
    )
    parser.add_argument("--accelerator-count", type=int, default=int(os.environ.get("SONGO_VERTEX_ACCELERATOR_COUNT", "1")))
    parser.add_argument(
        "--executor-image-uri",
        default=(
            str(
                os.environ.get(
                    "SONGO_VERTEX_EXECUTOR_IMAGE_URI",
                    "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
                )
            ).strip()
        ),
    )
    parser.add_argument("--service-account", default=(str(os.environ.get("SONGO_VERTEX_SERVICE_ACCOUNT", "")).strip()))
    parser.add_argument("--benchmark-target", default=(str(os.environ.get("SONGO_VERTEX_BENCHMARK_TARGET", "auto_latest")).strip()))
    parser.add_argument("--stream-logs", dest="stream_logs", action="store_true", default=False)
    parser.add_argument("--no-stream-logs", dest="stream_logs", action="store_false")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    summary = run_submit_vertex_custom_job(
        command=str(args.command),
        worktree=Path(str(args.worktree)),
        identity=str(args.identity),
        project_id=str(args.project_id),
        region=str(args.region),
        gcs_bucket=str(args.gcs_bucket),
        gcs_prefix=str(args.gcs_prefix),
        dataset_id=str(args.dataset_id),
        dataset_pointer_uri=str(args.dataset_pointer_uri),
        machine_type=str(args.machine_type),
        accelerator_type=str(args.accelerator_type),
        accelerator_count=int(args.accelerator_count),
        executor_image_uri=str(args.executor_image_uri),
        service_account=str(args.service_account),
        benchmark_target=str(args.benchmark_target),
        stream_logs=bool(args.stream_logs),
        heartbeat_seconds=int(args.heartbeat_seconds),
    )

    summary_path = str(args.summary_path or "").strip()
    if summary_path:
        _write_json(Path(summary_path), summary)
    if bool(args.print_json):
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
