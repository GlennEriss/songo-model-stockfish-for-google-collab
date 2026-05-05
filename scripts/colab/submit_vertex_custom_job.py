from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
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


def _assert_gcs_uri_exists(
    *,
    worktree: Path,
    gs_uri: str,
    heartbeat_seconds: int,
) -> None:
    uri = str(gs_uri or "").strip()
    if not uri:
        raise ValueError("URI GCS vide pendant la verification preflight.")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        _run_live_capture(
            ["gcloud", "storage", "ls", uri],
            cwd=worktree,
            env=env,
            heartbeat_s=int(heartbeat_seconds),
        )
    except Exception as exc:
        raise RuntimeError(
            "Preflight Vertex echec: artefact GCS introuvable ou inaccessible "
            f"| uri={uri} | cause={type(exc).__name__}: {exc}"
        ) from exc


def _verify_train_eval_dataset_artifacts(
    *,
    worktree: Path,
    bucket: str,
    prefix: str,
    dataset_id: str,
    heartbeat_seconds: int,
) -> None:
    required_uris = [
        _build_gs_uri(bucket, "data", "datasets", dataset_id, "train.npz", prefix=prefix),
        _build_gs_uri(bucket, "data", "datasets", dataset_id, "validation.npz", prefix=prefix),
        _build_gs_uri(bucket, "data", "datasets", dataset_id, "test.npz", prefix=prefix),
    ]
    for uri in required_uris:
        _assert_gcs_uri_exists(
            worktree=worktree,
            gs_uri=uri,
            heartbeat_seconds=int(heartbeat_seconds),
        )


def _gcs_object_exists(
    *,
    worktree: Path,
    object_uri: str,
) -> bool:
    uri = str(object_uri or "").strip()
    if not uri:
        return False
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = ["gcloud", "storage", "ls", uri]
    print("RUN:", cmd, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(worktree),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output = str(proc.stdout or "").strip()
    if output:
        print(output, flush=True)
    return proc.returncode == 0


def _ensure_gcs_object_text(
    *,
    worktree: Path,
    destination_uri: str,
    content: str,
    heartbeat_seconds: int,
) -> bool:
    if _gcs_object_exists(worktree=worktree, object_uri=destination_uri):
        return False
    with tempfile.NamedTemporaryFile(prefix="songo_gcs_seed_", suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        tmp_path.write_text(str(content or ""), encoding="utf-8")
        _upload_file_to_gcs(
            worktree=worktree,
            local_path=tmp_path,
            destination_uri=destination_uri,
            heartbeat_seconds=int(heartbeat_seconds),
        )
        return True
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _ensure_vertex_models_layout(
    *,
    worktree: Path,
    bucket: str,
    prefix: str,
    heartbeat_seconds: int,
) -> dict[str, Any]:
    created_objects: list[str] = []

    registry_uri = _build_gs_uri(bucket, "models", "model_registry.json", prefix=prefix)
    if _ensure_gcs_object_text(
        worktree=worktree,
        destination_uri=registry_uri,
        content=json.dumps({"models": []}, indent=2, ensure_ascii=True),
        heartbeat_seconds=int(heartbeat_seconds),
    ):
        created_objects.append(registry_uri)

    keep_uris = [
        _build_gs_uri(bucket, "models", ".keep", prefix=prefix),
        _build_gs_uri(bucket, "models", "final", ".keep", prefix=prefix),
        _build_gs_uri(bucket, "models", "checkpoints", ".keep", prefix=prefix),
        _build_gs_uri(bucket, "models", "lineage", ".keep", prefix=prefix),
        _build_gs_uri(bucket, "models", "promoted", ".keep", prefix=prefix),
        _build_gs_uri(bucket, "models", "promoted", "best", ".keep", prefix=prefix),
    ]
    for uri in keep_uris:
        if _ensure_gcs_object_text(
            worktree=worktree,
            destination_uri=uri,
            content="",
            heartbeat_seconds=int(heartbeat_seconds),
        ):
            created_objects.append(uri)

    print(
        "Vertex models layout check | "
        f"registry={registry_uri} | created_objects={len(created_objects)}"
        ,
        flush=True,
    )
    return {
        "registry_uri": registry_uri,
        "created_objects": created_objects,
    }


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
    machine_type: str,
    executor_image_uri: str,
    accelerator_type: str,
    accelerator_count: int,
    python_module: str,
) -> str:
    fields = [
        f"machine-type={machine_type}",
        "replica-count=1",
        f"executor-image-uri={executor_image_uri}",
        f"python-module={python_module}",
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


def _build_source_package_archive(*, worktree: Path, identity_key: str) -> Path:
    src_pkg_dir = worktree / "src" / "songo_model_stockfish"
    if not src_pkg_dir.exists():
        raise FileNotFoundError(f"Package source introuvable: {src_pkg_dir}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    version = f"0.0.0.dev{ts}"
    package_basename = f"songo-vertex-app-{identity_key}-{ts}"
    output_dir = worktree / "config" / "generated" / "vertex" / "packages"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{package_basename}.tar.gz"

    with tempfile.TemporaryDirectory(prefix="songo_vertex_pkg_") as tmp:
        pkg_root = Path(tmp) / package_basename
        (pkg_root / "src").mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            src_pkg_dir,
            pkg_root / "src" / "songo_model_stockfish",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        (pkg_root / "setup.py").write_text(
            (
                "from setuptools import find_packages, setup\n\n"
                "setup(\n"
                "    name='songo-vertex-app',\n"
                f"    version='{version}',\n"
                "    description='Songo Vertex Python package',\n"
                "    package_dir={'': 'src'},\n"
                "    packages=find_packages(where='src'),\n"
                "    install_requires=[\n"
                "        'PyYAML>=6.0',\n"
                "        'tqdm>=4.66',\n"
                "        'google-cloud-firestore>=2.16',\n"
                "        'upstash-redis>=1.4.0',\n"
                "    ],\n"
                "    include_package_data=False,\n"
                ")\n"
            ),
            encoding="utf-8",
        )
        (pkg_root / "README.md").write_text(
            "Songo Vertex package generated for Custom Job submission.\n",
            encoding="utf-8",
        )
        with tarfile.open(archive_path, mode="w:gz") as tar:
            tar.add(pkg_root, arcname=package_basename)

    return archive_path


def _upload_file_to_gcs(
    *,
    worktree: Path,
    local_path: Path,
    destination_uri: str,
    heartbeat_seconds: int,
) -> str:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    _run_live_capture(
        ["gcloud", "storage", "cp", str(local_path), destination_uri],
        cwd=worktree,
        env=env,
        heartbeat_s=int(heartbeat_seconds),
    )
    return destination_uri


def _to_gcs_fuse_path(gs_uri: str) -> str:
    text = str(gs_uri or "").strip()
    if not text.startswith("gs://"):
        raise ValueError(f"URI GCS invalide: {text}")
    suffix = text[len("gs://") :].strip().strip("/")
    if not suffix:
        raise ValueError(f"URI GCS invalide: {text}")
    return f"/gcs/{suffix}"


def _submit_custom_job(
    *,
    worktree: Path,
    project_id: str,
    region: str,
    display_name: str,
    worker_pool_spec: str,
    python_package_uris: list[str],
    args_list: list[str],
    service_account: str,
    labels: dict[str, str],
    heartbeat_seconds: int,
) -> str:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    labels_arg = ",".join([f"{k}={v}" for k, v in labels.items() if k and v])
    args_csv = ",".join(args_list)
    package_uris_csv = ",".join([str(uri).strip() for uri in (python_package_uris or []) if str(uri).strip()])
    if not package_uris_csv:
        raise ValueError("Aucun python package URI fourni pour le job Vertex.")

    cmd = [
        "gcloud",
        "ai",
        "custom-jobs",
        "create",
        f"--project={project_id}",
        f"--region={region}",
        f"--display-name={display_name}",
        f"--worker-pool-spec={worker_pool_spec}",
        f"--python-package-uris={package_uris_csv}",
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
    if command_name == "train-eval" and not resolved_dataset_id:
        pointer_uri = str(dataset_pointer_uri or "").strip()
        if not pointer_uri:
            pointer_uri = _build_gs_uri(bucket, "data", "datasets", "merged", "latest.json", prefix=prefix)
        resolved_dataset_id = _discover_dataset_id_from_gcs_pointer(
            pointer_uri=pointer_uri,
            worktree=worktree,
            heartbeat_seconds=int(heartbeat_seconds),
        )
    if command_name == "train-eval":
        _verify_train_eval_dataset_artifacts(
            worktree=worktree,
            bucket=bucket,
            prefix=prefix,
            dataset_id=resolved_dataset_id,
            heartbeat_seconds=int(heartbeat_seconds),
        )
    models_layout_summary = _ensure_vertex_models_layout(
        worktree=worktree,
        bucket=bucket,
        prefix=prefix,
        heartbeat_seconds=int(heartbeat_seconds),
    )

    use_gpu = bool(str(accelerator_type or "").strip()) and int(accelerator_count) > 0
    vertex_drive_root = _build_vertex_drive_root(bucket, prefix)

    package_archive_local = _build_source_package_archive(worktree=worktree, identity_key=identity_key)
    package_archive_uri = _build_gs_uri(
        bucket,
        "code",
        "python_packages",
        package_archive_local.name,
        prefix=prefix,
    )
    _upload_file_to_gcs(
        worktree=worktree,
        local_path=package_archive_local,
        destination_uri=package_archive_uri,
        heartbeat_seconds=int(heartbeat_seconds),
    )

    worker_pool_spec = _build_worker_pool_spec(
        machine_type=str(machine_type),
        executor_image_uri=str(executor_image_uri),
        accelerator_type=str(accelerator_type),
        accelerator_count=int(accelerator_count),
        python_module="songo_model_stockfish.ops.vertex_entrypoint",
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
        train_runtime_gs_uri = _build_gs_uri(
            bucket,
            "config",
            "generated",
            "vertex",
            train_runtime_path.name,
            prefix=prefix,
        )
        eval_runtime_gs_uri = _build_gs_uri(
            bucket,
            "config",
            "generated",
            "vertex",
            eval_runtime_path.name,
            prefix=prefix,
        )
        _upload_file_to_gcs(
            worktree=worktree,
            local_path=train_runtime_path,
            destination_uri=train_runtime_gs_uri,
            heartbeat_seconds=int(heartbeat_seconds),
        )
        _upload_file_to_gcs(
            worktree=worktree,
            local_path=eval_runtime_path,
            destination_uri=eval_runtime_gs_uri,
            heartbeat_seconds=int(heartbeat_seconds),
        )

        args_list = [
            "train-eval",
            "--train-config",
            _to_gcs_fuse_path(train_runtime_gs_uri),
            "--eval-config",
            _to_gcs_fuse_path(eval_runtime_gs_uri),
            "--heartbeat-seconds",
            str(int(heartbeat_seconds)),
        ]

        display_name = f"songo-train-eval-{identity_key}-{ts}"
        custom_job_id = _submit_custom_job(
            worktree=worktree,
            project_id=str(project_id),
            region=str(region),
            display_name=display_name,
            worker_pool_spec=worker_pool_spec,
            python_package_uris=[package_archive_uri],
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
                "train_gcs_uri": train_runtime_gs_uri,
                "evaluation_gcs_uri": eval_runtime_gs_uri,
            },
            "python_package": {
                "local_archive": str(package_archive_local),
                "gcs_uri": package_archive_uri,
                "python_module": "songo_model_stockfish.ops.vertex_entrypoint",
            },
            "worker_pool_spec": worker_pool_spec,
            "stream_logs": bool(stream_logs),
            "models_layout": models_layout_summary,
        }

    if command_name == "benchmark":
        benchmark_runtime_path = _prepare_benchmark_runtime_config(
            worktree=worktree,
            identity=identity_key,
            vertex_drive_root=vertex_drive_root,
            use_gpu=use_gpu,
            benchmark_target=str(benchmark_target or "auto_latest"),
        )
        benchmark_runtime_gs_uri = _build_gs_uri(
            bucket,
            "config",
            "generated",
            "vertex",
            benchmark_runtime_path.name,
            prefix=prefix,
        )
        _upload_file_to_gcs(
            worktree=worktree,
            local_path=benchmark_runtime_path,
            destination_uri=benchmark_runtime_gs_uri,
            heartbeat_seconds=int(heartbeat_seconds),
        )

        args_list = [
            "benchmark",
            "--config",
            _to_gcs_fuse_path(benchmark_runtime_gs_uri),
            "--heartbeat-seconds",
            str(int(heartbeat_seconds)),
        ]

        display_name = f"songo-benchmark-{identity_key}-{ts}"
        custom_job_id = _submit_custom_job(
            worktree=worktree,
            project_id=str(project_id),
            region=str(region),
            display_name=display_name,
            worker_pool_spec=worker_pool_spec,
            python_package_uris=[package_archive_uri],
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
                "benchmark_gcs_uri": benchmark_runtime_gs_uri,
            },
            "python_package": {
                "local_archive": str(package_archive_local),
                "gcs_uri": package_archive_uri,
                "python_module": "songo_model_stockfish.ops.vertex_entrypoint",
            },
            "worker_pool_spec": worker_pool_spec,
            "stream_logs": bool(stream_logs),
            "models_layout": models_layout_summary,
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
