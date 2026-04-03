from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from songo_model_stockfish.ops.logging import JsonlWriter, build_console_logger, utc_now_iso
from songo_model_stockfish.ops.paths import ProjectPaths, build_project_paths


def make_job_id(run_type: str) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    return f"{run_type}_{stamp}_{suffix}"


@dataclass
class JobContext:
    config: dict[str, Any]
    paths: ProjectPaths
    job_id: str
    run_type: str
    job_dir: Path
    logger: Any
    events: JsonlWriter
    metrics: JsonlWriter
    status_path: Path
    state_path: Path

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def write_event(self, message: str, **extra: Any) -> None:
        payload = {
            "timestamp": utc_now_iso(),
            "job_id": self.job_id,
            "run_type": self.run_type,
            "level": "INFO",
            "message": message,
            **extra,
        }
        self.events.write(payload)

    def write_metric(self, metric: dict[str, Any]) -> None:
        payload = {
            "timestamp": utc_now_iso(),
            "job_id": self.job_id,
            **metric,
        }
        self.metrics.write(payload)

    def write_status(self, status: str, *, phase: str = "initializing", extra: dict[str, Any] | None = None) -> None:
        previous = self.read_status()
        payload = {
            "job_id": self.job_id,
            "run_type": self.run_type,
            "status": status,
            "phase": phase,
            "created_at": previous.get("created_at", utc_now_iso()),
            "updated_at": utc_now_iso(),
            "resume_supported": True,
        }
        if extra:
            payload.update(extra)
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_state(self, state: dict[str, Any]) -> None:
        payload = {
            "job_id": self.job_id,
            "run_type": self.run_type,
            **state,
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def set_phase(self, phase: str) -> None:
        current = {}
        if self.status_path.exists():
            current = json.loads(self.status_path.read_text(encoding="utf-8"))
        current.update({"phase": phase, "updated_at": utc_now_iso()})
        self.status_path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def create_job_context(config: dict[str, Any], *, override_job_id: str | None = None) -> JobContext:
    paths = build_project_paths(config)
    for path in [paths.jobs_root, paths.logs_root, paths.reports_root, paths.models_root, paths.data_root]:
        path.mkdir(parents=True, exist_ok=True)

    job_cfg = config.get("job", {})
    run_type = str(job_cfg.get("run_type", "job"))
    requested_job_id = override_job_id or str(job_cfg.get("job_id", "auto"))
    job_id = make_job_id(run_type) if requested_job_id in {"", "auto"} else requested_job_id
    job_dir = paths.jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    logger = build_console_logger(job_id)
    events = JsonlWriter(job_dir / "events.jsonl")
    metrics = JsonlWriter(job_dir / "metrics.jsonl")
    status_path = job_dir / "run_status.json"
    state_path = job_dir / "state.json"

    context = JobContext(
        config=config,
        paths=paths,
        job_id=job_id,
        run_type=run_type,
        job_dir=job_dir,
        logger=logger,
        events=events,
        metrics=metrics,
        status_path=status_path,
        state_path=state_path,
    )

    (job_dir / "config.yaml").write_text(_dump_yaml_like(config), encoding="utf-8")
    if not status_path.exists():
        context.write_status("pending")
    if not state_path.exists():
        context.write_state({"last_completed_phase": "none"})
    return context


def _dump_yaml_like(config: dict[str, Any]) -> str:
    try:
        import yaml
    except ImportError:
        return json.dumps(config, indent=2)
    return yaml.safe_dump(config, sort_keys=False)
