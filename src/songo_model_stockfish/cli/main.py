from __future__ import annotations

import argparse
import json
from pathlib import Path

from songo_model_stockfish.benchmark.jobs import run_benchmark_job
from songo_model_stockfish.data.jobs import run_dataset_build, run_dataset_generation
from songo_model_stockfish.evaluation.jobs import run_evaluation
from songo_model_stockfish.ops.config import load_yaml_config
from songo_model_stockfish.ops.job import create_job_context
from songo_model_stockfish.training.jobs import run_train


def _apply_dataset_generate_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    dataset_cfg = dict(config.get("dataset_generation", {}))
    if getattr(args, "generation_mode", None):
        dataset_cfg["source_mode"] = args.generation_mode
    if getattr(args, "dataset_source_id", None):
        dataset_cfg["dataset_source_id"] = args.dataset_source_id
    if getattr(args, "source_dataset_id", None):
        dataset_cfg["source_dataset_id"] = args.source_dataset_id
    if getattr(args, "derivation_strategy", None):
        dataset_cfg["derivation_strategy"] = args.derivation_strategy
    updated = dict(config)
    updated["dataset_generation"] = dataset_cfg
    return updated


def _apply_dataset_build_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    dataset_cfg = dict(config.get("dataset_build", {}))
    if getattr(args, "source_dataset_id", None):
        dataset_cfg["source_dataset_id"] = args.source_dataset_id
    if getattr(args, "dataset_id_override", None):
        dataset_cfg["dataset_id"] = args.dataset_id_override
    updated = dict(config)
    updated["dataset_build"] = dataset_cfg
    return updated


def _add_common_arguments(parser: argparse.ArgumentParser, *, require_config: bool = True) -> None:
    if require_config:
        parser.add_argument("--config", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _execute_job(config_path: str, handler, *, override_job_id: str | None, dry_run: bool, config_override: dict[str, object] | None = None) -> int:
    config = config_override or load_yaml_config(config_path)
    job = create_job_context(config, override_job_id=override_job_id)
    job.write_status("running", phase="startup")
    job.write_event("job_started", config_path=str(config_path))
    if dry_run:
        print(json.dumps({"job_id": job.job_id, "run_type": job.run_type, "dry_run": True}, indent=2))
        job.write_status("completed", phase="dry_run")
        return 0

    try:
        result = handler(job)
    except Exception as exc:
        job.logger.exception("job failed")
        job.write_event("job_failed", error=f"{type(exc).__name__}: {exc}")
        job.write_status("failed", phase="error", extra={"error": f"{type(exc).__name__}: {exc}"})
        raise
    job.write_event("job_completed")
    job.write_status("completed", phase="done")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


def _resume_job(job_id: str) -> int:
    jobs_root = Path("/content/drive/MyDrive/songo-stockfish/jobs")
    if not jobs_root.exists():
        jobs_root = Path(__file__).resolve().parents[4] / "outputs" / "jobs"
    job_dir = jobs_root / job_id
    config_path = job_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config de job introuvable: {config_path}")
    config = load_yaml_config(config_path)
    run_type = str(config.get("job", {}).get("run_type", ""))
    handlers = {
        "dataset_generation": run_dataset_generation,
        "dataset_build": run_dataset_build,
        "train": run_train,
        "benchmark": run_benchmark_job,
        "evaluation": run_evaluation,
    }
    if run_type not in handlers:
        raise ValueError(f"Run type non supporte pour resume: {run_type}")
    return _execute_job(str(config_path), handlers[run_type], override_job_id=job_id, dry_run=False)


def _status(job_id: str) -> int:
    jobs_root = Path("/content/drive/MyDrive/songo-stockfish/jobs")
    if not jobs_root.exists():
        jobs_root = Path(__file__).resolve().parents[4] / "outputs" / "jobs"
    job_dir = jobs_root / job_id
    status_path = job_dir / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"run_status introuvable: {status_path}")
    print(status_path.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="songo_model_stockfish")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_generate = subparsers.add_parser("dataset-generate")
    _add_common_arguments(dataset_generate)
    dataset_generate.add_argument("--generation-mode", choices=["benchmatch", "clone_existing", "derive_existing"])
    dataset_generate.add_argument("--dataset-source-id")
    dataset_generate.add_argument("--source-dataset-id")
    dataset_generate.add_argument("--derivation-strategy", choices=["unique_positions", "endgame_focus", "high_branching"])

    dataset_build = subparsers.add_parser("dataset-build")
    _add_common_arguments(dataset_build)
    dataset_build.add_argument("--source-dataset-id")
    dataset_build.add_argument("--dataset-id-override")

    train = subparsers.add_parser("train")
    _add_common_arguments(train)

    benchmark = subparsers.add_parser("benchmark")
    _add_common_arguments(benchmark)

    evaluate = subparsers.add_parser("evaluate")
    _add_common_arguments(evaluate)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--job-id", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--job-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset-generate":
        config = _apply_dataset_generate_overrides(load_yaml_config(args.config), args)
        return _execute_job(
            args.config,
            run_dataset_generation,
            override_job_id=args.job_id,
            dry_run=args.dry_run,
            config_override=config,
        )
    if args.command == "dataset-build":
        config = _apply_dataset_build_overrides(load_yaml_config(args.config), args)
        return _execute_job(
            args.config,
            run_dataset_build,
            override_job_id=args.job_id,
            dry_run=args.dry_run,
            config_override=config,
        )
    if args.command == "train":
        return _execute_job(args.config, run_train, override_job_id=args.job_id, dry_run=args.dry_run)
    if args.command == "benchmark":
        return _execute_job(args.config, run_benchmark_job, override_job_id=args.job_id, dry_run=args.dry_run)
    if args.command == "evaluate":
        return _execute_job(args.config, run_evaluation, override_job_id=args.job_id, dry_run=args.dry_run)
    if args.command == "resume":
        return _resume_job(args.job_id)
    if args.command == "status":
        return _status(args.job_id)
    parser.error("Commande non supportee")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
