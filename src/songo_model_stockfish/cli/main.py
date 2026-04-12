from __future__ import annotations

import argparse
import json
from pathlib import Path

from songo_model_stockfish.ops.config import load_yaml_config
from songo_model_stockfish.ops.job import create_job_context
from songo_model_stockfish.ops.paths import build_project_paths


def _apply_dataset_generate_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    dataset_cfg = dict(config.get("dataset_generation", {}))
    derivation_params = dict(dataset_cfg.get("derivation_params", {}))
    if getattr(args, "generation_mode", None):
        dataset_cfg["source_mode"] = args.generation_mode
    if getattr(args, "dataset_source_id", None):
        dataset_cfg["dataset_source_id"] = args.dataset_source_id
    if getattr(args, "source_dataset_id", None):
        dataset_cfg["source_dataset_id"] = args.source_dataset_id
    if getattr(args, "source_dataset_ids", None):
        dataset_cfg["source_dataset_ids"] = [item.strip() for item in str(args.source_dataset_ids).split(",") if item.strip()]
    if getattr(args, "derivation_strategy", None):
        dataset_cfg["derivation_strategy"] = args.derivation_strategy
    if getattr(args, "augmentation_include_original_samples", None) is not None:
        derivation_params["include_original_samples"] = bool(args.augmentation_include_original_samples)
    if getattr(args, "augmentation_max_depth", None) is not None:
        derivation_params["max_depth"] = int(args.augmentation_max_depth)
    if getattr(args, "augmentation_max_branching", None) is not None:
        derivation_params["max_branching"] = int(args.augmentation_max_branching)
    if getattr(args, "augmentation_max_generated_per_source_sample", None) is not None:
        derivation_params["max_generated_per_source_sample"] = int(args.augmentation_max_generated_per_source_sample)
    dataset_cfg["derivation_params"] = derivation_params
    if getattr(args, "target_samples", None) is not None:
        dataset_cfg["target_samples"] = int(args.target_samples)
    if getattr(args, "merge_dedupe_sample_ids", None) is not None:
        dataset_cfg["merge_dedupe_sample_ids"] = bool(args.merge_dedupe_sample_ids)
    updated = dict(config)
    updated["dataset_generation"] = dataset_cfg
    return updated


def _apply_dataset_build_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    dataset_cfg = dict(config.get("dataset_build", {}))
    if getattr(args, "source_dataset_id", None):
        dataset_cfg["source_dataset_id"] = args.source_dataset_id
    if getattr(args, "dataset_id_override", None):
        dataset_cfg["dataset_id"] = args.dataset_id_override
    if getattr(args, "target_labeled_samples", None) is not None:
        dataset_cfg["target_labeled_samples"] = int(args.target_labeled_samples)
    updated = dict(config)
    updated["dataset_build"] = dataset_cfg
    return updated


def _apply_dataset_merge_final_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    merge_cfg = dict(config.get("dataset_merge_final", {}))
    if getattr(args, "dataset_id", None):
        merge_cfg["dataset_id"] = args.dataset_id
    if getattr(args, "source_dataset_ids", None):
        merge_cfg["source_dataset_ids"] = list(args.source_dataset_ids)
    if getattr(args, "include_all_built", None):
        merge_cfg["include_all_built"] = bool(args.include_all_built)
    if getattr(args, "dedupe_sample_ids", None) is not None:
        merge_cfg["dedupe_sample_ids"] = bool(args.dedupe_sample_ids)
    updated = dict(config)
    updated["dataset_merge_final"] = merge_cfg
    updated.setdefault("job", {})
    updated["job"] = dict(updated["job"])
    updated["job"]["run_type"] = "dataset_merge_final"
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
    if run_type == "dataset_generation":
        from songo_model_stockfish.data.jobs import run_dataset_generation

        handler = run_dataset_generation
    elif run_type == "dataset_build":
        from songo_model_stockfish.data.jobs import run_dataset_build

        handler = run_dataset_build
    elif run_type == "train":
        from songo_model_stockfish.training.jobs import run_train

        handler = run_train
    elif run_type == "benchmark":
        from songo_model_stockfish.benchmark.jobs import run_benchmark_job

        handler = run_benchmark_job
    elif run_type == "evaluation":
        from songo_model_stockfish.evaluation.jobs import run_evaluation

        handler = run_evaluation
    elif run_type == "dataset_merge_final":
        from songo_model_stockfish.data.jobs import run_dataset_merge_final

        handler = run_dataset_merge_final
    else:
        raise ValueError(f"Run type non supporte pour resume: {run_type}")
    return _execute_job(str(config_path), handler, override_job_id=job_id, dry_run=False)


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


def _dataset_list(config_path: str | None, *, kind: str, output_json: bool) -> int:
    config = load_yaml_config(config_path) if config_path else {}
    paths = build_project_paths({"storage": config.get("storage", {})})
    paths.data_root.mkdir(parents=True, exist_ok=True)
    registry_path = paths.data_root / "dataset_registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {"dataset_sources": [], "built_datasets": []}
    )

    if kind == "sources":
        payload = {"dataset_sources": registry.get("dataset_sources", [])}
    elif kind == "built":
        payload = {"built_datasets": registry.get("built_datasets", [])}
    else:
        payload = registry

    if output_json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    if kind in {"all", "sources"}:
        print("Dataset sources:")
        for item in payload.get("dataset_sources", []):
            print(
                f"- {item.get('dataset_source_id')} | mode={item.get('source_mode')} | "
                f"parent={item.get('source_dataset_id', '')} | parents={item.get('source_dataset_ids', [])} | samples={item.get('sampled_positions', 0)} | "
                f"strategy={item.get('derivation_strategy', '')}"
            )
        if not payload.get("dataset_sources"):
            print("- none")

    if kind == "all":
        print("")

    if kind in {"all", "built"}:
        print("Built datasets:")
        for item in payload.get("built_datasets", []):
            print(
                f"- {item.get('dataset_id')} | build_mode={item.get('build_mode', 'teacher_label')} | source={item.get('source_dataset_id')} | "
                f"teacher={item.get('teacher_engine')}:{item.get('teacher_level')} | "
                f"labeled={item.get('labeled_samples', 0)}"
            )
        if not payload.get("built_datasets"):
            print("- none")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="songo_model_stockfish")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_generate = subparsers.add_parser("dataset-generate")
    _add_common_arguments(dataset_generate)
    dataset_generate.add_argument("--generation-mode", choices=["benchmatch", "clone_existing", "derive_existing", "augment_existing", "merge_existing"])
    dataset_generate.add_argument("--dataset-source-id")
    dataset_generate.add_argument("--source-dataset-id")
    dataset_generate.add_argument("--source-dataset-ids", nargs="*")
    dataset_generate.add_argument(
        "--derivation-strategy",
        choices=[
            "unique_positions",
            "endgame_focus",
            "high_branching",
            "balanced_score_gap",
            "balanced_legal_moves",
            "rare_seed_profiles",
        ],
    )
    dataset_generate.add_argument("--target-samples", type=int)
    dataset_generate.add_argument("--augmentation-include-original-samples", dest="augmentation_include_original_samples", action="store_true", default=None)
    dataset_generate.add_argument("--augmentation-only-new-samples", dest="augmentation_include_original_samples", action="store_false")
    dataset_generate.add_argument("--augmentation-max-depth", type=int)
    dataset_generate.add_argument("--augmentation-max-branching", type=int)
    dataset_generate.add_argument("--augmentation-max-generated-per-source-sample", type=int)
    dataset_generate.add_argument("--merge-dedupe-sample-ids", dest="merge_dedupe_sample_ids", action="store_true", default=None)
    dataset_generate.add_argument("--keep-duplicate-sample-ids", dest="merge_dedupe_sample_ids", action="store_false")

    dataset_build = subparsers.add_parser("dataset-build")
    _add_common_arguments(dataset_build)
    dataset_build.add_argument("--source-dataset-id")
    dataset_build.add_argument("--dataset-id-override")
    dataset_build.add_argument("--target-labeled-samples", type=int)

    dataset_merge_final = subparsers.add_parser("dataset-merge-final")
    _add_common_arguments(dataset_merge_final)
    dataset_merge_final.add_argument("--dataset-id", required=True)
    dataset_merge_final.add_argument("--source-dataset-ids", nargs="*")
    dataset_merge_final.add_argument("--include-all-built", action="store_true")
    dataset_merge_final.add_argument("--dedupe-sample-ids", dest="dedupe_sample_ids", action="store_true", default=True)
    dataset_merge_final.add_argument("--keep-duplicate-sample-ids", dest="dedupe_sample_ids", action="store_false")

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

    dataset_list = subparsers.add_parser("dataset-list")
    dataset_list.add_argument("--config")
    dataset_list.add_argument("--kind", choices=["all", "sources", "built"], default="all")
    dataset_list.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset-generate":
        from songo_model_stockfish.data.jobs import run_dataset_generation

        config = _apply_dataset_generate_overrides(load_yaml_config(args.config), args)
        return _execute_job(
            args.config,
            run_dataset_generation,
            override_job_id=args.job_id,
            dry_run=args.dry_run,
            config_override=config,
        )
    if args.command == "dataset-build":
        from songo_model_stockfish.data.jobs import run_dataset_build

        config = _apply_dataset_build_overrides(load_yaml_config(args.config), args)
        return _execute_job(
            args.config,
            run_dataset_build,
            override_job_id=args.job_id,
            dry_run=args.dry_run,
            config_override=config,
        )
    if args.command == "train":
        from songo_model_stockfish.training.jobs import run_train

        return _execute_job(args.config, run_train, override_job_id=args.job_id, dry_run=args.dry_run)
    if args.command == "dataset-merge-final":
        from songo_model_stockfish.data.jobs import run_dataset_merge_final

        config = _apply_dataset_merge_final_overrides(load_yaml_config(args.config), args)
        return _execute_job(
            args.config,
            run_dataset_merge_final,
            override_job_id=args.job_id,
            dry_run=args.dry_run,
            config_override=config,
        )
    if args.command == "benchmark":
        from songo_model_stockfish.benchmark.jobs import run_benchmark_job

        return _execute_job(args.config, run_benchmark_job, override_job_id=args.job_id, dry_run=args.dry_run)
    if args.command == "evaluate":
        from songo_model_stockfish.evaluation.jobs import run_evaluation

        return _execute_job(args.config, run_evaluation, override_job_id=args.job_id, dry_run=args.dry_run)
    if args.command == "resume":
        return _resume_job(args.job_id)
    if args.command == "status":
        return _status(args.job_id)
    if args.command == "dataset-list":
        return _dataset_list(args.config, kind=args.kind, output_json=args.json)
    parser.error("Commande non supportee")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
