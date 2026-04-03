from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object in {path}")
    return data


def _dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _split_version(model_id: str) -> tuple[str, int]:
    match = re.match(r"^(.*)_v(\d+)$", model_id)
    if not match:
        raise ValueError(
            "Le model_id doit se terminer par un suffixe _vN, par exemple "
            "`songo_policy_value_colab_pro_v1`."
        )
    return match.group(1), int(match.group(2))


def build_next_configs(
    repo_root: Path,
    *,
    train_base: str,
    evaluation_base: str,
    benchmark_base: str,
    next_version: int | None = None,
) -> dict[str, str]:
    train_base_path = repo_root / train_base
    evaluation_base_path = repo_root / evaluation_base
    benchmark_base_path = repo_root / benchmark_base

    train_cfg = _load_yaml(train_base_path)
    evaluation_cfg = _load_yaml(evaluation_base_path)
    benchmark_cfg = _load_yaml(benchmark_base_path)

    current_model_id = str(train_cfg.get("train", {}).get("model_id", ""))
    prefix, current_version = _split_version(current_model_id)
    version = next_version if next_version is not None else current_version + 1
    next_model_id = f"{prefix}_v{version}"

    generated_dir = repo_root / "config" / "generated"
    train_out = generated_dir / f"train.colab_pro.v{version}.yaml"
    eval_out = generated_dir / f"evaluation.colab_pro.v{version}.yaml"
    bench_out = generated_dir / f"benchmark.colab_pro.v{version}.yaml"

    train_cfg["train"]["model_id"] = next_model_id
    train_cfg["train"]["init_checkpoint_path"] = ""
    train_cfg["train"]["init_from_promoted_best"] = True

    evaluation_cfg["evaluation"]["model_id"] = next_model_id
    evaluation_cfg["evaluation"]["checkpoint_path"] = f"models/final/{next_model_id}.pt"

    benchmark_cfg["benchmark"]["target"] = next_model_id
    benchmark_cfg["benchmark"]["checkpoint_path"] = f"models/final/{next_model_id}.pt"

    _dump_yaml(train_out, train_cfg)
    _dump_yaml(eval_out, evaluation_cfg)
    _dump_yaml(bench_out, benchmark_cfg)

    return {
        "model_id": next_model_id,
        "train_config": str(train_out.relative_to(repo_root)),
        "evaluation_config": str(eval_out.relative_to(repo_root)),
        "benchmark_config": str(bench_out.relative_to(repo_root)),
        "train_job_id": f"train_colab_pro_v{version}",
        "evaluation_job_id": f"eval_colab_pro_v{version}",
        "benchmark_job_id": f"benchmark_colab_pro_v{version}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--train-base", default="config/train.colab_pro.yaml")
    parser.add_argument("--evaluation-base", default="config/evaluation.colab_pro.yaml")
    parser.add_argument("--benchmark-base", default="config/benchmark.colab_pro.yaml")
    parser.add_argument("--next-version", type=int)
    args = parser.parse_args()

    summary = build_next_configs(
        args.repo_root,
        train_base=args.train_base,
        evaluation_base=args.evaluation_base,
        benchmark_base=args.benchmark_base,
        next_version=args.next_version,
    )

    worktree = "/content/songo-model-stockfish-for-google-collab"
    print(f"model_id: {summary['model_id']}")
    print(f"train_config: {summary['train_config']}")
    print(f"evaluation_config: {summary['evaluation_config']}")
    print(f"benchmark_config: {summary['benchmark_config']}")
    print()
    print("Recommended commands:")
    print(
        f"PYTHONPATH=src python -m songo_model_stockfish.cli.main train --config {summary['train_config']} "
        f"--job-id {summary['train_job_id']}"
    )
    print(
        f"PYTHONPATH=src python -m songo_model_stockfish.cli.main evaluate --config {summary['evaluation_config']} "
        f"--job-id {summary['evaluation_job_id']}"
    )
    print(
        f"PYTHONPATH=src python -m songo_model_stockfish.cli.main benchmark --config {summary['benchmark_config']} "
        f"--job-id {summary['benchmark_job_id']}"
    )
    print()
    print(f"Colab worktree: {worktree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
