from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _inject_storage(cfg: dict[str, Any], *, drive_root: Path, workspace_rel: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("storage", {})
    out["storage"]["drive_root"] = str(drive_root)
    out["storage"]["jobs_root"] = f"{workspace_rel}/jobs"
    out["storage"]["jobs_backup_root"] = f"{workspace_rel}/runtime_backup/jobs"
    out["storage"]["logs_root"] = f"{workspace_rel}/logs"
    out["storage"]["reports_root"] = f"{workspace_rel}/reports"
    out["storage"]["data_root"] = f"{workspace_rel}/data"
    # Global models partagés pour promotion globale entre Colabs.
    out["storage"]["models_root"] = "models"
    out["storage"]["runtime_state_backup_enabled"] = True
    out.setdefault("runtime", {})
    out["runtime"]["device"] = out["runtime"].get("device", "cuda")
    return out


def generate_active_configs(
    *,
    worktree: Path,
    drive_root: Path,
    identity: str,
    target_samples: int,
    target_labeled_samples: int,
    games_per_matchup: int,
) -> dict[str, Any]:
    drive_identity_key = str(identity or "").strip() or "unknown_drive_identity"
    workspace_rel = drive_identity_key

    base_source_id = f"sampled_full_matrix_{drive_identity_key}"
    base_dataset_id = f"dataset_full_matrix_{drive_identity_key}"
    run_ts = int(time.time())
    train_job_id = f"train_{drive_identity_key}_{run_ts}"
    eval_job_id = f"eval_{drive_identity_key}_{run_ts}"
    bench_job_id = f"benchmark_{drive_identity_key}_{run_ts}"

    generated = worktree / "config" / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    gen = _inject_storage(
        _load_yaml(worktree / "config" / "dataset_generation.full_matrix.colab_pro.yaml"),
        drive_root=drive_root,
        workspace_rel=workspace_rel,
    )
    gen.setdefault("job", {})
    gen["job"]["resume"] = True
    gen["job"]["run_type"] = "dataset_generate"
    gen["job"]["job_id"] = f"dataset_generate_{drive_identity_key}"
    gen.setdefault("dataset_generation", {})
    dgen = gen["dataset_generation"]
    dgen["source_mode"] = "benchmatch"
    dgen["dataset_source_id"] = base_source_id
    dgen["output_raw_dir"] = f"{workspace_rel}/data/raw/{base_source_id}"
    dgen["output_sampled_dir"] = f"{workspace_rel}/data/sampled/{base_source_id}"
    dgen["target_samples"] = int(target_samples)
    dgen["games"] = int(games_per_matchup)
    dgen["cycle_matchups_until_target"] = True
    dgen["max_matchup_cycles"] = 0
    dgen["matchups"] = [
        "minimax:medium vs minimax:hard",
        "minimax:hard vs minimax:medium",
        "mcts:medium vs mcts:hard",
        "mcts:hard vs mcts:medium",
        "minimax:medium vs mcts:medium",
        "minimax:medium vs mcts:hard",
        "minimax:hard vs mcts:medium",
        "minimax:hard vs mcts:hard",
        "mcts:medium vs minimax:medium",
        "mcts:medium vs minimax:hard",
        "mcts:hard vs minimax:medium",
        "mcts:hard vs minimax:hard",
    ]
    gen_active = generated / f"dataset_generation.{drive_identity_key}.active.yaml"
    _save_yaml(gen_active, gen)

    build = _inject_storage(
        _load_yaml(worktree / "config" / "dataset_build.full_matrix.colab_pro.yaml"),
        drive_root=drive_root,
        workspace_rel=workspace_rel,
    )
    build.setdefault("job", {})
    build["job"]["resume"] = True
    build["job"]["run_type"] = "dataset_build"
    build["job"]["job_id"] = f"dataset_build_{drive_identity_key}"
    build.setdefault("dataset_build", {})
    db = build["dataset_build"]
    db["build_mode"] = "auto"
    db["source_dataset_id"] = base_source_id
    db["input_sampled_dir"] = f"{workspace_rel}/data/sampled/{base_source_id}"
    db["dataset_id"] = base_dataset_id
    db["output_dir"] = f"{workspace_rel}/data/datasets/{base_dataset_id}"
    db["label_cache_dir"] = f"{workspace_rel}/data/label_cache/{base_dataset_id}"
    db["target_labeled_samples"] = int(target_labeled_samples)
    db["follow_source_updates"] = True
    db.setdefault("teacher", {})
    db["teacher"]["engine"] = "minimax"
    db["teacher"]["level"] = "insane"
    build_active = generated / f"dataset_build.{drive_identity_key}.active.yaml"
    _save_yaml(build_active, build)

    train = _inject_storage(
        _load_yaml(worktree / "config" / "train.full_matrix.colab_pro.yaml"),
        drive_root=drive_root,
        workspace_rel=workspace_rel,
    )
    train.setdefault("job", {})
    train["job"]["run_type"] = "train"
    train["job"]["resume"] = True
    train["job"]["job_id"] = train_job_id
    train.setdefault("train", {})
    tr = train["train"]
    tr["dataset_selection_mode"] = "largest_built"
    tr["dataset_id"] = "auto"
    tr["init_from_promoted_best"] = True
    tr["promoted_best_checkpoint_path"] = "models/promoted/best/model.pt"
    tr["model_id_prefix"] = f"songo_policy_value_colab_pro_{drive_identity_key}"
    tr["checkpoint_dir"] = f"models/checkpoints/{drive_identity_key}"
    tr["lineage_dir"] = f"models/lineage/{drive_identity_key}"
    train_active = generated / f"train.{drive_identity_key}.active.yaml"
    _save_yaml(train_active, train)

    eval_cfg = _inject_storage(
        _load_yaml(worktree / "config" / "evaluation.full_matrix.colab_pro.yaml"),
        drive_root=drive_root,
        workspace_rel=workspace_rel,
    )
    eval_cfg.setdefault("job", {})
    eval_cfg["job"]["run_type"] = "evaluation"
    eval_cfg["job"]["resume"] = True
    eval_cfg["job"]["job_id"] = eval_job_id
    eval_cfg.setdefault("evaluation", {})
    eval_cfg["evaluation"]["model_id"] = "auto_latest"
    eval_cfg["evaluation"]["dataset_selection_mode"] = "largest_built"
    eval_cfg["evaluation"]["output_dir"] = f"{workspace_rel}/reports/evaluations"
    eval_active = generated / f"evaluation.{drive_identity_key}.active.yaml"
    _save_yaml(eval_active, eval_cfg)

    bench = _inject_storage(
        _load_yaml(worktree / "config" / "benchmark.colab_pro.yaml"),
        drive_root=drive_root,
        workspace_rel=workspace_rel,
    )
    bench.setdefault("job", {})
    bench["job"]["run_type"] = "benchmark"
    bench["job"]["resume"] = True
    bench["job"]["job_id"] = bench_job_id
    bench.setdefault("benchmark", {})
    b = bench["benchmark"]
    b["target"] = "auto_latest"
    b["model_search_profile"] = "fort_plusplus"
    b["model_search_depth"] = 3
    b["model_search_top_k"] = 6
    b["model_search_top_k_child"] = 4
    b["model_search_alpha_beta"] = True
    b["games_per_matchup"] = 50
    b["matchups"] = ["minimax:medium", "minimax:hard", "mcts:medium", "mcts:hard", "mcts:insane"]
    b["output_dir"] = f"{workspace_rel}/reports/benchmarks"
    bench_active = generated / f"benchmark.{drive_identity_key}.active.yaml"
    _save_yaml(bench_active, bench)

    summary: dict[str, Any] = {
        "drive_identity_key": drive_identity_key,
        "workspace_rel": workspace_rel,
        "target_samples": int(target_samples),
        "target_labeled_samples": int(target_labeled_samples),
        "games_per_matchup": int(games_per_matchup),
        "active_configs": {
            "dataset_generate": str(gen_active),
            "dataset_build": str(build_active),
            "train": str(train_active),
            "evaluation": str(eval_active),
            "benchmark": str(bench_active),
        },
    }

    print("Configs actives:")
    print(" -", summary["active_configs"]["dataset_generate"])
    print(" -", summary["active_configs"]["dataset_build"])
    print(" -", summary["active_configs"]["train"])
    print(" -", summary["active_configs"]["evaluation"])
    print(" -", summary["active_configs"]["benchmark"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/songo-stockfish")
    parser.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    parser.add_argument("--target-samples", type=int, default=int(os.environ.get("SONGO_TARGET_SAMPLES", "1000000000")))
    parser.add_argument("--target-labeled-samples", type=int, default=None)
    parser.add_argument("--games-per-matchup", type=int, default=int(os.environ.get("SONGO_GAMES_PER_MATCHUP", "400")))
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    target_labeled_samples = (
        int(args.target_labeled_samples)
        if args.target_labeled_samples is not None
        else int(os.environ.get("SONGO_TARGET_LABELED_SAMPLES", str(int(args.target_samples))))
    )

    summary = generate_active_configs(
        worktree=Path(str(args.worktree)),
        drive_root=Path(str(args.drive_root)),
        identity=str(args.identity),
        target_samples=int(args.target_samples),
        target_labeled_samples=int(target_labeled_samples),
        games_per_matchup=int(args.games_per_matchup),
    )

    summary_path = str(args.summary_path or "").strip()
    if summary_path:
        Path(summary_path).write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    if bool(args.print_json):
        print(json.dumps(summary, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
