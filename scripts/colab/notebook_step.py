from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from bootstrap_workspace import bootstrap_workspace
from generate_active_configs import generate_active_configs
from run_streaming_pipeline import run_streaming_pipeline
from run_job import _run_single_command, _run_train_eval_benchmark


def _write_summary(summary: dict[str, Any], summary_path: str, print_json: bool) -> None:
    path_text = str(summary_path or "").strip()
    if path_text:
        Path(path_text).write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    if bool(print_json):
        print(json.dumps(summary, indent=2, ensure_ascii=True))


def _run_audit_storage(*, drive_root: Path) -> dict[str, Any]:
    print("Aucun nettoyage automatique dans ce notebook (purge desactivee).")
    print("Drive root =", drive_root)

    if not drive_root.exists():
        raise RuntimeError(f"Drive root introuvable: {drive_root}")

    print("\nContenu racine:")
    root_entries: list[dict[str, str]] = []
    for item in sorted(drive_root.iterdir(), key=lambda p: p.name):
        item_type = "dir" if item.is_dir() else "file"
        root_entries.append({"name": item.name, "type": item_type})
        typ = "DIR " if item_type == "dir" else "FILE"
        print(f" - [{typ}] {item.name}")

    print("\nWorkspaces Colab detectes:")
    workspaces = [
        p
        for p in drive_root.iterdir()
        if p.is_dir() and (p.name.startswith("colab_") or p.name == "unknown_drive_identity")
    ]
    if not workspaces:
        print(" - aucun")
    else:
        for ws in sorted(workspaces, key=lambda p: p.name):
            print(" -", ws)

    return {
        "drive_root": str(drive_root),
        "root_entries": root_entries,
        "workspaces": [str(p) for p in sorted(workspaces, key=lambda p: p.name)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="step", required=True)

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument(
        "--git-repo-url",
        default="https://github.com/GlennEriss/songo-model-stockfish-for-google-collab.git",
    )
    bootstrap.add_argument("--git-branch", default="main")
    bootstrap.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    bootstrap.add_argument("--drive-project-name", default="songo-stockfish")
    bootstrap.add_argument("--colab-identity", default="")
    bootstrap.add_argument("--mydrive-root", default="/content/drive/MyDrive")
    bootstrap.add_argument("--python-bin", default=(sys.executable or "python3"))
    bootstrap.add_argument("--skip-install", action="store_true")
    bootstrap.add_argument("--summary-path", default="")
    bootstrap.add_argument("--print-json", action="store_true")

    configs = subparsers.add_parser("generate-configs")
    configs.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    configs.add_argument(
        "--drive-root",
        default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"),
    )
    configs.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    configs.add_argument("--target-samples", type=int, default=int(os.environ.get("SONGO_TARGET_SAMPLES", "1000000000")))
    configs.add_argument("--target-labeled-samples", type=int, default=None)
    configs.add_argument("--games-per-matchup", type=int, default=int(os.environ.get("SONGO_GAMES_PER_MATCHUP", "400")))
    configs.add_argument("--summary-path", default="")
    configs.add_argument("--print-json", action="store_true")

    audit = subparsers.add_parser("audit-storage")
    audit.add_argument(
        "--drive-root",
        default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"),
    )
    audit.add_argument("--summary-path", default="")
    audit.add_argument("--print-json", action="store_true")

    run_job = subparsers.add_parser("run-job")
    run_job.add_argument(
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
    run_job.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    run_job.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    run_job.add_argument("--python-bin", default=(sys.executable or os.environ.get("PYTHON_BIN", "python3")))
    run_job.add_argument("--heartbeat-seconds", type=int, default=30)
    run_job.add_argument("--config", default="")
    run_job.add_argument("--train-config", default="")
    run_job.add_argument("--eval-config", default="")
    run_job.add_argument("--benchmark-config", default="")
    run_job.add_argument(
        "--drive-root",
        default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"),
    )
    run_job.add_argument("--summary-path", default="")
    run_job.add_argument("--print-json", action="store_true")

    streaming = subparsers.add_parser("streaming-pipeline")
    streaming.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    streaming.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    streaming.add_argument("--python-bin", default=(sys.executable or os.environ.get("PYTHON_BIN", "python3")))
    streaming.add_argument(
        "--drive-root",
        default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"),
    )
    streaming.add_argument("--heartbeat-seconds", type=int, default=30)
    streaming.add_argument("--poll-seconds", type=float, default=20.0)
    streaming.add_argument("--train-min-samples", type=int, default=int(os.environ.get("SONGO_STREAM_TRAIN_MIN_SAMPLES", "50000")))
    streaming.add_argument(
        "--train-min-delta-samples",
        type=int,
        default=int(os.environ.get("SONGO_STREAM_TRAIN_MIN_DELTA_SAMPLES", "50000")),
    )
    streaming.add_argument("--max-train-runs", type=int, default=int(os.environ.get("SONGO_STREAM_MAX_TRAIN_RUNS", "0")))
    streaming.add_argument("--disable-auto-train", action="store_true")
    streaming.add_argument("--continue-on-train-error", action="store_true")
    streaming.add_argument("--skip-generate", action="store_true")
    streaming.add_argument("--skip-build", action="store_true")
    streaming.add_argument("--state-path", default="")
    streaming.add_argument("--summary-path", default="")
    streaming.add_argument("--print-json", action="store_true")

    tournament = subparsers.add_parser("model-tournament")
    tournament.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    tournament.add_argument(
        "--drive-root",
        default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"),
    )
    tournament.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    tournament.add_argument("--games-per-pair", type=int, default=10)
    tournament.add_argument("--max-moves", type=int, default=400)
    tournament.add_argument("--max-models", type=int, default=0)
    tournament.add_argument("--device", default="")
    tournament.add_argument("--search-enabled", choices=["auto", "true", "false"], default="auto")
    tournament.add_argument("--search-alpha-beta", choices=["auto", "true", "false"], default="auto")
    tournament.add_argument("--search-depth", type=int, default=0)
    tournament.add_argument("--search-top-k", type=int, default=0)
    tournament.add_argument("--search-top-k-child", type=int, default=0)
    tournament.add_argument("--search-policy-weight", type=float, default=None)
    tournament.add_argument("--search-value-weight", type=float, default=None)
    tournament.add_argument("--summary-path", default="")
    tournament.add_argument("--print-json", action="store_true")

    args = parser.parse_args()
    step = str(args.step)

    if step == "bootstrap":
        summary = bootstrap_workspace(
            git_repo_url=str(args.git_repo_url),
            git_branch=str(args.git_branch),
            worktree=Path(str(args.worktree)),
            drive_project_name=str(args.drive_project_name),
            colab_identity=str(args.colab_identity),
            mydrive_root=Path(str(args.mydrive_root)),
            python_bin=str(args.python_bin),
            install_requirements=(not bool(args.skip_install)),
        )
        _write_summary(summary, str(args.summary_path), bool(args.print_json))
        return 0

    if step == "generate-configs":
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
        _write_summary(summary, str(args.summary_path), bool(args.print_json))
        return 0

    if step == "audit-storage":
        summary = _run_audit_storage(drive_root=Path(str(args.drive_root)))
        _write_summary(summary, str(args.summary_path), bool(args.print_json))
        return 0

    if step == "streaming-pipeline":
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
        _write_summary(summary, str(args.summary_path), bool(args.print_json))
        return 0

    if step == "model-tournament":
        from run_model_tournament import run_model_tournament

        summary = run_model_tournament(
            worktree=Path(str(args.worktree)),
            drive_root=Path(str(args.drive_root)),
            identity=str(args.identity),
            games_per_pair=int(args.games_per_pair),
            max_moves=int(args.max_moves),
            max_models=int(args.max_models),
            device=str(args.device),
            search_enabled_choice=str(args.search_enabled),
            search_alpha_beta_choice=str(args.search_alpha_beta),
            search_depth=int(args.search_depth),
            search_top_k=int(args.search_top_k),
            search_top_k_child=int(args.search_top_k_child),
            search_policy_weight=(None if args.search_policy_weight is None else float(args.search_policy_weight)),
            search_value_weight=(None if args.search_value_weight is None else float(args.search_value_weight)),
        )
        _write_summary(summary, str(args.summary_path), bool(args.print_json))
        return 0

    command = str(args.command)
    if command == "train-eval-benchmark":
        summary = _run_train_eval_benchmark(
            python_bin=str(args.python_bin),
            worktree=Path(str(args.worktree)),
            identity=str(args.identity),
            heartbeat_seconds=int(args.heartbeat_seconds),
            train_config=(Path(str(args.train_config)) if str(args.train_config).strip() else None),
            eval_config=(Path(str(args.eval_config)) if str(args.eval_config).strip() else None),
            benchmark_config=(Path(str(args.benchmark_config)) if str(args.benchmark_config).strip() else None),
            drive_root=Path(str(args.drive_root)),
        )
        _write_summary(summary, str(args.summary_path), bool(args.print_json))
        return 0

    _run_single_command(
        python_bin=str(args.python_bin),
        worktree=Path(str(args.worktree)),
        identity=str(args.identity),
        command=command,
        config_path=(Path(str(args.config)) if str(args.config).strip() else None),
        heartbeat_seconds=int(args.heartbeat_seconds),
    )
    _write_summary({"command": command, "status": "ok"}, str(args.summary_path), bool(args.print_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
