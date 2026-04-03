from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

import numpy as np

from songo_model_stockfish.adapters import songo_ai_game
from songo_model_stockfish.ops.job import JobContext
from songo_model_stockfish.ops.logging import utc_now_iso
from songo_model_stockfish.training.features import encode_raw_state


def _slugify_matchup(matchup_spec: str) -> str:
    return matchup_spec.replace(":", "_").replace(" ", "").replace("/", "_")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _resolve_storage_path(base: Path, configured: str | None, fallback: Path) -> Path:
    if not configured:
        return fallback
    path = Path(configured)
    if path.is_absolute():
        return path
    return base / path


def _append_jsonl(path: Path, payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _build_external_agent(spec: str):
    from songo_model_stockfish.reference_songo.agents import MCTSAgent, MinimaxAgent

    kind, level = spec.split(":", 1)
    if kind == "minimax":
        return MinimaxAgent(level)
    if kind == "mcts":
        return MCTSAgent(level)
    raise ValueError(f"Unsupported dataset generation agent: {spec}")


def _teacher_choose(state: Any, *, engine: str, level: str) -> tuple[int, dict[str, Any]]:
    if engine == "minimax":
        from songo_model_stockfish.reference_songo.levels import get_config
        from songo_model_stockfish.reference_songo.minimax import choose_move

        return choose_move(songo_ai_game.clone_state(state), get_config(level))
    if engine == "mcts":
        from songo_model_stockfish.reference_songo.levels import get_mcts_config
        from songo_model_stockfish.reference_songo.mcts import choose_move

        return choose_move(songo_ai_game.clone_state(state), get_mcts_config(level))
    raise ValueError(f"Unsupported teacher engine: {engine}")


def _parse_matchup(matchup_spec: str) -> tuple[str, str]:
    parts = matchup_spec.split(" vs ")
    if len(parts) != 2:
        raise ValueError(f"Invalid matchup spec: {matchup_spec}")
    return parts[0].strip(), parts[1].strip()


def _sample_position(
    *,
    game_id: str,
    matchup_id: str,
    sample_index: int,
    ply: int,
    seed: int,
    state: Any,
) -> dict[str, Any]:
    raw_state = songo_ai_game.to_raw_state(state)
    raw_state["turn_index"] = ply
    legal_moves = songo_ai_game.legal_moves(state)
    effective_terminal = bool(raw_state.get("is_terminal", False)) or not legal_moves
    raw_state["is_terminal"] = effective_terminal
    return {
        "sample_id": f"{game_id}_sample_{sample_index:06d}",
        "game_id": game_id,
        "matchup_id": matchup_id,
        "ply": ply,
        "seed": seed,
        "player_to_move": raw_state["player_to_move"],
        "state": raw_state,
        "legal_moves": [] if effective_terminal else legal_moves,
        "source_engine": "match_replay",
        "source_level": "mixed",
    }


def _raw_state_to_runtime_state(raw_state: dict[str, Any]) -> Any:
    board_flat = list(raw_state["board"])
    if len(board_flat) != 14:
        raise ValueError("Invalid raw_state board length")
    current_player = 0 if raw_state["player_to_move"] == "south" else 1
    scores = raw_state["scores"]
    return {
        "board": [board_flat[:7], board_flat[7:]],
        "scores": [int(scores["south"]), int(scores["north"])],
        "current_player": current_player,
        "finished": bool(raw_state.get("is_terminal", False)),
        "winner": None,
        "reason": "",
        "turn_index": int(raw_state.get("turn_index", 0)),
    }


def _normalize_value(value: float) -> float:
    clipped = max(-1_000_000.0, min(1_000_000.0, float(value)))
    return float(np.tanh(clipped / 200.0))


def _build_policy_distribution(best_move: int, legal_moves: list[int]) -> dict[str, float]:
    distribution = {str(move): 0.0 for move in legal_moves}
    if not legal_moves:
        return distribution
    if best_move not in legal_moves:
        best_move = legal_moves[0]
    if len(legal_moves) == 1:
        distribution[str(best_move)] = 1.0
        return distribution
    off_value = 0.25 / float(len(legal_moves) - 1)
    for move in legal_moves:
        distribution[str(move)] = off_value
    distribution[str(best_move)] = 0.75
    return distribution


def _label_sample(sample: dict[str, Any], *, teacher_engine: str, teacher_level: str) -> dict[str, Any]:
    if bool(sample["state"].get("is_terminal", False)):
        raise ValueError(f"Cannot label terminal sample: {sample['sample_id']}")

    if not sample["legal_moves"]:
        raise ValueError(f"Cannot label sample without legal moves: {sample['sample_id']}")

    runtime_state = _raw_state_to_runtime_state(sample["state"])
    best_move, info = _teacher_choose(runtime_state, engine=teacher_engine, level=teacher_level)
    legal_moves = list(sample["legal_moves"])
    if best_move not in legal_moves and legal_moves:
        best_move = legal_moves[0]

    teacher_score = 0.0
    if teacher_engine == "minimax":
        teacher_score = float(info.get("score", 0.0))
    elif teacher_engine == "mcts":
        root_q = info.get("root_q", {})
        teacher_score = float(root_q.get(best_move, 0.0))

    labeled = dict(sample)
    labeled["teacher_engine"] = teacher_engine
    labeled["teacher_level"] = teacher_level
    labeled["policy_target"] = {
        "best_move": int(best_move),
        "distribution": _build_policy_distribution(int(best_move), legal_moves),
    }
    labeled["value_target"] = _normalize_value(teacher_score)
    return labeled


def _encode_features(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, float]:
    features, legal_mask = encode_raw_state(sample["state"], sample["legal_moves"])
    policy_index = int(sample["policy_target"]["best_move"]) - 1
    value = float(sample["value_target"])
    return features, legal_mask, policy_index, value


def _play_and_sample_game(
    agent_a,
    agent_b,
    *,
    matchup_id: str,
    game_id: str,
    seed: int,
    starter: int,
    sample_every_n_plies: int,
    max_moves: int = 300,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = songo_ai_game.create_state()
    agents = [agent_a, agent_b] if starter == 0 else [agent_b, agent_a]
    moves: list[int] = []
    samples: list[dict[str, Any]] = []
    ply = 0
    sample_index = 0
    started_at = utc_now_iso()
    end_reason = "finished"

    while not songo_ai_game.is_terminal(state) and ply < max_moves:
        if ply % sample_every_n_plies == 0:
            sample_index += 1
            samples.append(
                _sample_position(
                    game_id=game_id,
                    matchup_id=matchup_id,
                    sample_index=sample_index,
                    ply=ply,
                    seed=seed,
                    state=state,
                )
            )

        legal = songo_ai_game.legal_moves(state)
        if not legal:
            end_reason = "no_legal_moves_available"
            break

        current = songo_ai_game.current_player(state)
        try:
            move, _info = agents[current].choose(songo_ai_game.clone_state(state))
        except Exception:
            if legal:
                move = legal[0]
            else:
                end_reason = "agent_failed_no_legal_moves"
                break
        if move not in legal:
            move = legal[0]
        moves.append(int(move))
        state = songo_ai_game.simulate_move(state, int(move))
        ply += 1

    if sample_every_n_plies > 0 and (not samples or samples[-1]["ply"] != ply):
        sample_index += 1
        samples.append(
            _sample_position(
                game_id=game_id,
                matchup_id=matchup_id,
                sample_index=sample_index,
                ply=ply,
                seed=seed,
                state=state,
            )
        )

    winner = songo_ai_game.winner(state)
    if starter == 1 and winner is not None:
        logical_winner = "player_a" if winner == 1 else "player_b"
    elif winner is not None:
        logical_winner = "player_a" if winner == 0 else "player_b"
    else:
        logical_winner = None

    raw_log = {
        "game_id": game_id,
        "matchup_id": matchup_id,
        "seed": seed,
        "starter": starter,
        "player_a": agent_a.display_name,
        "player_b": agent_b.display_name,
        "winner": logical_winner,
        "moves": moves,
        "ply_count": ply,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "scores": list(songo_ai_game.scores(state)),
        "reason": "finished" if songo_ai_game.is_terminal(state) else (end_reason if end_reason != "finished" else f"max_moves_reached:{max_moves}"),
    }
    return raw_log, samples


def _play_and_sample_game_from_specs(
    matchup_a: str,
    matchup_b: str,
    *,
    matchup_id: str,
    game_id: str,
    seed: int,
    starter: int,
    sample_every_n_plies: int,
    max_moves: int = 300,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    agent_a = _build_external_agent(matchup_a)
    agent_b = _build_external_agent(matchup_b)
    return _play_and_sample_game(
        agent_a,
        agent_b,
        matchup_id=matchup_id,
        game_id=game_id,
        seed=seed,
        starter=starter,
        sample_every_n_plies=sample_every_n_plies,
        max_moves=max_moves,
    )


def run_dataset_generation(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("dataset_generation", {})
    runtime_cfg = job.config.get("runtime", {})
    games = int(cfg.get("games", 20))
    matchups = list(cfg.get("matchups", []))
    sample_every_n_plies = int(cfg.get("sample_every_n_plies", 2))
    if sample_every_n_plies <= 0:
        raise ValueError("`sample_every_n_plies` doit etre strictement positif")
    base_seed = int(job.config.get("runtime", {}).get("seed", 42))
    max_moves = int(cfg.get("max_moves", 300))
    num_workers = max(1, int(runtime_cfg.get("num_workers", 1)))
    max_pending_futures = max(1, int(cfg.get("max_pending_futures", num_workers * 2)))

    dataset_dir = job.job_dir / "dataset_generation"
    raw_dir = _resolve_storage_path(job.paths.drive_root, cfg.get("output_raw_dir"), dataset_dir / "raw_match_logs")
    sampled_dir = _resolve_storage_path(job.paths.drive_root, cfg.get("output_sampled_dir"), dataset_dir / "sampled_positions")
    raw_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    state = job.read_state()
    completed_matchups = set(state.get("completed_matchups", []))
    summaries: list[dict[str, Any]] = []

    job.logger.info("dataset generation started")
    job.set_phase("dataset_generation")
    job.write_event("dataset_generation_started")
    job.write_metric({"metric_type": "dataset_generation_started"})

    for matchup_index, matchup_spec in enumerate(matchups):
        matchup_a, matchup_b = _parse_matchup(str(matchup_spec))
        matchup_id = _slugify_matchup(str(matchup_spec))
        job.set_phase(f"dataset_generation:{matchup_id}")
        summary_path = dataset_dir / f"{matchup_id}_summary.json"
        if matchup_id in completed_matchups and summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            job.logger.info("dataset matchup skipped (already completed): %s", matchup_spec)
            continue

        matchup_game_count = 0
        matchup_sample_count = 0
        job.logger.info(
            "dataset matchup started | %s/%s | %s vs %s | games=%s | sample_every=%s | workers=%s",
            matchup_index + 1,
            len(matchups),
            matchup_a,
            matchup_b,
            games,
            sample_every_n_plies,
            num_workers,
        )
        job.write_event(
            "dataset_matchup_started",
            matchup=matchup_id,
            matchup_index=matchup_index + 1,
            total_matchups=len(matchups),
            player_a=matchup_a,
            player_b=matchup_b,
            games=games,
            sample_every_n_plies=sample_every_n_plies,
            num_workers=num_workers,
        )

        pending_games: list[dict[str, Any]] = []
        for game_index in range(games):
            starter = game_index % 2
            seed = base_seed + (matchup_index * 1_000_000) + game_index
            game_id = f"{matchup_id}_game_{game_index + 1:06d}"
            raw_game_path = raw_dir / matchup_id / f"{game_id}.json"
            sampled_game_path = sampled_dir / matchup_id / f"{game_id}.jsonl"
            if raw_game_path.exists() and sampled_game_path.exists():
                raw_payload = json.loads(raw_game_path.read_text(encoding="utf-8"))
                sample_count = _count_jsonl_lines(sampled_game_path)
                job.logger.info(
                    "dataset game reused | matchup=%s | game=%s/%s | moves=%s | samples=%s | winner=%s",
                    matchup_id,
                    game_index + 1,
                    games,
                    raw_payload["ply_count"],
                    sample_count,
                    raw_payload["winner"],
                )
                matchup_game_count += 1
                matchup_sample_count += sample_count
                job.write_state(
                    {
                        "completed_matchups": sorted(completed_matchups),
                        "current_matchup": matchup_id,
                        "completed_games": matchup_game_count,
                        "remaining_games": games - matchup_game_count,
                        "last_completed_game_id": game_id,
                        "sample_count": matchup_sample_count,
                    }
                )
                continue

            pending_games.append(
                {
                    "game_index": game_index,
                    "game_no": game_index + 1,
                    "starter": starter,
                    "seed": seed,
                    "game_id": game_id,
                    "raw_game_path": raw_game_path,
                    "sampled_game_path": sampled_game_path,
                }
            )

        if not pending_games:
            job.logger.info("dataset matchup already fully materialized on disk | matchup=%s", matchup_id)
        elif num_workers <= 1:
            agent_a = _build_external_agent(matchup_a)
            agent_b = _build_external_agent(matchup_b)
            for pending in pending_games:
                job.logger.info(
                    "dataset game running | matchup=%s | game=%s/%s | seed=%s | starter=%s | mode=sequential",
                    matchup_id,
                    pending["game_no"],
                    games,
                    pending["seed"],
                    pending["starter"],
                )
                raw_payload, samples = _play_and_sample_game(
                    agent_a,
                    agent_b,
                    matchup_id=matchup_id,
                    game_id=str(pending["game_id"]),
                    seed=int(pending["seed"]),
                    starter=int(pending["starter"]),
                    sample_every_n_plies=sample_every_n_plies,
                    max_moves=max_moves,
                )
                _write_json(Path(pending["raw_game_path"]), raw_payload)
                sampled_game_path = Path(pending["sampled_game_path"])
                sampled_game_path.parent.mkdir(parents=True, exist_ok=True)
                with sampled_game_path.open("w", encoding="utf-8") as handle:
                    for sample in samples:
                        handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
                sample_count = len(samples)
                job.logger.info(
                    "dataset game completed | matchup=%s | game=%s/%s | moves=%s | samples=%s | winner=%s",
                    matchup_id,
                    pending["game_no"],
                    games,
                    raw_payload["ply_count"],
                    sample_count,
                    raw_payload["winner"],
                )
                job.write_event(
                    "dataset_game_completed",
                    matchup=matchup_id,
                    game_id=pending["game_id"],
                    game_index=pending["game_no"],
                    samples=sample_count,
                    winner=raw_payload["winner"],
                    execution_mode="sequential",
                )
                job.write_metric(
                    {
                        "metric_type": "dataset_game_completed",
                        "matchup_id": matchup_id,
                        "game_id": pending["game_id"],
                        "samples": sample_count,
                        "moves": raw_payload["ply_count"],
                    }
                )
                matchup_game_count += 1
                matchup_sample_count += sample_count
                job.write_state(
                    {
                        "completed_matchups": sorted(completed_matchups),
                        "current_matchup": matchup_id,
                        "completed_games": matchup_game_count,
                        "remaining_games": games - matchup_game_count,
                        "last_completed_game_id": pending["game_id"],
                        "sample_count": matchup_sample_count,
                    }
                )
        else:
            job.logger.info(
                "dataset matchup parallel execution | matchup=%s | workers=%s | pending_games=%s | max_pending=%s",
                matchup_id,
                num_workers,
                len(pending_games),
                max_pending_futures,
            )
            job.write_event(
                "dataset_parallel_execution_started",
                matchup=matchup_id,
                workers=num_workers,
                pending_games=len(pending_games),
                max_pending_futures=max_pending_futures,
            )

            future_map: dict[concurrent.futures.Future, dict[str, Any]] = {}
            pending_queue = list(pending_games)
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                while pending_queue or future_map:
                    while pending_queue and len(future_map) < max_pending_futures:
                        pending = pending_queue.pop(0)
                        job.logger.info(
                            "dataset game scheduled | matchup=%s | game=%s/%s | seed=%s | starter=%s | mode=parallel",
                            matchup_id,
                            pending["game_no"],
                            games,
                            pending["seed"],
                            pending["starter"],
                        )
                        future = executor.submit(
                            _play_and_sample_game_from_specs,
                            matchup_a,
                            matchup_b,
                            matchup_id=matchup_id,
                            game_id=str(pending["game_id"]),
                            seed=int(pending["seed"]),
                            starter=int(pending["starter"]),
                            sample_every_n_plies=sample_every_n_plies,
                            max_moves=max_moves,
                        )
                        future_map[future] = pending

                    done, _not_done = concurrent.futures.wait(
                        future_map.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        pending = future_map.pop(future)
                        raw_payload, samples = future.result()
                        _write_json(Path(pending["raw_game_path"]), raw_payload)
                        sampled_game_path = Path(pending["sampled_game_path"])
                        sampled_game_path.parent.mkdir(parents=True, exist_ok=True)
                        with sampled_game_path.open("w", encoding="utf-8") as handle:
                            for sample in samples:
                                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
                        sample_count = len(samples)
                        job.logger.info(
                            "dataset game completed | matchup=%s | game=%s/%s | moves=%s | samples=%s | winner=%s | mode=parallel",
                            matchup_id,
                            pending["game_no"],
                            games,
                            raw_payload["ply_count"],
                            sample_count,
                            raw_payload["winner"],
                        )
                        job.write_event(
                            "dataset_game_completed",
                            matchup=matchup_id,
                            game_id=pending["game_id"],
                            game_index=pending["game_no"],
                            samples=sample_count,
                            winner=raw_payload["winner"],
                            execution_mode="parallel",
                        )
                        job.write_metric(
                            {
                                "metric_type": "dataset_game_completed",
                                "matchup_id": matchup_id,
                                "game_id": pending["game_id"],
                                "samples": sample_count,
                                "moves": raw_payload["ply_count"],
                            }
                        )
                        matchup_game_count += 1
                        matchup_sample_count += sample_count
                        job.write_state(
                            {
                                "completed_matchups": sorted(completed_matchups),
                                "current_matchup": matchup_id,
                                "completed_games": matchup_game_count,
                                "remaining_games": games - matchup_game_count,
                                "last_completed_game_id": pending["game_id"],
                                "sample_count": matchup_sample_count,
                            }
                        )

        summary_payload = {
            "matchup_id": matchup_id,
            "player_a": matchup_a,
            "player_b": matchup_b,
            "games": matchup_game_count,
            "samples": matchup_sample_count,
            "sample_every_n_plies": sample_every_n_plies,
            "num_workers": num_workers,
        }
        _write_json(summary_path, summary_payload)
        summaries.append(summary_payload)
        completed_matchups.add(matchup_id)
        job.logger.info(
            "dataset matchup completed | %s vs %s | games=%s | samples=%s",
            matchup_a,
            matchup_b,
            matchup_game_count,
            matchup_sample_count,
        )
        job.write_state(
            {
                "completed_matchups": sorted(completed_matchups),
                "current_matchup": matchup_id,
                "completed_games": matchup_game_count,
                "remaining_games": 0,
                "last_completed_game_id": f"{matchup_id}_game_{matchup_game_count:06d}",
                "sample_count": matchup_sample_count,
            }
        )
        job.write_metric({"metric_type": "dataset_matchup_completed", **summary_payload})
        job.write_event("dataset_matchup_completed", matchup=matchup_id, samples=matchup_sample_count)

    summary = {"job_id": job.job_id, "matchups": summaries}
    _write_json(dataset_dir / "dataset_generation_summary.json", summary)
    total_games = sum(int(item["games"]) for item in summaries)
    total_samples = sum(int(item["samples"]) for item in summaries)
    job.logger.info(
        "dataset generation completed | matchups=%s | games=%s | samples=%s",
        len(summaries),
        total_games,
        total_samples,
    )
    job.write_state(
        {
            "completed_matchups": sorted(completed_matchups),
            "current_matchup": None,
            "completed_games": total_games,
            "remaining_games": 0,
            "last_completed_game_id": state.get("last_completed_game_id"),
            "sample_count": total_samples,
        }
    )
    return summary


def run_dataset_build(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("dataset_build", {})
    teacher_cfg = cfg.get("teacher", {})
    teacher_engine = str(teacher_cfg.get("engine", "minimax"))
    teacher_level = str(teacher_cfg.get("level", "hard"))
    dataset_id = str(cfg.get("dataset_id", "dataset_v1"))
    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.8))
    validation_ratio = float(split_cfg.get("validation", 0.1))

    dataset_dir = job.job_dir / "dataset_build"
    sampled_root = _resolve_storage_path(job.paths.drive_root, cfg.get("input_sampled_dir"), dataset_dir.parent / "dataset_generation" / "sampled_positions")
    labeled_root = dataset_dir / "labeled_positions"
    output_root = _resolve_storage_path(job.paths.drive_root, cfg.get("output_dir"), dataset_dir / "datasets" / dataset_id)
    labeled_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    sampled_files = sorted(sampled_root.rglob("*.jsonl"))
    if not sampled_files:
        raise FileNotFoundError(f"Aucun fichier sampled_positions trouve dans {sampled_root}")

    state = job.read_state()
    completed_files = set(state.get("completed_files", []))
    labeled_samples: list[dict[str, Any]] = []
    processed_count = 0
    skipped_terminal_samples = 0
    skipped_no_legal_samples = 0

    job.logger.info("dataset build started")
    job.set_phase("dataset_build")
    job.write_event("dataset_build_started", teacher_engine=teacher_engine, teacher_level=teacher_level)
    job.write_metric({"metric_type": "dataset_build_started"})

    for sampled_file in sampled_files:
        relative_name = str(sampled_file.relative_to(sampled_root))
        output_labeled = labeled_root / relative_name
        output_labeled.parent.mkdir(parents=True, exist_ok=True)

        if relative_name in completed_files and output_labeled.exists():
            file_samples = list(_iter_jsonl(output_labeled))
        else:
            source_samples = list(_iter_jsonl(sampled_file))
            file_samples = []
            for sample in source_samples:
                if bool(sample["state"].get("is_terminal", False)):
                    skipped_terminal_samples += 1
                    continue
                if not sample["legal_moves"]:
                    skipped_no_legal_samples += 1
                    continue
                file_samples.append(_label_sample(sample, teacher_engine=teacher_engine, teacher_level=teacher_level))
            with output_labeled.open("w", encoding="utf-8") as handle:
                for sample in file_samples:
                    handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
            completed_files.add(relative_name)
            job.write_event("dataset_labeled_file_completed", file=relative_name, samples=len(file_samples))
            job.write_metric({"metric_type": "dataset_labeled_file_completed", "file": relative_name, "samples": len(file_samples)})

        labeled_samples.extend(file_samples)
        processed_count += 1
        job.write_state(
            {
                "completed_files": sorted(completed_files),
                "processed_files": processed_count,
                "remaining_files": len(sampled_files) - processed_count,
                "labeled_samples": len(labeled_samples),
                "skipped_terminal_samples": skipped_terminal_samples,
                "skipped_no_legal_samples": skipped_no_legal_samples,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in labeled_samples:
        grouped.setdefault(str(sample["game_id"]), []).append(sample)

    game_ids = sorted(grouped)
    split_train_end = int(len(game_ids) * train_ratio)
    split_validation_end = split_train_end + int(len(game_ids) * validation_ratio)
    split_game_ids = {
        "train": set(game_ids[:split_train_end]),
        "validation": set(game_ids[split_train_end:split_validation_end]),
        "test": set(game_ids[split_validation_end:]),
    }

    split_summary: dict[str, dict[str, int]] = {}
    for split_name, selected_game_ids in split_game_ids.items():
        split_samples: list[dict[str, Any]] = []
        for game_id in sorted(selected_game_ids):
            split_samples.extend(grouped[game_id])

        features_list = []
        masks_list = []
        policy_list = []
        value_list = []
        sample_ids = []
        game_id_list = []
        for sample in split_samples:
            features, legal_mask, policy_index, value = _encode_features(sample)
            features_list.append(features)
            masks_list.append(legal_mask)
            policy_list.append(policy_index)
            value_list.append(value)
            sample_ids.append(str(sample["sample_id"]))
            game_id_list.append(str(sample["game_id"]))

        x = np.asarray(features_list, dtype=np.float32) if features_list else np.zeros((0, 17), dtype=np.float32)
        legal_mask = np.asarray(masks_list, dtype=np.float32) if masks_list else np.zeros((0, 7), dtype=np.float32)
        policy_index = np.asarray(policy_list, dtype=np.int64) if policy_list else np.zeros((0,), dtype=np.int64)
        value_target = np.asarray(value_list, dtype=np.float32) if value_list else np.zeros((0,), dtype=np.float32)
        np.savez_compressed(
            output_root / f"{split_name}.npz",
            x=x,
            legal_mask=legal_mask,
            policy_index=policy_index,
            value_target=value_target,
            sample_ids=np.asarray(sample_ids, dtype=object),
            game_ids=np.asarray(game_id_list, dtype=object),
        )
        split_summary[split_name] = {"games": len(selected_game_ids), "samples": len(split_samples)}

    summary = {
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "teacher_engine": teacher_engine,
        "teacher_level": teacher_level,
        "splits": split_summary,
        "output_dir": str(output_root),
        "skipped_terminal_samples": skipped_terminal_samples,
        "skipped_no_legal_samples": skipped_no_legal_samples,
    }
    _write_json(dataset_dir / "dataset_build_summary.json", summary)
    job.write_metric({"metric_type": "dataset_build_completed", "dataset_id": dataset_id})
    job.write_event("dataset_build_completed", dataset_id=dataset_id)
    return summary
