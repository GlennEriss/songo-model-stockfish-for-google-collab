from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
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


def _count_total_jsonl_lines(root: Path) -> int:
    total = 0
    for path in sorted(root.rglob("*.jsonl")):
        total += _count_jsonl_lines(path)
    return total


def _existing_game_numbers(raw_dir: Path, sampled_dir: Path, matchup_id: str) -> set[int]:
    raw_matchup_dir = raw_dir / matchup_id
    sampled_matchup_dir = sampled_dir / matchup_id
    raw_stems = {path.stem for path in raw_matchup_dir.glob("*.json")} if raw_matchup_dir.exists() else set()
    sampled_stems = {path.stem for path in sampled_matchup_dir.glob("*.jsonl")} if sampled_matchup_dir.exists() else set()
    completed = raw_stems & sampled_stems
    game_numbers: set[int] = set()
    prefix = f"{matchup_id}_game_"
    for stem in completed:
        if stem.startswith(prefix):
            suffix = stem[len(prefix) :]
            if suffix.isdigit():
                game_numbers.add(int(suffix))
    return game_numbers


def _build_pending_incremental_games(
    *,
    raw_dir: Path,
    sampled_dir: Path,
    matchup_id: str,
    games_to_add: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    existing_numbers = _existing_game_numbers(raw_dir, sampled_dir, matchup_id)
    pending: list[dict[str, Any]] = []
    candidate = 1
    while len(pending) < games_to_add:
        game_id = f"{matchup_id}_game_{candidate:06d}"
        raw_game_path = raw_dir / matchup_id / f"{game_id}.json"
        sampled_game_path = sampled_dir / matchup_id / f"{game_id}.jsonl"
        if candidate not in existing_numbers or not (raw_game_path.exists() and sampled_game_path.exists()):
            pending.append(
                {
                    "game_index": candidate - 1,
                    "game_no": candidate,
                    "starter": (candidate - 1) % 2,
                    "seed": seed_base + (candidate - 1),
                    "game_id": game_id,
                    "raw_game_path": raw_game_path,
                    "sampled_game_path": sampled_game_path,
                }
            )
        candidate += 1
    return pending


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


def _label_samples_from_file(
    sampled_file_path: str,
    *,
    teacher_engine: str,
    teacher_level: str,
) -> tuple[int, list[dict[str, Any]], int, int]:
    source_samples = list(_iter_jsonl(Path(sampled_file_path)))
    source_count = len(source_samples)
    labeled_samples: list[dict[str, Any]] = []
    skipped_terminal = 0
    skipped_no_legal = 0
    for sample in source_samples:
        if bool(sample["state"].get("is_terminal", False)):
            skipped_terminal += 1
            continue
        if not sample["legal_moves"]:
            skipped_no_legal += 1
            continue
        labeled_samples.append(_label_sample(sample, teacher_engine=teacher_engine, teacher_level=teacher_level))
    return source_count, labeled_samples, skipped_terminal, skipped_no_legal


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


def _materialize_completed_game(
    job: JobContext,
    *,
    pending: dict[str, Any],
    raw_payload: dict[str, Any],
    samples: list[dict[str, Any]],
    matchup_id: str,
    games: int,
    execution_mode: str,
) -> tuple[int, int]:
    _write_json(Path(pending["raw_game_path"]), raw_payload)
    sampled_game_path = Path(pending["sampled_game_path"])
    sampled_game_path.parent.mkdir(parents=True, exist_ok=True)
    with sampled_game_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")

    sample_count = len(samples)
    job.logger.info(
        "dataset game completed | matchup=%s | game=%s/%s | moves=%s | samples=%s | winner=%s | mode=%s",
        matchup_id,
        pending["game_no"],
        games,
        raw_payload["ply_count"],
        sample_count,
        raw_payload["winner"],
        execution_mode,
    )
    job.write_event(
        "dataset_game_completed",
        matchup=matchup_id,
        game_id=pending["game_id"],
        game_index=pending["game_no"],
        samples=sample_count,
        winner=raw_payload["winner"],
        execution_mode=execution_mode,
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
    return 1, sample_count


def _run_pending_games_sequential(
    job: JobContext,
    *,
    pending_games: list[dict[str, Any]],
    matchup_a: str,
    matchup_b: str,
    matchup_id: str,
    games: int,
    sample_every_n_plies: int,
    max_moves: int,
) -> tuple[int, int, str]:
    agent_a = _build_external_agent(matchup_a)
    agent_b = _build_external_agent(matchup_b)
    completed_games = 0
    completed_samples = 0
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
        games_inc, samples_inc = _materialize_completed_game(
            job,
            pending=pending,
            raw_payload=raw_payload,
            samples=samples,
            matchup_id=matchup_id,
            games=games,
            execution_mode="sequential",
        )
        completed_games += games_inc
        completed_samples += samples_inc
    return completed_games, completed_samples, "sequential"


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
    multiprocessing_start_method = str(runtime_cfg.get("multiprocessing_start_method", "spawn")).strip().lower() or "spawn"
    max_tasks_per_child = int(runtime_cfg.get("max_tasks_per_child", 25))
    target_samples = int(cfg.get("target_samples", 0))

    dataset_dir = job.job_dir / "dataset_generation"
    raw_dir = _resolve_storage_path(job.paths.drive_root, cfg.get("output_raw_dir"), dataset_dir / "raw_match_logs")
    sampled_dir = _resolve_storage_path(job.paths.drive_root, cfg.get("output_sampled_dir"), dataset_dir / "sampled_positions")
    raw_dir.mkdir(parents=True, exist_ok=True)
    sampled_dir.mkdir(parents=True, exist_ok=True)

    state = job.read_state()
    summaries: list[dict[str, Any]] = []
    initial_total_samples = _count_total_jsonl_lines(sampled_dir)
    total_samples_after_run = initial_total_samples

    job.logger.info("dataset generation started")
    job.set_phase("dataset_generation")
    job.write_event(
        "dataset_generation_started",
        existing_samples=initial_total_samples,
        target_samples=target_samples,
    )
    job.write_metric(
        {
            "metric_type": "dataset_generation_started",
            "existing_samples": initial_total_samples,
            "target_samples": target_samples,
        }
    )

    if target_samples > 0 and initial_total_samples >= target_samples:
        summary = {
            "job_id": job.job_id,
            "matchups": [],
            "existing_samples": initial_total_samples,
            "added_samples": 0,
            "total_samples": initial_total_samples,
            "target_samples": target_samples,
        }
        _write_json(dataset_dir / "dataset_generation_summary.json", summary)
        job.logger.info(
            "dataset generation skipped | target already reached | existing_samples=%s | target_samples=%s",
            initial_total_samples,
            target_samples,
        )
        job.write_state(
            {
                "current_matchup": None,
                "completed_games": 0,
                "remaining_games": 0,
                "last_completed_game_id": state.get("last_completed_game_id"),
                "sample_count": initial_total_samples,
                "target_samples": target_samples,
            }
        )
        return summary

    for matchup_index, matchup_spec in enumerate(matchups):
        matchup_a, matchup_b = _parse_matchup(str(matchup_spec))
        matchup_id = _slugify_matchup(str(matchup_spec))
        job.set_phase(f"dataset_generation:{matchup_id}")
        summary_path = dataset_dir / f"{matchup_id}_summary.json"
        if target_samples > 0 and total_samples_after_run >= target_samples:
            job.logger.info(
                "dataset generation target reached during run | current_samples=%s | target_samples=%s",
                total_samples_after_run,
                target_samples,
            )
            break

        matchup_game_count = 0
        matchup_sample_count = 0
        existing_numbers = _existing_game_numbers(raw_dir, sampled_dir, matchup_id)
        existing_game_count = len(existing_numbers)
        job.logger.info(
            "dataset matchup started | %s/%s | %s vs %s | add_games=%s | existing_games=%s | sample_every=%s | workers=%s | target_samples=%s",
            matchup_index + 1,
            len(matchups),
            matchup_a,
            matchup_b,
            games,
            existing_game_count,
            sample_every_n_plies,
            num_workers,
            target_samples,
        )
        job.write_event(
            "dataset_matchup_started",
            matchup=matchup_id,
            matchup_index=matchup_index + 1,
            total_matchups=len(matchups),
            player_a=matchup_a,
            player_b=matchup_b,
            add_games=games,
            existing_games=existing_game_count,
            sample_every_n_plies=sample_every_n_plies,
            num_workers=num_workers,
        )

        pending_games = _build_pending_incremental_games(
            raw_dir=raw_dir,
            sampled_dir=sampled_dir,
            matchup_id=matchup_id,
            games_to_add=games,
            seed_base=base_seed + (matchup_index * 1_000_000),
        )

        if num_workers <= 1:
            completed_games, completed_samples, _execution_mode = _run_pending_games_sequential(
                job,
                pending_games=pending_games,
                matchup_a=matchup_a,
                matchup_b=matchup_b,
                matchup_id=matchup_id,
                games=games,
                sample_every_n_plies=sample_every_n_plies,
                max_moves=max_moves,
            )
            matchup_game_count += completed_games
            matchup_sample_count += completed_samples
            if pending_games:
                job.write_state(
                    {
                        "current_matchup": matchup_id,
                        "completed_games": matchup_game_count,
                        "remaining_games": games - matchup_game_count,
                        "last_completed_game_id": pending_games[-1]["game_id"],
                        "sample_count": total_samples_after_run + matchup_sample_count,
                        "target_samples": target_samples,
                    }
                )
        else:
            job.logger.info(
                "dataset matchup parallel execution | matchup=%s | workers=%s | pending_games=%s | max_pending=%s | start_method=%s | max_tasks_per_child=%s",
                matchup_id,
                num_workers,
                len(pending_games),
                max_pending_futures,
                multiprocessing_start_method,
                max_tasks_per_child,
            )
            job.write_event(
                "dataset_parallel_execution_started",
                matchup=matchup_id,
                workers=num_workers,
                pending_games=len(pending_games),
                max_pending_futures=max_pending_futures,
                multiprocessing_start_method=multiprocessing_start_method,
                max_tasks_per_child=max_tasks_per_child,
            )

            future_map: dict[concurrent.futures.Future, dict[str, Any]] = {}
            pending_queue = list(pending_games)
            try:
                mp_context = multiprocessing.get_context(multiprocessing_start_method)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=mp_context,
                    max_tasks_per_child=max_tasks_per_child,
                ) as executor:
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
                            games_inc, samples_inc = _materialize_completed_game(
                                job,
                                pending=pending,
                                raw_payload=raw_payload,
                                samples=samples,
                                matchup_id=matchup_id,
                                games=games,
                                execution_mode="parallel",
                            )
                            matchup_game_count += games_inc
                            matchup_sample_count += samples_inc
                            total_samples_after_run += samples_inc
                            job.write_state(
                                {
                                    "current_matchup": matchup_id,
                                    "completed_games": matchup_game_count,
                                    "remaining_games": games - matchup_game_count,
                                    "last_completed_game_id": pending["game_id"],
                                    "sample_count": total_samples_after_run,
                                    "target_samples": target_samples,
                                }
                            )
            except concurrent.futures.process.BrokenProcessPool as exc:
                failed_pending = list(future_map.values()) + list(pending_queue)
                job.logger.warning(
                    "dataset parallel pool broken | matchup=%s | completed_games=%s | remaining_fallback=%s | error=%s",
                    matchup_id,
                    matchup_game_count,
                    len(failed_pending),
                    exc,
                )
                job.write_event(
                    "dataset_parallel_execution_broken",
                    matchup=matchup_id,
                    completed_games=matchup_game_count,
                    remaining_fallback=len(failed_pending),
                    error=str(exc),
                )
                completed_games, completed_samples, _execution_mode = _run_pending_games_sequential(
                    job,
                    pending_games=failed_pending,
                    matchup_a=matchup_a,
                    matchup_b=matchup_b,
                    matchup_id=matchup_id,
                    games=games,
                    sample_every_n_plies=sample_every_n_plies,
                    max_moves=max_moves,
                )
                matchup_game_count += completed_games
                matchup_sample_count += completed_samples
                total_samples_after_run += completed_samples
                if failed_pending:
                    job.write_state(
                        {
                            "current_matchup": matchup_id,
                            "completed_games": matchup_game_count,
                            "remaining_games": games - matchup_game_count,
                            "last_completed_game_id": failed_pending[-1]["game_id"],
                            "sample_count": total_samples_after_run,
                            "target_samples": target_samples,
                        }
                    )

        if num_workers <= 1 and pending_games:
            total_samples_after_run += matchup_sample_count

        summary_payload = {
            "matchup_id": matchup_id,
            "player_a": matchup_a,
            "player_b": matchup_b,
            "existing_games": existing_game_count,
            "games_added": matchup_game_count,
            "samples_added": matchup_sample_count,
            "sample_every_n_plies": sample_every_n_plies,
            "num_workers": num_workers,
        }
        _write_json(summary_path, summary_payload)
        summaries.append(summary_payload)
        job.logger.info(
            "dataset matchup completed | %s vs %s | games_added=%s | samples_added=%s | total_samples=%s",
            matchup_a,
            matchup_b,
            matchup_game_count,
            matchup_sample_count,
            total_samples_after_run,
        )
        job.write_state(
            {
                "current_matchup": matchup_id,
                "completed_games": matchup_game_count,
                "remaining_games": 0,
                "last_completed_game_id": pending_games[-1]["game_id"] if pending_games else state.get("last_completed_game_id"),
                "sample_count": total_samples_after_run,
                "target_samples": target_samples,
            }
        )
        job.write_metric({"metric_type": "dataset_matchup_completed", **summary_payload})
        job.write_event("dataset_matchup_completed", matchup=matchup_id, samples=matchup_sample_count)

    added_games = sum(int(item["games_added"]) for item in summaries)
    added_samples = sum(int(item["samples_added"]) for item in summaries)
    summary = {
        "job_id": job.job_id,
        "matchups": summaries,
        "existing_samples": initial_total_samples,
        "added_games": added_games,
        "added_samples": added_samples,
        "total_samples": initial_total_samples + added_samples,
        "target_samples": target_samples,
    }
    _write_json(dataset_dir / "dataset_generation_summary.json", summary)
    job.logger.info(
        "dataset generation completed | matchups=%s | added_games=%s | added_samples=%s | total_samples=%s | target_samples=%s",
        len(summaries),
        added_games,
        added_samples,
        initial_total_samples + added_samples,
        target_samples,
    )
    job.write_state(
        {
            "current_matchup": None,
            "completed_games": added_games,
            "remaining_games": 0,
            "last_completed_game_id": state.get("last_completed_game_id"),
            "sample_count": initial_total_samples + added_samples,
            "target_samples": target_samples,
        }
    )
    return summary


def run_dataset_build(job: JobContext) -> dict[str, object]:
    cfg = job.config.get("dataset_build", {})
    runtime_cfg = job.config.get("runtime", {})
    teacher_cfg = cfg.get("teacher", {})
    teacher_engine = str(teacher_cfg.get("engine", "minimax"))
    teacher_level = str(teacher_cfg.get("level", "hard"))
    dataset_id = str(cfg.get("dataset_id", "dataset_v1"))
    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.8))
    validation_ratio = float(split_cfg.get("validation", 0.1))
    num_workers = max(1, int(cfg.get("num_workers", runtime_cfg.get("num_workers", 1))))
    max_pending_futures = max(1, int(cfg.get("max_pending_futures", num_workers * 2)))
    multiprocessing_start_method = str(runtime_cfg.get("multiprocessing_start_method", "spawn")).strip().lower() or "spawn"
    max_tasks_per_child = int(runtime_cfg.get("max_tasks_per_child", 25))

    dataset_dir = job.job_dir / "dataset_build"
    sampled_root = _resolve_storage_path(job.paths.drive_root, cfg.get("input_sampled_dir"), dataset_dir.parent / "dataset_generation" / "sampled_positions")
    label_cache_dir = _resolve_storage_path(
        job.paths.drive_root,
        cfg.get("label_cache_dir"),
        job.paths.data_root / "label_cache" / dataset_id / f"{teacher_engine}_{teacher_level}",
    )
    labeled_root = label_cache_dir / "labeled_positions"
    output_root = _resolve_storage_path(job.paths.drive_root, cfg.get("output_dir"), dataset_dir / "datasets" / dataset_id)
    labeled_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    target_labeled_samples = int(cfg.get("target_labeled_samples", 0))
    label_cache_metadata_path = label_cache_dir / "metadata.json"
    _write_json(
        label_cache_metadata_path,
        {
            "dataset_id": dataset_id,
            "teacher_engine": teacher_engine,
            "teacher_level": teacher_level,
            "label_cache_dir": str(label_cache_dir),
        },
    )

    sampled_files = sorted(sampled_root.rglob("*.jsonl"))
    if not sampled_files:
        raise FileNotFoundError(f"Aucun fichier sampled_positions trouve dans {sampled_root}")

    state = job.read_state()
    completed_files = set(state.get("completed_files", []))
    file_sample_counts: dict[str, int] = {}
    processed_count = 0
    skipped_terminal_samples = 0
    skipped_no_legal_samples = 0
    log_every_n_files = max(1, int(cfg.get("log_every_n_files", 1)))
    last_logged_progress_count = -1

    def _write_build_state() -> None:
        job.write_state(
            {
                "completed_files": sorted(completed_files),
                "processed_files": processed_count,
                "remaining_files": len(sampled_files) - processed_count,
                "labeled_samples": sum(file_sample_counts.values()),
                "skipped_terminal_samples": skipped_terminal_samples,
                "skipped_no_legal_samples": skipped_no_legal_samples,
                "target_labeled_samples": target_labeled_samples,
            }
        )

    def _log_build_progress_if_needed() -> None:
        nonlocal last_logged_progress_count
        if processed_count == last_logged_progress_count:
            return
        if processed_count % log_every_n_files != 0 and processed_count != len(sampled_files):
            return
        job.logger.info(
            "dataset build progress | files=%s/%s | labeled_samples=%s | skipped_terminal=%s | skipped_no_legal=%s",
            processed_count,
            len(sampled_files),
            sum(file_sample_counts.values()),
            skipped_terminal_samples,
            skipped_no_legal_samples,
        )
        last_logged_progress_count = processed_count

    job.logger.info(
        "dataset build started | dataset=%s | teacher=%s:%s | sampled_root=%s | label_cache=%s | files=%s | target_labeled_samples=%s | workers=%s",
        dataset_id,
        teacher_engine,
        teacher_level,
        sampled_root,
        label_cache_dir,
        len(sampled_files),
        target_labeled_samples,
        num_workers,
    )
    job.set_phase("dataset_build")
    job.write_event(
        "dataset_build_started",
        teacher_engine=teacher_engine,
        teacher_level=teacher_level,
        sampled_root=str(sampled_root),
        label_cache_dir=str(label_cache_dir),
        sampled_files=len(sampled_files),
        target_labeled_samples=target_labeled_samples,
        num_workers=num_workers,
    )
    job.write_metric({"metric_type": "dataset_build_started"})

    pending_files: list[dict[str, Any]] = []
    for file_index, sampled_file in enumerate(sampled_files, start=1):
        relative_name = str(sampled_file.relative_to(sampled_root))
        output_labeled = labeled_root / relative_name
        output_labeled.parent.mkdir(parents=True, exist_ok=True)

        if output_labeled.exists():
            file_samples = list(_iter_jsonl(output_labeled))
            source_count = len(file_samples)
            job.logger.info(
                "dataset build file reused | %s/%s | file=%s | labeled_samples=%s",
                file_index,
                len(sampled_files),
                relative_name,
                len(file_samples),
            )
            completed_files.add(relative_name)
            file_sample_counts[relative_name] = len(file_samples)
            processed_count += 1
        else:
            pending_files.append(
                {
                    "file_index": file_index,
                    "relative_name": relative_name,
                    "sampled_file": sampled_file,
                    "output_labeled": output_labeled,
                }
            )

        _log_build_progress_if_needed()
        _write_build_state()

    reused_count = processed_count
    pending_count = len(pending_files)
    job.logger.info(
        "dataset build scan completed | reused_files=%s | pending_files=%s | total_files=%s | labeled_samples=%s",
        reused_count,
        pending_count,
        len(sampled_files),
        sum(file_sample_counts.values()),
    )
    job.write_event(
        "dataset_build_scan_completed",
        reused_files=reused_count,
        pending_files=pending_count,
        total_files=len(sampled_files),
        labeled_samples=sum(file_sample_counts.values()),
    )

    def _materialize_labeled_file(
        file_item: dict[str, Any],
        source_count: int,
        file_samples: list[dict[str, Any]],
        skipped_terminal: int,
        skipped_no_legal: int,
        *,
        mode: str,
    ) -> None:
        nonlocal processed_count, skipped_terminal_samples, skipped_no_legal_samples
        output_labeled = Path(file_item["output_labeled"])
        output_labeled.parent.mkdir(parents=True, exist_ok=True)
        with output_labeled.open("w", encoding="utf-8") as handle:
            for sample in file_samples:
                handle.write(json.dumps(sample, ensure_ascii=True) + "\n")

        skipped_terminal_samples += skipped_terminal
        skipped_no_legal_samples += skipped_no_legal
        completed_files.add(str(file_item["relative_name"]))
        file_sample_counts[str(file_item["relative_name"])] = len(file_samples)
        processed_count += 1
        job.logger.info(
            "dataset build file completed | %s/%s | file=%s | mode=%s | source_samples=%s | labeled_samples=%s | skipped_terminal=%s | skipped_no_legal=%s",
            file_item["file_index"],
            len(sampled_files),
            file_item["relative_name"],
            mode,
            source_count,
            len(file_samples),
            skipped_terminal_samples,
            skipped_no_legal_samples,
        )
        job.write_event("dataset_labeled_file_completed", file=str(file_item["relative_name"]), samples=len(file_samples))
        job.write_metric({"metric_type": "dataset_labeled_file_completed", "file": str(file_item["relative_name"]), "samples": len(file_samples)})
        _log_build_progress_if_needed()
        _write_build_state()

    if pending_files:
        if num_workers <= 1:
            for file_item in pending_files:
                job.logger.info(
                    "dataset build file started | %s/%s | file=%s | mode=sequential",
                    file_item["file_index"],
                    len(sampled_files),
                    file_item["relative_name"],
                )
                source_count, file_samples, skipped_terminal, skipped_no_legal = _label_samples_from_file(
                    str(file_item["sampled_file"]),
                    teacher_engine=teacher_engine,
                    teacher_level=teacher_level,
                )
                _materialize_labeled_file(
                    file_item,
                    source_count,
                    file_samples,
                    skipped_terminal,
                    skipped_no_legal,
                    mode="sequential",
                )
        else:
            job.logger.info(
                "dataset build parallel execution | workers=%s | pending_files=%s | max_pending=%s | start_method=%s | max_tasks_per_child=%s",
                num_workers,
                len(pending_files),
                max_pending_futures,
                multiprocessing_start_method,
                max_tasks_per_child,
            )
            job.write_event(
                "dataset_build_parallel_execution_started",
                workers=num_workers,
                pending_files=len(pending_files),
                max_pending_futures=max_pending_futures,
                multiprocessing_start_method=multiprocessing_start_method,
                max_tasks_per_child=max_tasks_per_child,
            )
            future_map: dict[concurrent.futures.Future, dict[str, Any]] = {}
            pending_queue = list(pending_files)
            try:
                mp_context = multiprocessing.get_context(multiprocessing_start_method)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=mp_context,
                    max_tasks_per_child=max_tasks_per_child,
                ) as executor:
                    while pending_queue or future_map:
                        while pending_queue and len(future_map) < max_pending_futures:
                            file_item = pending_queue.pop(0)
                            job.logger.info(
                                "dataset build file started | %s/%s | file=%s | mode=parallel",
                                file_item["file_index"],
                                len(sampled_files),
                                file_item["relative_name"],
                            )
                            future = executor.submit(
                                _label_samples_from_file,
                                str(file_item["sampled_file"]),
                                teacher_engine=teacher_engine,
                                teacher_level=teacher_level,
                            )
                            future_map[future] = file_item

                        done, _not_done = concurrent.futures.wait(
                            future_map.keys(),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            file_item = future_map.pop(future)
                            source_count, file_samples, skipped_terminal, skipped_no_legal = future.result()
                            _materialize_labeled_file(
                                file_item,
                                source_count,
                                file_samples,
                                skipped_terminal,
                                skipped_no_legal,
                                mode="parallel",
                            )
            except concurrent.futures.process.BrokenProcessPool as exc:
                failed_pending = list(future_map.values()) + list(pending_queue)
                job.logger.warning(
                    "dataset build parallel pool broken | completed_files=%s | remaining_fallback=%s | error=%s",
                    processed_count,
                    len(failed_pending),
                    exc,
                )
                job.write_event(
                    "dataset_build_parallel_execution_broken",
                    completed_files=processed_count,
                    remaining_fallback=len(failed_pending),
                    error=str(exc),
                )
                for file_item in failed_pending:
                    job.logger.info(
                        "dataset build file started | %s/%s | file=%s | mode=sequential_fallback",
                        file_item["file_index"],
                        len(sampled_files),
                        file_item["relative_name"],
                    )
                    source_count, file_samples, skipped_terminal, skipped_no_legal = _label_samples_from_file(
                        str(file_item["sampled_file"]),
                        teacher_engine=teacher_engine,
                        teacher_level=teacher_level,
                    )
                    _materialize_labeled_file(
                        file_item,
                        source_count,
                        file_samples,
                        skipped_terminal,
                        skipped_no_legal,
                        mode="sequential_fallback",
                    )

    game_files = [str(path.relative_to(sampled_root)) for path in sampled_files]
    split_train_end = int(len(game_files) * train_ratio)
    split_validation_end = split_train_end + int(len(game_files) * validation_ratio)
    split_files = {
        "train": game_files[:split_train_end],
        "validation": game_files[split_train_end:split_validation_end],
        "test": game_files[split_validation_end:],
    }

    split_summary: dict[str, dict[str, int]] = {}
    for split_name, selected_files in split_files.items():
        selected_sample_count = sum(file_sample_counts.get(relative_name, 0) for relative_name in selected_files)
        job.logger.info(
            "dataset build split export started | split=%s | games=%s | samples=%s",
            split_name,
            len(selected_files),
            selected_sample_count,
        )

        features_list = []
        masks_list = []
        policy_list = []
        value_list = []
        sample_ids = []
        game_id_list = []
        for relative_name in selected_files:
            labeled_path = labeled_root / relative_name
            for sample in _iter_jsonl(labeled_path):
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
        split_summary[split_name] = {"games": len(selected_files), "samples": selected_sample_count}
        job.logger.info(
            "dataset build split export completed | split=%s | games=%s | samples=%s | path=%s",
            split_name,
            len(selected_files),
            selected_sample_count,
            output_root / f"{split_name}.npz",
        )

    summary = {
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "teacher_engine": teacher_engine,
        "teacher_level": teacher_level,
        "label_cache_dir": str(label_cache_dir),
        "splits": split_summary,
        "output_dir": str(output_root),
        "labeled_samples": sum(file_sample_counts.values()),
        "target_labeled_samples": target_labeled_samples,
        "skipped_terminal_samples": skipped_terminal_samples,
        "skipped_no_legal_samples": skipped_no_legal_samples,
    }
    _write_json(dataset_dir / "dataset_build_summary.json", summary)
    job.logger.info(
        "dataset build completed | dataset=%s | train=%s | validation=%s | test=%s | skipped_terminal=%s | skipped_no_legal=%s",
        dataset_id,
        split_summary.get("train", {}).get("samples", 0),
        split_summary.get("validation", {}).get("samples", 0),
        split_summary.get("test", {}).get("samples", 0),
        skipped_terminal_samples,
        skipped_no_legal_samples,
    )
    job.write_metric({"metric_type": "dataset_build_completed", "dataset_id": dataset_id})
    job.write_event("dataset_build_completed", dataset_id=dataset_id)
    return summary
