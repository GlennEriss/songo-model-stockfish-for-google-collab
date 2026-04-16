from __future__ import annotations

import concurrent.futures
import json
import math
from pathlib import Path
import time

from songo_model_stockfish.adapters.songo_ai_game import clone_state
from songo_model_stockfish.benchmark.play_match import AgentLike
from songo_model_stockfish.benchmark.model_agent import ModelAgent
from songo_model_stockfish.engine.config import EngineConfig
from songo_model_stockfish.engine.search import choose_move
from songo_model_stockfish.ops.job import JobContext
from songo_model_stockfish.ops.model_registry import (
    latest_model_record,
    load_registry,
    promote_best_model,
    promoted_best_metadata,
    upsert_model_record,
)


class EngineAgent:
    def __init__(self, name: str = "engine_v1", config: EngineConfig | None = None) -> None:
        self._name = name
        self._config = config or EngineConfig()

    @property
    def display_name(self) -> str:
        return self._name

    def choose(self, state):
        move, info = choose_move(clone_state(state), self._config)
        return move, {"best_score": info.best_score, "depth": info.depth_reached}


def _build_external_agent(spec: str) -> AgentLike:
    from songo_model_stockfish.reference_songo.agents import MCTSAgent, MinimaxAgent

    kind, level = spec.split(":", 1)
    if kind == "minimax":
        return MinimaxAgent(level)
    if kind == "mcts":
        return MCTSAgent(level)
    raise ValueError(f"Unsupported benchmark opponent: {spec}")


def _slugify_matchup(opponent_spec: str) -> str:
    return opponent_spec.replace(":", "_").replace(" ", "").replace("/", "_")


def _load_existing_game_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _resolve_storage_path(base: Path, configured: str | None, fallback: Path) -> Path:
    if not configured:
        return fallback
    path = Path(configured)
    if path.is_absolute():
        return path
    return base / path


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y", "t"}:
        return True
    if text in {"0", "false", "no", "off", "n", "f"}:
        return False
    return bool(default)


def _winner_label(winner: object, target_name: str, opponent_name: str) -> str:
    if winner == 0:
        return target_name
    if winner == 1:
        return opponent_name
    return "draw"


def _opponent_weight(opponent_spec: str) -> float:
    base_weights = {
        "minimax:medium": 1.0,
        "mcts:medium": 1.1,
        "minimax:hard": 1.25,
        "mcts:hard": 1.4,
        "minimax:insane": 1.7,
        "mcts:insane": 1.8,
    }
    return float(base_weights.get(opponent_spec, 1.0))


def _opponent_rating(opponent_spec: str, configured: dict[str, object] | None = None) -> float:
    if configured and opponent_spec in configured:
        return float(configured[opponent_spec])
    base_ratings = {
        "minimax:medium": 1200.0,
        "mcts:medium": 1275.0,
        "minimax:hard": 1375.0,
        "mcts:hard": 1450.0,
        "minimax:insane": 1600.0,
        "mcts:insane": 1675.0,
    }
    return float(base_ratings.get(opponent_spec, 1300.0))


def _record_by_role(stats: dict[str, int], winner: object) -> None:
    if winner == 0:
        stats["wins"] += 1
    elif winner == 1:
        stats["losses"] += 1
    else:
        stats["draws"] += 1


def _compute_weighted_benchmark_score(matchups: list[dict[str, object]]) -> float:
    if not matchups:
        return 0.0
    weighted_total = 0.0
    total_weight = 0.0
    for item in matchups:
        opponent = str(item.get("opponent", ""))
        weight = float(item.get("difficulty_weight", _opponent_weight(opponent)))
        weighted_total += float(item.get("winrate", 0.0)) * weight
        total_weight += weight
    return (weighted_total / total_weight) if total_weight > 0 else 0.0


def _score_rate(item: dict[str, object]) -> float:
    games = int(item.get("games", 0))
    if games <= 0:
        return 0.0
    wins = int(item.get("wins_a", 0))
    draws = int(item.get("draws", 0))
    return (wins + 0.5 * draws) / games


def _estimate_benchmark_elo(matchups: list[dict[str, object]]) -> float | None:
    estimates: list[float] = []
    weights: list[float] = []
    for item in matchups:
        games = int(item.get("games", 0))
        if games <= 0:
            continue
        score = min(max(_score_rate(item), 0.01), 0.99)
        opponent = str(item.get("opponent", ""))
        opponent_rating = float(item.get("opponent_rating", _opponent_rating(opponent)))
        # Inverse of expected score formula: E = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        rating_estimate = opponent_rating - 400.0 * math.log10((1.0 / score) - 1.0)
        estimates.append(rating_estimate)
        weights.append(float(item.get("difficulty_weight", _opponent_weight(opponent))) * games)
    if not estimates:
        return None
    total_weight = sum(weights)
    if total_weight <= 0:
        return sum(estimates) / len(estimates)
    return sum(value * weight for value, weight in zip(estimates, weights)) / total_weight


def _append_benchmark_history(models_root: Path, model_id: str, payload: dict[str, object]) -> Path:
    history_dir = models_root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "benchmark_history.jsonl"
    record = {"model_id": model_id, **payload}
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    return history_path


def _update_model_card_after_benchmark(models_root: Path, model_id: str, summary_payload: dict[str, object], benchmark_score: float, report_path: Path) -> None:
    model_card_path = models_root / "final" / f"{model_id}.model_card.json"
    if not model_card_path.exists():
        return
    payload = json.loads(model_card_path.read_text(encoding="utf-8"))
    payload["benchmark_summary_path"] = str(report_path)
    payload["benchmark_score"] = benchmark_score
    payload["benchmark_score_weighted"] = float(summary_payload.get("benchmark_score_weighted", benchmark_score))
    payload["benchmark_elo_estimate"] = summary_payload.get("benchmark_elo_estimate")
    if summary_payload.get("benchmark_history_path"):
        payload["benchmark_history_path"] = summary_payload.get("benchmark_history_path")
    payload["benchmark_matchups"] = len(summary_payload.get("matchups", []))
    model_card_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _build_target_agent(job: JobContext) -> AgentLike:
    benchmark_cfg = job.config.get("benchmark", {})
    target = str(benchmark_cfg.get("target", "engine_v1"))
    if target == "engine_v1":
        return EngineAgent()

    checkpoint_cfg = str(benchmark_cfg.get("checkpoint_path", "")).strip()
    if target in {"auto_best", "auto_promoted_best"} or checkpoint_cfg in {"auto_best", "auto_promoted_best"}:
        metadata = promoted_best_metadata(job.paths.models_root)
        if not metadata:
            raise FileNotFoundError("Aucun modele promu disponible pour le benchmark auto_best.")
        target = str(metadata.get("model_id", "promoted_best"))
        checkpoint_path = job.paths.models_root / "promoted" / "best" / "model.pt"
    elif target in {"auto", "auto_latest"} or checkpoint_cfg in {"", "auto", "auto_latest"}:
        latest = latest_model_record(job.paths.models_root)
        if not latest:
            raise FileNotFoundError("Aucun modele disponible dans le registre pour le benchmark auto_latest.")
        target = str(latest.get("model_id", ""))
        checkpoint_path = Path(str(latest.get("checkpoint_path", "")).strip())
    else:
        checkpoint_path = _resolve_storage_path(job.paths.drive_root, benchmark_cfg.get("checkpoint_path"), job.job_dir / "model.pt")
    device = str(job.config.get("runtime", {}).get("device", "cpu"))
    return ModelAgent(
        str(checkpoint_path),
        display_name=target,
        device=device,
        search_enabled=bool(benchmark_cfg.get("model_search_enabled", True)),
        search_top_k=max(1, int(benchmark_cfg.get("model_search_top_k", 4))),
        search_policy_weight=float(benchmark_cfg.get("model_search_policy_weight", 0.35)),
        search_value_weight=float(benchmark_cfg.get("model_search_value_weight", 1.0)),
    )


def run_benchmark_job(job: JobContext) -> dict[str, object]:
    benchmark_cfg = job.config.get("benchmark", {})
    games = int(benchmark_cfg.get("games_per_matchup", 20))
    max_moves = int(benchmark_cfg.get("max_moves", 300))
    matchups = list(benchmark_cfg.get("matchups", []))
    parallel_enabled = _as_bool(benchmark_cfg.get("parallel_enabled", True), default=True)
    parallel_backend = str(benchmark_cfg.get("parallel_backend", "thread")).strip().lower() or "thread"
    parallel_workers = max(1, int(benchmark_cfg.get("parallel_workers", 16) or 16))
    if parallel_backend not in {"thread", "sequential"}:
        job.logger.warning("benchmark parallel backend non supporte: %s | fallback=thread", parallel_backend)
        parallel_backend = "thread"
    if not parallel_enabled:
        parallel_backend = "sequential"
    configured_ratings = benchmark_cfg.get("opponent_ratings", {})
    if not isinstance(configured_ratings, dict):
        configured_ratings = {}

    job.logger.info(
        "benchmark parallel config | enabled=%s | backend=%s | workers=%s",
        parallel_enabled,
        parallel_backend,
        parallel_workers,
    )

    target_agent = _build_target_agent(job)
    benchmark_dir = job.job_dir / "benchmark"
    games_dir = benchmark_dir / "games"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    games_dir.mkdir(parents=True, exist_ok=True)

    state = job.read_state()
    completed_matchups = set(state.get("completed_matchups", []))
    summaries: list[dict[str, object]] = []
    for opponent_spec in matchups:
        matchup_key = _slugify_matchup(str(opponent_spec))
        job.set_phase(f"benchmark:{matchup_key}")
        if matchup_key in completed_matchups:
            summary_path = benchmark_dir / f"{matchup_key}_summary.json"
            if summary_path.exists():
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                job.logger.info("benchmark matchup skipped (already completed): %s", opponent_spec)
                continue

        opponent = _build_external_agent(str(opponent_spec))
        job.logger.info("benchmark matchup started: %s vs %s", target_agent.display_name, opponent.display_name)
        wins_a = 0
        wins_b = 0
        draws = 0
        total_moves = 0
        total_choose_fallbacks_target = 0
        total_choose_fallbacks_opponent = 0
        as_first = {"wins": 0, "losses": 0, "draws": 0}
        as_second = {"wins": 0, "losses": 0, "draws": 0}
        as_first_choose_fallbacks_target = 0
        as_second_choose_fallbacks_target = 0
        game_results: dict[int, dict[str, object]] = {}
        pending_game_indices: list[int] = []
        for game_index in range(games):
            game_filename = f"{matchup_key}_game_{game_index + 1:06d}.json"
            game_path = games_dir / game_filename
            if game_path.exists():
                try:
                    game_results[game_index] = _load_existing_game_result(game_path)
                    continue
                except Exception as exc:
                    job.logger.warning(
                        "benchmark game cache invalide | matchup=%s | game=%s/%s | cause=%s: %s | recompute=true",
                        matchup_key,
                        game_index + 1,
                        games,
                        type(exc).__name__,
                        exc,
                    )
            pending_game_indices.append(game_index)

        if game_results:
            latest_existing_game_index = max(game_results.keys()) + 1
            job.write_state(
                {
                    "completed_matchups": sorted(completed_matchups),
                    "current_matchup": matchup_key,
                    "completed_games": len(game_results),
                    "remaining_games": games - len(game_results),
                    "last_completed_game_id": f"{matchup_key}_game_{latest_existing_game_index:06d}",
                }
            )

        from songo_model_stockfish.benchmark.play_match import play_match

        def _play_single_game_payload(game_index: int) -> tuple[int, dict[str, object]]:
            starter = game_index % 2
            match_result = play_match(target_agent, opponent, max_moves=max_moves, starter=starter)
            payload = {
                "game_index": game_index + 1,
                "matchup": matchup_key,
                "opponent": str(opponent_spec),
                "engine": target_agent.display_name,
                **match_result.to_dict(),
            }
            return game_index, payload

        def _persist_new_game(game_index: int, payload: dict[str, object]) -> None:
            game_filename = f"{matchup_key}_game_{game_index + 1:06d}.json"
            game_path = games_dir / game_filename
            _write_json(game_path, payload)
            job.write_event("benchmark_game_completed", matchup=matchup_key, game_index=game_index + 1, winner=payload["winner"])
            job.write_metric(
                {
                    "metric_type": "game_result",
                    "matchup_id": matchup_key,
                    "game_id": game_filename.removesuffix(".json"),
                    "winner": payload["winner"],
                    "moves": payload["moves"],
                    "choose_fallbacks_target": int((payload.get("choose_fallbacks") or [0, 0])[0] or 0),
                    "choose_fallbacks_opponent": int((payload.get("choose_fallbacks") or [0, 0])[1] or 0),
                    "avg_move_time_ms": (
                        (sum(payload["think_ms"]) / len(payload["think_ms"]))
                        if payload["think_ms"]
                        else 0.0
                    ),
                }
            )
            completed_games = len(game_results)
            job.write_state(
                {
                    "completed_matchups": sorted(completed_matchups),
                    "current_matchup": matchup_key,
                    "completed_games": completed_games,
                    "remaining_games": games - completed_games,
                    "last_completed_game_id": game_filename.removesuffix(".json"),
                }
            )

        if pending_game_indices:
            can_run_parallel = (
                parallel_backend == "thread"
                and parallel_workers > 1
                and len(pending_game_indices) > 1
            )
            if can_run_parallel:
                worker_count = min(parallel_workers, len(pending_game_indices))
                job.logger.info(
                    "benchmark matchup parallel run | matchup=%s | workers=%s | pending=%s",
                    matchup_key,
                    worker_count,
                    len(pending_game_indices),
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_to_game_index = {
                        executor.submit(_play_single_game_payload, game_index): game_index
                        for game_index in pending_game_indices
                    }
                    for future in concurrent.futures.as_completed(future_to_game_index):
                        game_index = future_to_game_index[future]
                        try:
                            resolved_game_index, game_payload = future.result()
                        except Exception as exc:
                            job.logger.exception(
                                "benchmark worker echec | matchup=%s | game=%s/%s | cause=%s: %s | strict_mode_abort=true",
                                matchup_key,
                                game_index + 1,
                                games,
                                type(exc).__name__,
                                exc,
                            )
                            raise RuntimeError(
                                f"Benchmark game failed in strict mode | matchup={matchup_key} | "
                                f"game={game_index + 1}/{games} | cause={type(exc).__name__}: {exc}"
                            ) from exc
                        game_results[resolved_game_index] = game_payload
                        _persist_new_game(resolved_game_index, game_payload)
            else:
                for game_index in pending_game_indices:
                    resolved_game_index, game_payload = _play_single_game_payload(game_index)
                    game_results[resolved_game_index] = game_payload
                    _persist_new_game(resolved_game_index, game_payload)

        if len(game_results) != games:
            raise RuntimeError(
                f"Benchmark incomplet pour matchup={matchup_key}: "
                f"{len(game_results)}/{games} games disponibles."
            )

        for game_index in range(games):
            starter = game_index % 2
            game_payload = game_results[game_index]
            winner = game_payload["winner"]
            choose_fallbacks = game_payload.get("choose_fallbacks", [0, 0])
            if not isinstance(choose_fallbacks, (list, tuple)) or len(choose_fallbacks) < 2:
                choose_fallbacks = [0, 0]
            target_fallbacks = int(choose_fallbacks[0] or 0)
            opponent_fallbacks = int(choose_fallbacks[1] or 0)
            if winner == 0:
                wins_a += 1
            elif winner == 1:
                wins_b += 1
            else:
                draws += 1
            total_choose_fallbacks_target += target_fallbacks
            total_choose_fallbacks_opponent += opponent_fallbacks
            if int(game_payload.get("starter", starter)) == 0:
                _record_by_role(as_first, winner)
                as_first_choose_fallbacks_target += target_fallbacks
            else:
                _record_by_role(as_second, winner)
                as_second_choose_fallbacks_target += target_fallbacks
            total_moves += int(game_payload["moves"])
            winner_label = _winner_label(winner, target_agent.display_name, opponent.display_name)
            job.logger.info(
                "benchmark game completed | matchup=%s | game=%s/%s | winner=%s | score=%s-%s | moves=%s | reason=%s | choose_fallbacks_target=%s | choose_fallbacks_opponent=%s",
                matchup_key,
                game_index + 1,
                games,
                winner_label,
                game_payload["scores"][0],
                game_payload["scores"][1],
                game_payload["moves"],
                game_payload.get("reason", ""),
                target_fallbacks,
                opponent_fallbacks,
            )
        payload = {
            "opponent": str(opponent_spec),
            "engine": target_agent.display_name,
            "games": games,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "winrate": wins_a / games if games else 0.0,
            "score_rate": (wins_a + 0.5 * draws) / games if games else 0.0,
            "avg_moves": total_moves / games if games else 0.0,
            "choose_fallbacks_target_total": int(total_choose_fallbacks_target),
            "choose_fallbacks_opponent_total": int(total_choose_fallbacks_opponent),
            "difficulty_weight": _opponent_weight(str(opponent_spec)),
            "opponent_rating": _opponent_rating(str(opponent_spec), configured_ratings),
            "as_first_player": {
                **as_first,
                "choose_fallbacks_target": int(as_first_choose_fallbacks_target),
                "games": sum(as_first.values()),
                "winrate": (as_first["wins"] / sum(as_first.values())) if sum(as_first.values()) else 0.0,
                "score_rate": ((as_first["wins"] + 0.5 * as_first["draws"]) / sum(as_first.values())) if sum(as_first.values()) else 0.0,
            },
            "as_second_player": {
                **as_second,
                "choose_fallbacks_target": int(as_second_choose_fallbacks_target),
                "games": sum(as_second.values()),
                "winrate": (as_second["wins"] / sum(as_second.values())) if sum(as_second.values()) else 0.0,
                "score_rate": ((as_second["wins"] + 0.5 * as_second["draws"]) / sum(as_second.values())) if sum(as_second.values()) else 0.0,
            },
        }
        _write_json(benchmark_dir / f"{matchup_key}_summary.json", payload)
        summaries.append(payload)
        completed_matchups.add(matchup_key)
        cumulative_winrate = sum(float(item.get("winrate", 0.0)) for item in summaries) / len(summaries)
        cumulative_weighted = _compute_weighted_benchmark_score(summaries)
        cumulative_elo = _estimate_benchmark_elo(summaries)
        job.logger.info(
            "benchmark matchup completed | matchup=%s | target=%s | opponent=%s | wins=%s | losses=%s | draws=%s | winrate=%.4f | score_rate=%.4f | as_first=%.4f | as_second=%.4f | weighted=%.4f | elo_estimate=%.1f",
            matchup_key,
            target_agent.display_name,
            opponent.display_name,
            wins_a,
            wins_b,
            draws,
            payload["winrate"],
            payload["score_rate"],
            float(payload["as_first_player"]["winrate"]),
            float(payload["as_second_player"]["winrate"]),
            float(payload["winrate"]) * float(payload["difficulty_weight"]),
            cumulative_elo if cumulative_elo is not None else -1.0,
        )
        job.logger.info(
            "benchmark cumulative summary | completed_matchups=%s/%s | cumulative_winrate=%.4f | cumulative_weighted_score=%.4f | cumulative_elo_estimate=%.1f",
            len(summaries),
            len(matchups),
            cumulative_winrate,
            cumulative_weighted,
            cumulative_elo if cumulative_elo is not None else -1.0,
        )
        job.write_state(
            {
                "completed_matchups": sorted(completed_matchups),
                "current_matchup": matchup_key,
                "completed_games": games,
                "remaining_games": 0,
                "last_completed_game_id": f"{matchup_key}_game_{games:06d}",
            }
        )
        job.write_metric({"metric_type": "benchmark_matchup_completed", "matchup_id": matchup_key, **payload})
        job.write_event("benchmark_matchup_completed", matchup=matchup_key)

    benchmark_score = (
        sum(float(item.get("winrate", 0.0)) for item in summaries) / len(summaries)
        if summaries
        else 0.0
    )
    benchmark_score_weighted = _compute_weighted_benchmark_score(summaries)
    benchmark_elo_estimate = _estimate_benchmark_elo(summaries)
    summary_payload = {
        "job_id": job.job_id,
        "engine": target_agent.display_name,
        "matchups": summaries,
        "benchmark_score": benchmark_score,
        "benchmark_score_weighted": benchmark_score_weighted,
        "benchmark_elo_estimate": benchmark_elo_estimate,
        "opponent_ratings": configured_ratings,
    }
    report_path = benchmark_dir / "benchmark_summary.json"
    _write_json(report_path, summary_payload)
    if target_agent.display_name != "engine_v1":
        history_path = _append_benchmark_history(
            job.paths.models_root,
            target_agent.display_name,
            {
                "job_id": job.job_id,
                "created_at": time.time(),
                "benchmark_summary_path": str(report_path),
                "benchmark_score": benchmark_score,
                "benchmark_score_weighted": benchmark_score_weighted,
                "benchmark_elo_estimate": benchmark_elo_estimate,
                "matchups": [
                    {
                        "opponent": item.get("opponent"),
                        "games": item.get("games"),
                        "winrate": item.get("winrate"),
                        "score_rate": item.get("score_rate"),
                        "opponent_rating": item.get("opponent_rating"),
                    }
                    for item in summaries
                ],
            },
        )
        summary_payload["benchmark_history_path"] = str(history_path)
        _write_json(report_path, summary_payload)
        _update_model_card_after_benchmark(job.paths.models_root, target_agent.display_name, summary_payload, benchmark_score, report_path)
        registry = load_registry(job.paths.models_root)
        existing = next((item for item in registry.get("models", []) if str(item.get("model_id")) == target_agent.display_name), {})
        upsert_model_record(
            job.paths.models_root,
            {
                "model_id": target_agent.display_name,
                "sort_ts": time.time(),
                "dataset_id": existing.get("dataset_id", ""),
                "training_job_id": existing.get("training_job_id", ""),
                "checkpoint_path": existing.get("checkpoint_path", ""),
                "best_validation_metric": existing.get("best_validation_metric", -1.0),
                "evaluation_top1": float(existing.get("evaluation_top1", -1.0)),
                "benchmark_score": benchmark_score_weighted,
                "benchmark_elo_estimate": benchmark_elo_estimate if benchmark_elo_estimate is not None else -1.0,
                "benchmark_history_path": str(history_path),
                "benchmark_summary_path": str(report_path),
                "model_card_path": str(job.paths.models_root / "final" / f"{target_agent.display_name}.model_card.json"),
            },
        )
        promote_best_model(job.paths.models_root)
    return summary_payload
