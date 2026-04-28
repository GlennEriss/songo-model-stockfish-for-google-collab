from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_benchmark_defaults(*, worktree: Path, identity: str) -> dict[str, Any]:
    active_cfg = worktree / "config" / "generated" / f"benchmark.{identity}.active.yaml"
    fallback_cfg = worktree / "config" / "benchmark.colab_pro.yaml"

    if active_cfg.exists():
        payload = _load_yaml(active_cfg)
    elif fallback_cfg.exists():
        payload = _load_yaml(fallback_cfg)
    else:
        payload = {}

    runtime_cfg = payload.get("runtime", {}) if isinstance(payload, dict) else {}
    benchmark_cfg = payload.get("benchmark", {}) if isinstance(payload, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}
    if not isinstance(benchmark_cfg, dict):
        benchmark_cfg = {}

    return {
        "runtime": runtime_cfg,
        "benchmark": benchmark_cfg,
        "config_path": str(active_cfg if active_cfg.exists() else fallback_cfg),
    }


def _resolve_bool_choice(choice: str, *, fallback: bool) -> bool:
    normalized = str(choice or "auto").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return bool(fallback)


def _resolve_search_settings(
    *,
    defaults: dict[str, Any],
    device_override: str,
    search_enabled_choice: str,
    search_alpha_beta_choice: str,
    search_depth: int,
    search_top_k: int,
    search_top_k_child: int,
    search_policy_weight: float | None,
    search_value_weight: float | None,
) -> dict[str, Any]:
    runtime_cfg = defaults.get("runtime", {})
    benchmark_cfg = defaults.get("benchmark", {})

    runtime_device = str(runtime_cfg.get("device", "cpu")).strip() or "cpu"
    device = str(device_override).strip() or runtime_device

    enabled = _resolve_bool_choice(
        search_enabled_choice,
        fallback=bool(benchmark_cfg.get("model_search_enabled", True)),
    )
    alpha_beta = _resolve_bool_choice(
        search_alpha_beta_choice,
        fallback=bool(benchmark_cfg.get("model_search_alpha_beta", True)),
    )

    resolved = {
        "device": device,
        "search_enabled": bool(enabled),
        "search_depth": int(search_depth if int(search_depth) > 0 else int(benchmark_cfg.get("model_search_depth", 3))),
        "search_top_k": int(search_top_k if int(search_top_k) > 0 else int(benchmark_cfg.get("model_search_top_k", 6))),
        "search_top_k_child": int(
            search_top_k_child
            if int(search_top_k_child) > 0
            else int(benchmark_cfg.get("model_search_top_k_child", 4))
        ),
        "search_policy_weight": float(
            search_policy_weight
            if search_policy_weight is not None
            else float(benchmark_cfg.get("model_search_policy_weight", 0.35))
        ),
        "search_value_weight": float(
            search_value_weight
            if search_value_weight is not None
            else float(benchmark_cfg.get("model_search_value_weight", 1.0))
        ),
        "search_alpha_beta": bool(alpha_beta),
    }
    return resolved


def _resolve_checkpoint_path(checkpoint_raw: str, *, models_root: Path) -> Path:
    text = str(checkpoint_raw or "").strip()
    if not text:
        return Path()
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return models_root.parent / candidate


def _collect_models(*, models_root: Path, max_models: int) -> list[dict[str, Any]]:
    from songo_model_stockfish.ops.model_registry import load_registry

    registry = load_registry(models_root)
    models = list(registry.get("models", [])) if isinstance(registry, dict) else []

    normalized: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id", "")).strip()
        checkpoint_raw = str(item.get("checkpoint_path", "")).strip()
        if not model_id or not checkpoint_raw:
            continue
        checkpoint_path = _resolve_checkpoint_path(checkpoint_raw, models_root=models_root)
        if not checkpoint_path.exists():
            continue
        normalized.append(
            {
                "model_id": model_id,
                "checkpoint_path": str(checkpoint_path),
                "sort_ts": float(item.get("sort_ts", 0.0)),
                "benchmark_score": float(item.get("benchmark_score", -1.0)),
                "evaluation_top1": float(item.get("evaluation_top1", -1.0)),
            }
        )

    normalized = sorted(
        normalized,
        key=lambda row: (float(row.get("sort_ts", 0.0)), str(row.get("model_id", ""))),
    )

    if max_models > 0:
        normalized = normalized[-int(max_models) :]
    return normalized


def _empty_model_stats(model_id: str) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "points": 0,
        "moves_played": 0,
        "seeds_captured": 0,
        "think_ms_min": None,
        "think_ms_max": None,
        "think_ms_total": 0.0,
        "think_turns": 0,
        "best_move": {
            "captured": 0,
            "pit": None,
            "ply": None,
            "opponent": "",
            "pair": "",
            "game": "",
        },
    }


def _summarize_times(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "turns": 0,
            "min": None,
            "max": None,
            "avg": None,
            "total": 0.0,
        }
    total = float(sum(values))
    return {
        "turns": int(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": float(total / len(values)),
        "total": total,
    }


def _format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def _play_model_game(
    *,
    model_a_id: str,
    model_b_id: str,
    agent_a: Any,
    agent_b: Any,
    starter: int,
    max_moves: int,
) -> dict[str, Any]:
    from songo_model_stockfish.adapters import songo_ai_game

    state = songo_ai_game.create_state()

    if starter == 0:
        model_by_player = {0: model_a_id, 1: model_b_id}
        agent_by_player = {0: agent_a, 1: agent_b}
    else:
        model_by_player = {0: model_b_id, 1: model_a_id}
        agent_by_player = {0: agent_b, 1: agent_a}

    move_count = 0
    reason = "finished"
    captured_by_model: dict[str, int] = {model_a_id: 0, model_b_id: 0}
    turns_by_model: dict[str, int] = {model_a_id: 0, model_b_id: 0}
    think_times_by_model: dict[str, list[float]] = {model_a_id: [], model_b_id: []}
    best_move_by_model: dict[str, dict[str, Any]] = {
        model_a_id: {"captured": 0, "pit": None, "ply": None},
        model_b_id: {"captured": 0, "pit": None, "ply": None},
    }

    while (not songo_ai_game.is_terminal(state)) and move_count < max(1, int(max_moves)):
        legal = songo_ai_game.legal_moves(state)
        if not legal:
            reason = "no_legal_moves"
            break

        player = int(songo_ai_game.current_player(state))
        model_id = str(model_by_player[player])
        agent = agent_by_player[player]
        score_before = songo_ai_game.scores(state)

        t0 = time.perf_counter()
        move, _info = agent.choose(songo_ai_game.clone_state(state))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if move not in legal:
            raise ValueError(
                "ModelAgent returned illegal move | "
                f"model={model_id} | move={move} | legal={legal}"
            )

        next_state = songo_ai_game.simulate_move(state, int(move))
        score_after = songo_ai_game.scores(next_state)
        captured = max(0, int(score_after[player]) - int(score_before[player]))

        move_count += 1
        turns_by_model[model_id] = int(turns_by_model.get(model_id, 0)) + 1
        think_times = think_times_by_model.setdefault(model_id, [])
        think_times.append(float(elapsed_ms))
        captured_by_model[model_id] = int(captured_by_model.get(model_id, 0)) + int(captured)

        current_best = best_move_by_model.setdefault(model_id, {"captured": 0, "pit": None, "ply": None})
        if int(captured) > int(current_best.get("captured", 0)):
            current_best["captured"] = int(captured)
            current_best["pit"] = int(move)
            current_best["ply"] = int(move_count)

        state = next_state

    score_south, score_north = songo_ai_game.scores(state)
    winner_player = songo_ai_game.winner(state)
    if winner_player is None:
        if score_south > score_north:
            winner_player = 0
        elif score_north > score_south:
            winner_player = 1

    winner_model_id = str(model_by_player[int(winner_player)]) if winner_player is not None else None

    points = {model_a_id: 0, model_b_id: 0}
    if winner_model_id == model_a_id:
        points[model_a_id] = 3
        points[model_b_id] = 0
    elif winner_model_id == model_b_id:
        points[model_a_id] = 0
        points[model_b_id] = 3
    else:
        points[model_a_id] = 1
        points[model_b_id] = 1

    final_scores = {
        str(model_by_player[0]): int(score_south),
        str(model_by_player[1]): int(score_north),
    }

    return {
        "starter": int(starter),
        "starter_model_id": (model_a_id if starter == 0 else model_b_id),
        "winner_model_id": winner_model_id,
        "winner_label": (winner_model_id if winner_model_id else "draw"),
        "reason": ("finished" if songo_ai_game.is_terminal(state) else reason),
        "moves": int(move_count),
        "points": points,
        "final_scores": final_scores,
        "captured_by_model": captured_by_model,
        "turns_by_model": turns_by_model,
        "think_ms_by_model": {
            model_a_id: _summarize_times(think_times_by_model.get(model_a_id, [])),
            model_b_id: _summarize_times(think_times_by_model.get(model_b_id, [])),
        },
        "best_move_by_model": best_move_by_model,
    }


def _update_model_stats(*, stats: dict[str, Any], model_id: str, opponent_id: str, game: dict[str, Any], pair_label: str, game_label: str) -> None:
    stats["games"] = int(stats.get("games", 0)) + 1
    points = int(game.get("points", {}).get(model_id, 0))
    stats["points"] = int(stats.get("points", 0)) + points

    winner_model_id = game.get("winner_model_id")
    if winner_model_id == model_id:
        stats["wins"] = int(stats.get("wins", 0)) + 1
    elif winner_model_id is None:
        stats["draws"] = int(stats.get("draws", 0)) + 1
    else:
        stats["losses"] = int(stats.get("losses", 0)) + 1

    turns_by_model = game.get("turns_by_model", {})
    captured_by_model = game.get("captured_by_model", {})
    think_by_model = game.get("think_ms_by_model", {})

    turns = int(turns_by_model.get(model_id, 0) or 0)
    stats["moves_played"] = int(stats.get("moves_played", 0)) + turns
    stats["seeds_captured"] = int(stats.get("seeds_captured", 0)) + int(captured_by_model.get(model_id, 0) or 0)

    think_summary = think_by_model.get(model_id, {}) if isinstance(think_by_model, dict) else {}
    turns_count = int(think_summary.get("turns", 0) or 0)
    think_total = float(think_summary.get("total", 0.0) or 0.0)
    think_min = think_summary.get("min")
    think_max = think_summary.get("max")

    if turns_count > 0:
        stats["think_turns"] = int(stats.get("think_turns", 0)) + turns_count
        stats["think_ms_total"] = float(stats.get("think_ms_total", 0.0)) + think_total

        current_min = stats.get("think_ms_min")
        if current_min is None or (think_min is not None and float(think_min) < float(current_min)):
            stats["think_ms_min"] = float(think_min)

        current_max = stats.get("think_ms_max")
        if current_max is None or (think_max is not None and float(think_max) > float(current_max)):
            stats["think_ms_max"] = float(think_max)

    best_move = game.get("best_move_by_model", {}).get(model_id, {})
    best_captured = int(best_move.get("captured", 0) or 0)
    current_best = stats.get("best_move", {})
    if best_captured > int(current_best.get("captured", 0) or 0):
        current_best["captured"] = best_captured
        current_best["pit"] = best_move.get("pit")
        current_best["ply"] = best_move.get("ply")
        current_best["opponent"] = opponent_id
        current_best["pair"] = pair_label
        current_best["game"] = game_label
        stats["best_move"] = current_best


def _build_ranking(model_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, stats in model_stats.items():
        games = int(stats.get("games", 0))
        points = int(stats.get("points", 0))
        max_points = max(1, games * 3)
        score_rate = float(points / max_points)
        think_turns = int(stats.get("think_turns", 0) or 0)
        think_avg = (float(stats.get("think_ms_total", 0.0)) / think_turns) if think_turns > 0 else None
        row = {
            "model_id": model_id,
            "games": games,
            "wins": int(stats.get("wins", 0)),
            "draws": int(stats.get("draws", 0)),
            "losses": int(stats.get("losses", 0)),
            "points": points,
            "score_rate": score_rate,
            "moves_played": int(stats.get("moves_played", 0)),
            "seeds_captured": int(stats.get("seeds_captured", 0)),
            "think_ms_min": stats.get("think_ms_min"),
            "think_ms_avg": think_avg,
            "think_ms_max": stats.get("think_ms_max"),
            "best_move": stats.get("best_move", {}),
        }
        rows.append(row)

    rows = sorted(
        rows,
        key=lambda item: (
            int(item.get("points", 0)),
            float(item.get("score_rate", 0.0)),
            int(item.get("wins", 0)),
            int(item.get("seeds_captured", 0)),
            str(item.get("model_id", "")),
        ),
        reverse=True,
    )

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _print_ranking_table(ranking: list[dict[str, Any]]) -> None:
    print("\nClassement live (3 pts victoire, 1 nul):", flush=True)
    header = (
        "Rk  Model                                    Pts  W  D  L  G   Score%  Seeds  "
        "Think(min/avg/max ms)   Best move"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for row in ranking:
        model_id = str(row.get("model_id", ""))
        think = f"{_format_ms(row.get('think_ms_min'))}/{_format_ms(row.get('think_ms_avg'))}/{_format_ms(row.get('think_ms_max'))}"
        best_move = row.get("best_move", {}) if isinstance(row.get("best_move"), dict) else {}
        best_captured = int(best_move.get("captured", 0) or 0)
        if best_captured > 0:
            best_label = f"case {best_move.get('pit')} (+{best_captured})"
        else:
            best_label = "-"
        print(
            f"{int(row.get('rank', 0)):>2}  "
            f"{model_id[:40]:<40} "
            f"{int(row.get('points', 0)):>3}  "
            f"{int(row.get('wins', 0)):>2} "
            f"{int(row.get('draws', 0)):>2} "
            f"{int(row.get('losses', 0)):>2} "
            f"{int(row.get('games', 0)):>2}  "
            f"{100.0 * float(row.get('score_rate', 0.0)):>6.2f}%  "
            f"{int(row.get('seeds_captured', 0)):>5}  "
            f"{think:<22} "
            f"{best_label}",
            flush=True,
        )


def run_model_tournament(
    *,
    worktree: Path,
    drive_root: Path,
    identity: str,
    games_per_pair: int,
    max_moves: int,
    max_models: int,
    device: str,
    search_enabled_choice: str,
    search_alpha_beta_choice: str,
    search_depth: int,
    search_top_k: int,
    search_top_k_child: int,
    search_policy_weight: float | None,
    search_value_weight: float | None,
) -> dict[str, Any]:
    from songo_model_stockfish.benchmark.model_agent import ModelAgent

    if int(games_per_pair) <= 0:
        raise ValueError("games_per_pair doit etre > 0")

    defaults = _resolve_benchmark_defaults(worktree=worktree, identity=identity)
    search_cfg = _resolve_search_settings(
        defaults=defaults,
        device_override=device,
        search_enabled_choice=search_enabled_choice,
        search_alpha_beta_choice=search_alpha_beta_choice,
        search_depth=search_depth,
        search_top_k=search_top_k,
        search_top_k_child=search_top_k_child,
        search_policy_weight=search_policy_weight,
        search_value_weight=search_value_weight,
    )

    models_root = drive_root / "models"
    all_models = _collect_models(models_root=models_root, max_models=int(max_models))
    if len(all_models) < 2:
        raise RuntimeError(
            "Tournoi impossible: au moins 2 modeles valides avec checkpoint sont requis dans model_registry."
        )

    model_ids = [str(item["model_id"]) for item in all_models]
    print("Tournoi modeles detectes =", model_ids, flush=True)
    print("Config benchmark source =", defaults.get("config_path", ""), flush=True)
    print("Search settings =", json.dumps(search_cfg, ensure_ascii=True), flush=True)

    pairings: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(len(all_models)):
        for j in range(i + 1, len(all_models)):
            pairings.append((all_models[i], all_models[j]))

    total_games = len(pairings) * int(games_per_pair)
    print(
        f"Tournoi: {len(all_models)} modeles | {len(pairings)} paires | {games_per_pair} matchs/pair | total={total_games}",
        flush=True,
    )

    model_stats: dict[str, dict[str, Any]] = {
        str(item["model_id"]): _empty_model_stats(str(item["model_id"])) for item in all_models
    }
    pairs_summary: list[dict[str, Any]] = []
    games_payload: list[dict[str, Any]] = []

    started = time.time()
    played_games = 0

    for pair_index, (model_a, model_b) in enumerate(pairings, start=1):
        model_a_id = str(model_a["model_id"])
        model_b_id = str(model_b["model_id"])
        pair_label = f"{model_a_id} vs {model_b_id}"

        print(
            f"\n=== Pair {pair_index}/{len(pairings)}: {pair_label} | games={games_per_pair} ===",
            flush=True,
        )

        agent_a = ModelAgent(
            str(model_a["checkpoint_path"]),
            display_name=model_a_id,
            device=str(search_cfg["device"]),
            search_enabled=bool(search_cfg["search_enabled"]),
            search_top_k=int(search_cfg["search_top_k"]),
            search_top_k_child=int(search_cfg["search_top_k_child"]),
            search_depth=int(search_cfg["search_depth"]),
            search_policy_weight=float(search_cfg["search_policy_weight"]),
            search_value_weight=float(search_cfg["search_value_weight"]),
            search_profile="fort_plusplus",
            search_alpha_beta=bool(search_cfg["search_alpha_beta"]),
        )
        agent_b = ModelAgent(
            str(model_b["checkpoint_path"]),
            display_name=model_b_id,
            device=str(search_cfg["device"]),
            search_enabled=bool(search_cfg["search_enabled"]),
            search_top_k=int(search_cfg["search_top_k"]),
            search_top_k_child=int(search_cfg["search_top_k_child"]),
            search_depth=int(search_cfg["search_depth"]),
            search_policy_weight=float(search_cfg["search_policy_weight"]),
            search_value_weight=float(search_cfg["search_value_weight"]),
            search_profile="fort_plusplus",
            search_alpha_beta=bool(search_cfg["search_alpha_beta"]),
        )

        wins_a = 0
        wins_b = 0
        draws = 0
        points_a = 0
        points_b = 0
        total_moves = 0

        for game_in_pair in range(1, int(games_per_pair) + 1):
            starter = (game_in_pair - 1) % 2
            game_payload = _play_model_game(
                model_a_id=model_a_id,
                model_b_id=model_b_id,
                agent_a=agent_a,
                agent_b=agent_b,
                starter=starter,
                max_moves=int(max_moves),
            )

            played_games += 1
            game_label = f"pair{pair_index:03d}_game{game_in_pair:03d}"
            game_payload["pair_index"] = pair_index
            game_payload["pair_label"] = pair_label
            game_payload["game_in_pair"] = game_in_pair
            game_payload["game_id"] = game_label
            game_payload["model_a"] = model_a_id
            game_payload["model_b"] = model_b_id
            games_payload.append(game_payload)

            total_moves += int(game_payload.get("moves", 0))
            points_a += int(game_payload.get("points", {}).get(model_a_id, 0) or 0)
            points_b += int(game_payload.get("points", {}).get(model_b_id, 0) or 0)

            winner_model_id = game_payload.get("winner_model_id")
            if winner_model_id == model_a_id:
                wins_a += 1
            elif winner_model_id == model_b_id:
                wins_b += 1
            else:
                draws += 1

            _update_model_stats(
                stats=model_stats[model_a_id],
                model_id=model_a_id,
                opponent_id=model_b_id,
                game=game_payload,
                pair_label=pair_label,
                game_label=game_label,
            )
            _update_model_stats(
                stats=model_stats[model_b_id],
                model_id=model_b_id,
                opponent_id=model_a_id,
                game=game_payload,
                pair_label=pair_label,
                game_label=game_label,
            )

            think_a = game_payload.get("think_ms_by_model", {}).get(model_a_id, {})
            think_b = game_payload.get("think_ms_by_model", {}).get(model_b_id, {})
            best_a = game_payload.get("best_move_by_model", {}).get(model_a_id, {})
            best_b = game_payload.get("best_move_by_model", {}).get(model_b_id, {})
            caps = game_payload.get("captured_by_model", {})
            scores = game_payload.get("final_scores", {})

            print(
                " | ".join(
                    [
                        f"game {played_games}/{total_games}",
                        f"pair={pair_label}",
                        f"winner={game_payload.get('winner_label', 'draw')}",
                        f"moves={game_payload.get('moves', 0)}",
                        f"scores={model_a_id}:{scores.get(model_a_id, 0)} {model_b_id}:{scores.get(model_b_id, 0)}",
                        f"captured={model_a_id}:{caps.get(model_a_id, 0)} {model_b_id}:{caps.get(model_b_id, 0)}",
                        f"think_ms[{model_a_id}]={_format_ms(think_a.get('min'))}/{_format_ms(think_a.get('avg'))}/{_format_ms(think_a.get('max'))}",
                        f"think_ms[{model_b_id}]={_format_ms(think_b.get('min'))}/{_format_ms(think_b.get('avg'))}/{_format_ms(think_b.get('max'))}",
                        f"best[{model_a_id}]=case {best_a.get('pit')} (+{best_a.get('captured', 0)})",
                        f"best[{model_b_id}]=case {best_b.get('pit')} (+{best_b.get('captured', 0)})",
                    ]
                ),
                flush=True,
            )

            ranking_live = _build_ranking(model_stats)
            _print_ranking_table(ranking_live)

        score_rate_a = (points_a / float(int(games_per_pair) * 3)) if int(games_per_pair) > 0 else 0.0
        score_rate_b = (points_b / float(int(games_per_pair) * 3)) if int(games_per_pair) > 0 else 0.0
        pairs_summary.append(
            {
                "model_a": model_a_id,
                "model_b": model_b_id,
                "games": int(games_per_pair),
                "wins_a": int(wins_a),
                "wins_b": int(wins_b),
                "draws": int(draws),
                "points_a": int(points_a),
                "points_b": int(points_b),
                "score_rate_a": float(score_rate_a),
                "score_rate_b": float(score_rate_b),
                "avg_moves": (float(total_moves) / float(int(games_per_pair))) if int(games_per_pair) > 0 else 0.0,
            }
        )

        print(
            f"Pair termine: {pair_label} | W-D-L({model_a_id})={wins_a}-{draws}-{wins_b} | "
            f"points={points_a}-{points_b}",
            flush=True,
        )

    ranking = _build_ranking(model_stats)
    ended = time.time()
    winner_model_id = str(ranking[0]["model_id"]) if ranking else ""

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended)),
        "identity": identity,
        "games_per_pair": int(games_per_pair),
        "max_moves": int(max_moves),
        "models": all_models,
        "search_config": search_cfg,
        "pairs": pairs_summary,
        "ranking": ranking,
        "auto_actions": {
            "winner_model_id": winner_model_id,
            "scoring": "win=3, draw=1, loss=0",
        },
        "games": games_payload,
        "elapsed_seconds": float(ended - started),
        "total_games": int(total_games),
    }

    report_dir = drive_root / "reports" / "benchmarks" / "model_tournaments"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = int(ended)
    summary_path = report_dir / f"model_tournament_{identity}_{ts}.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    latest_path = report_dir / f"model_tournament_latest_{identity}.json"
    latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print("\n=== Tournoi termine ===", flush=True)
    _print_ranking_table(ranking)
    print("winner_model_id =", winner_model_id or "<none>", flush=True)
    print("summary_path    =", summary_path, flush=True)
    print("latest_path     =", latest_path, flush=True)

    return {
        "winner_model_id": winner_model_id,
        "summary_path": str(summary_path),
        "latest_path": str(latest_path),
        "total_games": int(total_games),
        "total_models": len(all_models),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", default="/content/songo-model-stockfish-for-google-collab")
    parser.add_argument(
        "--drive-root",
        default=(str(os.environ.get("SONGO_DRIVE_ROOT", "")).strip() or "/content/drive/MyDrive/songo-stockfish"),
    )
    parser.add_argument(
        "--identity",
        default=(str(os.environ.get("SONGO_DRIVE_IDENTITY_KEY", "")).strip() or "unknown_drive_identity"),
    )
    parser.add_argument("--games-per-pair", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=400)
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--search-enabled", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--search-alpha-beta", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--search-depth", type=int, default=0)
    parser.add_argument("--search-top-k", type=int, default=0)
    parser.add_argument("--search-top-k-child", type=int, default=0)
    parser.add_argument("--search-policy-weight", type=float, default=None)
    parser.add_argument("--search-value-weight", type=float, default=None)
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

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

    if str(args.summary_path or "").strip():
        Path(str(args.summary_path)).write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    if bool(args.print_json):
        print(json.dumps(summary, indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
