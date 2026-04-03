from __future__ import annotations

import math
import time

from songo_model_stockfish.adapters import songo_ai_game
from songo_model_stockfish.engine.config import EngineConfig
from songo_model_stockfish.engine.order import order_moves
from songo_model_stockfish.engine.types import SearchInfo
from songo_model_stockfish.evaluation import evaluate_state

MATE_SCORE = 1_000_000.0


def _terminal_score(state, ply: int) -> float:
    winner = songo_ai_game.winner(state)
    if winner is None:
        return 0.0
    player = songo_ai_game.current_player(state)
    return (MATE_SCORE - ply) if winner == player else (-MATE_SCORE + ply)


def _negamax(state, depth: int, alpha: float, beta: float, ply: int, deadline: float | None, counters: dict[str, int]) -> tuple[float, list[int]]:
    counters["nodes"] += 1
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("search timeout")

    if songo_ai_game.is_terminal(state):
        return _terminal_score(state, ply), []

    moves = songo_ai_game.legal_moves(state)
    if depth == 0 or not moves:
        return evaluate_state(state), []

    best_score = -math.inf
    best_pv: list[int] = []

    if moves and counters.get("use_move_ordering", 1):
        moves = order_moves(state, moves)

    for move in moves:
        child = songo_ai_game.simulate_move(state, move)
        score, pv = _negamax(child, depth - 1, -beta, -alpha, ply + 1, deadline, counters)
        score = -score
        if score > best_score:
            best_score = score
            best_pv = [move] + pv
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best_score, best_pv


def choose_move(state, config: EngineConfig) -> tuple[int, SearchInfo]:
    t0 = time.perf_counter()
    deadline = None if config.time_ms is None else t0 + (config.time_ms / 1000.0)
    legal = songo_ai_game.legal_moves(state)
    if not legal:
        raise ValueError("No legal move available")

    best_move = legal[0]
    best_info = SearchInfo(best_score=-math.inf, depth_reached=0, nodes_searched=0, elapsed_ms=0.0, pv=[best_move])

    depths = range(1, config.max_depth + 1) if config.use_iterative_deepening else [config.max_depth]
    for depth in depths:
        counters = {"nodes": 0, "use_move_ordering": 1 if config.use_move_ordering else 0}
        try:
            score, pv = _negamax(state, depth, -math.inf, math.inf, 0, deadline, counters)
        except TimeoutError:
            break
        if pv:
            best_move = pv[0]
        best_info = SearchInfo(
            best_score=score,
            depth_reached=depth,
            nodes_searched=counters["nodes"],
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            pv=pv or [best_move],
        )

    return best_move, best_info
