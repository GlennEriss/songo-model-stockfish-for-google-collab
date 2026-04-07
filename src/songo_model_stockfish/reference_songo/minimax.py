from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from songo_model_stockfish.reference_songo.engine import NUM_PITS, State, other_player, side_total, steps_to_reach_opponent
from songo_model_stockfish.reference_songo.game import legal_moves, simulate_move, terminal_winner_utility


@dataclass(frozen=True)
class SearchConfig:
    time_ms: int = 250
    max_depth: int = 12
    use_tt: bool = True
    order_moves: bool = True
    eval_mode: str = "standard"


TTKey = Tuple[int, Tuple[int, ...], int, int, int]
TTVal = Tuple[float, int]


def _hash_state(state: State, depth: int) -> TTKey:
    board = state["board"]
    flat = tuple(board[0] + board[1])
    scores = state["scores"]
    player = int(state["current_player"])
    return (player, flat, int(scores[0]), int(scores[1]), depth)


def _legal_move_count_for_player(state: State, player: int) -> int:
    probe: State = {
        "board": state["board"],
        "scores": state["scores"],
        "current_player": player,
        "finished": state.get("finished", False),
        "winner": state.get("winner"),
        "reason": state.get("reason", ""),
    }
    return len(legal_moves(probe))


def _capturable_and_yinda_counts(side: list[int]) -> tuple[int, int]:
    capturable = sum(1 for s in side if 2 <= int(s) <= 4)
    yinda = len(side) - capturable
    return capturable, yinda


def _transport_bidoua_terms(side: list[int], player: int) -> tuple[int, int]:
    safe_transport = 0
    forced_transmit = 0
    for pit_index in range(NUM_PITS):
        seeds = int(side[pit_index])
        if seeds <= 0:
            continue
        steps = max(1, int(steps_to_reach_opponent(player, pit_index)))
        local_capacity = max(0, steps - 1)
        safe_transport += min(seeds, local_capacity)
        forced_transmit += max(0, seeds - local_capacity)
    return safe_transport, forced_transmit


def _owona_bidoua_single(n: int, x: int) -> float:
    if n <= 0 or x <= 1:
        return 0.0
    n_use = min(n, x - 1)
    return -0.5 * (float(n_use * n_use) + float((1 - 2 * x) * n_use) - 2.0)


def _owona_bidoua_multi(n: int, x: int) -> float:
    if n <= 0 or x <= 1:
        return 0.0
    if n < x:
        return _owona_bidoua_single(n, x)

    total = 0.0
    remaining = int(n)
    xk = int(x)
    while remaining > 0 and xk > 1:
        nk = min(remaining, xk - 1)
        total += _owona_bidoua_single(nk, xk)
        remaining -= nk
        xk -= 1
    return total


def _owona_yinda_bidoua(n: int, x: int) -> float:
    if n <= 0:
        return 0.0
    b = _owona_bidoua_multi(n, x)
    k = 0.5 * float((n * n) - n + 2)
    return b - k


def _owona_transport_bidoua(n: int, x: int) -> float:
    if n <= 0 or x <= 1:
        return 0.0
    if n >= x:
        return 0.0
    b_n = _owona_bidoua_multi(n, x)
    if (n + 1) > x:
        return b_n
    b_np1 = _owona_bidoua_multi(n + 1, x)
    return b_np1 - b_n


def _owona_bidoua_side(side: list[int], player: int) -> tuple[float, float, float]:
    b_total = 0.0
    j_total = 0.0
    t_total = 0.0
    for pit_index in range(NUM_PITS):
        n = int(side[pit_index])
        if n <= 0:
            continue
        x = max(1, int(steps_to_reach_opponent(player, pit_index)) - 1)
        b_total += _owona_bidoua_multi(n, x)
        j_total += _owona_yinda_bidoua(n, x)
        t_total += _owona_transport_bidoua(n, x)
    return b_total, j_total, t_total


def evaluate(state: State, root_player: int, mode: str = "standard") -> float:
    scores = state["scores"]
    board = state["board"]
    p = root_player
    o = other_player(p)

    score_diff = int(scores[p]) - int(scores[o])
    seeds_diff = side_total(board, p) - side_total(board, o)
    base = 100.0 * score_diff + 1.5 * seeds_diff

    if mode == "standard":
        return base

    if mode == "bidoua_math":
        own_mob = _legal_move_count_for_player(state, p)
        opp_mob = _legal_move_count_for_player(state, o)
        mobility_diff = own_mob - opp_mob

        own_b, own_j, own_t = _owona_bidoua_side(board[p], p)
        opp_b, opp_j, opp_t = _owona_bidoua_side(board[o], o)
        b_diff = own_b - opp_b
        j_diff = own_j - opp_j
        t_diff = own_t - opp_t

        own_side = side_total(board, p)
        opp_side = side_total(board, o)
        starvation_pressure = max(0, 10 - opp_side) - max(0, 10 - own_side)

        return (
            140.0 * score_diff
            + 2.0 * seeds_diff
            + 8.0 * mobility_diff
            + 6.0 * b_diff
            + 4.0 * j_diff
            + 5.0 * t_diff
            + 3.0 * starvation_pressure
        )

    own_mob = _legal_move_count_for_player(state, p)
    opp_mob = _legal_move_count_for_player(state, o)
    mobility_diff = own_mob - opp_mob

    own_capturable, own_yinda = _capturable_and_yinda_counts(board[p])
    opp_capturable, opp_yinda = _capturable_and_yinda_counts(board[o])
    capturable_diff = opp_capturable - own_capturable
    yinda_diff = own_yinda - opp_yinda

    own_non_empty = sum(1 for s in board[p] if int(s) > 0)
    opp_non_empty = sum(1 for s in board[o] if int(s) > 0)
    non_empty_diff = own_non_empty - opp_non_empty

    own_safe_transport, own_forced = _transport_bidoua_terms(board[p], p)
    opp_safe_transport, opp_forced = _transport_bidoua_terms(board[o], o)
    transport_diff = (own_safe_transport - opp_safe_transport) - (own_forced - opp_forced)

    own_side = side_total(board, p)
    opp_side = side_total(board, o)
    starvation_pressure = max(0, 10 - opp_side) - max(0, 10 - own_side)

    return (
        130.0 * score_diff
        + 2.0 * seeds_diff
        + 12.0 * mobility_diff
        + 4.0 * capturable_diff
        + 5.0 * yinda_diff
        + 3.0 * transport_diff
        + 2.0 * non_empty_diff
        + 3.0 * starvation_pressure
    )


def _move_ordering(state: State, moves: list[int], root_player: int, deadline: float, eval_mode: str) -> list[int]:
    scored = []
    remaining: list[int] = []
    for m in moves:
        if time.perf_counter() >= deadline:
            remaining.append(m)
            continue
        s2 = simulate_move(state, m)
        if eval_mode != "standard":
            h = evaluate(s2, root_player, mode=eval_mode)
            scored.append((h, m))
        else:
            before = int(state["scores"][root_player])
            after = int(s2["scores"][root_player])
            gain = after - before
            scored.append((gain, m))
    scored.sort(reverse=True)
    scored_moves = [m for _, m in scored]
    return scored_moves + remaining


def choose_move(state: State, cfg: SearchConfig = SearchConfig()) -> Tuple[int, dict]:
    start = time.perf_counter()
    deadline = start + cfg.time_ms / 1000.0

    root_player = int(state["current_player"])
    moves = legal_moves(state)
    if not moves:
        raise RuntimeError("No legal moves available (should be terminal).")

    best_move = moves[0]
    best_score = float("-inf")
    completed_depth = 0
    tt: Dict[TTKey, TTVal] = {}
    best_root_scores: Dict[int, float] = {}

    for depth in range(1, cfg.max_depth + 1):
        if time.perf_counter() >= deadline:
            break

        search_moves = moves
        if cfg.order_moves:
            search_moves = _move_ordering(state, moves, root_player, deadline, cfg.eval_mode)

        alpha = float("-inf")
        beta = float("inf")
        local_best_move = best_move
        local_best_score = float("-inf")
        local_root_scores: Dict[int, float] = {}

        for m in search_moves:
            if time.perf_counter() >= deadline:
                break
            s2 = simulate_move(state, m)
            score = _alphabeta(s2, depth - 1, alpha, beta, root_player, deadline, tt if cfg.use_tt else None, cfg.eval_mode)
            local_root_scores[int(m)] = float(score)
            if score > local_best_score:
                local_best_score = score
                local_best_move = m
            alpha = max(alpha, local_best_score)

        if time.perf_counter() < deadline:
            best_move, best_score = local_best_move, local_best_score
            completed_depth = depth
            best_root_scores = dict(local_root_scores)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return best_move, {
        "time_ms": elapsed_ms,
        "depth_reached": completed_depth,
        "score": best_score,
        "root_scores": best_root_scores,
        "eval_mode": cfg.eval_mode,
    }


def _alphabeta(
    state: State,
    depth: int,
    alpha: float,
    beta: float,
    root_player: int,
    deadline: float,
    tt: Optional[Dict[TTKey, TTVal]],
    eval_mode: str,
) -> float:
    if time.perf_counter() >= deadline:
        return evaluate(state, root_player, mode=eval_mode)

    if bool(state["finished"]):
        return terminal_winner_utility(state, root_player)

    if depth <= 0:
        return evaluate(state, root_player, mode=eval_mode)

    if tt is not None:
        key = _hash_state(state, depth)
        if key in tt:
            val, d = tt[key]
            if d >= depth:
                return val

    player_to_move = int(state["current_player"])
    maximizing = player_to_move == root_player

    moves = legal_moves(state)
    if not moves:
        return evaluate(state, root_player, mode=eval_mode)

    if maximizing:
        value = float("-inf")
        for m in moves:
            if time.perf_counter() >= deadline:
                break
            s2 = simulate_move(state, m)
            value = max(value, _alphabeta(s2, depth - 1, alpha, beta, root_player, deadline, tt, eval_mode))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = float("inf")
        for m in moves:
            if time.perf_counter() >= deadline:
                break
            s2 = simulate_move(state, m)
            value = min(value, _alphabeta(s2, depth - 1, alpha, beta, root_player, deadline, tt, eval_mode))
            beta = min(beta, value)
            if alpha >= beta:
                break

    if tt is not None:
        tt[_hash_state(state, depth)] = (value, depth)

    return value
