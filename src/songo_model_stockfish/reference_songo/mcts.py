from __future__ import annotations

from collections import OrderedDict
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from songo_model_stockfish.reference_songo.engine import State, other_player, play_turn
from songo_model_stockfish.reference_songo.game import clone_state, legal_moves
from songo_model_stockfish.reference_songo.minimax import evaluate as _minimax_evaluate


@dataclass(frozen=True)
class MCTSConfig:
    time_ms: int = 80
    sims: int = 0
    c_uct: float = 1.25
    max_rollout_depth: int = 0
    eval_mode: str = "standard"
    use_tt: bool = True
    tt_max_entries: int = 250_000
    seed: Optional[int] = None


TTKey = Tuple[int, Tuple[int, ...], int, int, int, int]


@dataclass
class _TTEntry:
    n: int = 0
    w: float = 0.0
    children: Dict[int, TTKey] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


def _hash_state(state: State) -> TTKey:
    board = state["board"]
    scores = state["scores"]
    finished = 1 if bool(state.get("finished", False)) else 0
    winner = state.get("winner")
    w = -1 if winner is None else int(winner)
    flat = tuple(board[0] + board[1])
    return (finished, flat, int(scores[0]), int(scores[1]), int(state["current_player"]), w)


def _terminal_value(state: State, root_player: int) -> float:
    winner = state.get("winner")
    if winner is None:
        return 0.0
    return 1.0 if int(winner) == int(root_player) else -1.0


def _rollout_value(state: State, root_player: int, eval_mode: str = "standard") -> float:
    h = float(_minimax_evaluate(state, root_player, mode=eval_mode))
    return math.tanh(h / 200.0)


def choose_move(state: State, cfg: MCTSConfig = MCTSConfig()) -> Tuple[int, dict]:
    root_moves = legal_moves(state)
    if not root_moves:
        raise RuntimeError("No legal moves available (should be terminal).")

    start = time.perf_counter()
    deadline = start + max(0, int(cfg.time_ms)) / 1000.0

    import random

    rng = random.Random(cfg.seed if cfg.seed is not None else int(start * 1e9) & 0xFFFFFFFF)

    root_player = int(state["current_player"])
    root_state = clone_state(state)
    root_key = _hash_state(root_state)

    tt: OrderedDict[TTKey, _TTEntry] = OrderedDict()
    tt[root_key] = _TTEntry()

    sims_done = 0
    while True:
        if cfg.sims and sims_done >= cfg.sims:
            break
        if cfg.time_ms and time.perf_counter() >= deadline:
            break

        cur_state = clone_state(root_state)
        cur_key = root_key
        path: list[TTKey] = [cur_key]

        while True:
            if bool(cur_state.get("finished", False)):
                break
            entry = _tt_get(tt, cur_key)
            moves = legal_moves(cur_state)
            if not moves:
                break
            if any(m not in entry.children for m in moves):
                break
            move = _uct_select_move(tt, cur_state, cur_key, moves, cfg.c_uct, root_player)
            play_turn(cur_state, move)
            cur_key = _hash_state(cur_state)
            path.append(cur_key)

        if not bool(cur_state.get("finished", False)):
            entry = _tt_get(tt, cur_key)
            moves = legal_moves(cur_state)
            unexpanded = [m for m in moves if m not in entry.children]
            if unexpanded:
                move = unexpanded[rng.randrange(len(unexpanded))]
                play_turn(cur_state, move)
                child_key = _hash_state(cur_state)
                entry.children[move] = child_key
                _tt_touch(tt, cur_key, cfg)
                _tt_get(tt, child_key)
                cur_key = child_key
                path.append(cur_key)

        value = _simulate_from(cur_state, root_player, cfg.max_rollout_depth, cfg.eval_mode, rng)
        _backprop_tt(tt, path, value, cfg)
        sims_done += 1

    best_move = root_moves[0]
    best_n = -1
    root_entry = _tt_get(tt, root_key)
    for m in root_moves:
        ck = root_entry.children.get(m)
        if ck is None:
            continue
        cn = _tt_get(tt, ck).n
        if cn > best_n:
            best_n = cn
            best_move = m

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    info = {
        "sims": sims_done,
        "time_ms": elapsed_ms,
        "eval_mode": cfg.eval_mode,
        "root_visits": {m: _tt_get(tt, root_entry.children[m]).n for m in root_entry.children},
        "root_q": {m: _tt_get(tt, root_entry.children[m]).q for m in root_entry.children},
        "tt_size": len(tt),
    }
    return best_move, info


def _uct_select_move(
    tt: OrderedDict[TTKey, _TTEntry],
    state: State,
    key: TTKey,
    moves: list[int],
    c_uct: float,
    root_player: int,
) -> int:
    entry = _tt_get(tt, key)
    log_parent = math.log(entry.n + 1.0)
    maximize = int(state["current_player"]) == int(root_player)
    best_move = moves[0]
    best_score = float("-inf")
    for m in moves:
        ck = entry.children.get(m)
        if ck is None:
            score = float("inf")
        else:
            ch = _tt_get(tt, ck)
            if ch.n <= 0:
                score = float("inf")
            else:
                q = ch.q if maximize else -ch.q
                score = q + c_uct * math.sqrt(log_parent / (1.0 + ch.n))
        if score > best_score:
            best_score = score
            best_move = m
    return best_move


def _simulate_from(state: State, root_player: int, max_depth: int, eval_mode: str, rng) -> float:
    s = clone_state(state)
    depth = max(0, int(max_depth))
    for _ in range(depth):
        if bool(s.get("finished", False)):
            return _terminal_value(s, root_player)
        moves = legal_moves(s)
        if not moves:
            break
        m = moves[rng.randrange(len(moves))]
        play_turn(s, m)
    if bool(s.get("finished", False)):
        return _terminal_value(s, root_player)
    return _rollout_value(s, root_player, eval_mode=eval_mode)


def _tt_get(tt: OrderedDict[TTKey, _TTEntry], key: TTKey) -> _TTEntry:
    ent = tt.get(key)
    if ent is None:
        ent = _TTEntry()
        tt[key] = ent
    return ent


def _tt_touch(tt: OrderedDict[TTKey, _TTEntry], key: TTKey, cfg: MCTSConfig) -> None:
    if not cfg.use_tt:
        return
    try:
        tt.move_to_end(key)
    except KeyError:
        return
    if cfg.tt_max_entries and len(tt) > cfg.tt_max_entries:
        tt.popitem(last=False)


def _backprop_tt(tt: OrderedDict[TTKey, _TTEntry], path: list[TTKey], value_root_perspective: float, cfg: MCTSConfig) -> None:
    for k in path:
        ent = _tt_get(tt, k)
        ent.n += 1
        ent.w += value_root_perspective
        _tt_touch(tt, k, cfg)

