from __future__ import annotations

from dataclasses import asdict
import time
from dataclasses import dataclass
from typing import Any, Protocol

from songo_model_stockfish.adapters import songo_ai_game


class AgentLike(Protocol):
    @property
    def display_name(self) -> str:
        ...

    def choose(self, state: Any) -> tuple[int, dict[str, Any]]:
        ...


@dataclass
class MatchResult:
    winner: int | None
    moves: int
    scores: tuple[int, int]
    think_ms: tuple[float, float]
    reason: str
    starter: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = list(self.scores)
        payload["think_ms"] = list(self.think_ms)
        return payload


def play_match(agent_a: AgentLike, agent_b: AgentLike, *, max_moves: int = 300, starter: int = 0) -> MatchResult:
    state = songo_ai_game.create_state()
    think = [0.0, 0.0]
    agents = [agent_a, agent_b] if starter == 0 else [agent_b, agent_a]
    moves = 0

    while not songo_ai_game.is_terminal(state) and moves < max_moves:
        player = songo_ai_game.current_player(state)
        t0 = time.perf_counter()
        move, _info = agents[player].choose(songo_ai_game.clone_state(state))
        think[player] += (time.perf_counter() - t0) * 1000.0
        legal = songo_ai_game.legal_moves(state)
        if move not in legal:
            move = legal[0]
        state = songo_ai_game.simulate_move(state, move)
        moves += 1

    raw_winner = songo_ai_game.winner(state)
    logical_winner = raw_winner
    if starter == 1 and raw_winner is not None:
        logical_winner = 0 if raw_winner == 1 else 1

    return MatchResult(
        winner=logical_winner,
        moves=moves,
        scores=songo_ai_game.scores(state),
        think_ms=(round(think[0], 2), round(think[1], 2)),
        reason="finished" if songo_ai_game.is_terminal(state) else f"max_moves_reached:{max_moves}",
        starter=starter,
    )
