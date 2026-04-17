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
    choose_fallbacks: tuple[int, int]
    opening_plies: tuple[int, ...]
    first_move_agent_a: int | None
    first_move_agent_b: int | None
    reason: str
    starter: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = list(self.scores)
        payload["think_ms"] = list(self.think_ms)
        payload["choose_fallbacks"] = list(self.choose_fallbacks)
        payload["opening_plies"] = list(self.opening_plies)
        return payload


def play_match(agent_a: AgentLike, agent_b: AgentLike, *, max_moves: int = 300, starter: int = 0) -> MatchResult:
    state = songo_ai_game.create_state()
    think = [0.0, 0.0]
    agents = [agent_a, agent_b] if starter == 0 else [agent_b, agent_a]
    moves = 0
    end_reason = "finished"
    move_history: list[int] = []
    first_move_by_logical_agent: dict[int, int | None] = {0: None, 1: None}

    while not songo_ai_game.is_terminal(state) and moves < max_moves:
        legal = songo_ai_game.legal_moves(state)
        if not legal:
            end_reason = "no_legal_moves_available"
            break

        player = songo_ai_game.current_player(state)
        t0 = time.perf_counter()
        try:
            move, _info = agents[player].choose(songo_ai_game.clone_state(state))
        except Exception as exc:
            raise RuntimeError(
                "Agent choose failed | "
                f"player={player} | starter={starter} | agent={agents[player].display_name} | "
                f"legal={legal} | cause={type(exc).__name__}: {exc}"
            ) from exc
        think[player] += (time.perf_counter() - t0) * 1000.0
        if move not in legal:
            raise ValueError(
                "Agent returned illegal move | "
                f"player={player} | starter={starter} | agent={agents[player].display_name} | "
                f"move={move} | legal={legal}"
            )
        move_int = int(move)
        move_history.append(move_int)
        logical_agent = int(player) if int(starter) == 0 else int(1 - int(player))
        if first_move_by_logical_agent.get(logical_agent) is None:
            first_move_by_logical_agent[logical_agent] = move_int
        state = songo_ai_game.simulate_move(state, move)
        moves += 1

    raw_winner = songo_ai_game.winner(state)
    if raw_winner is None and end_reason == "no_legal_moves_available":
        south_score, north_score = songo_ai_game.scores(state)
        if south_score > north_score:
            raw_winner = 0
        elif north_score > south_score:
            raw_winner = 1
    logical_winner = raw_winner
    if starter == 1 and raw_winner is not None:
        logical_winner = 0 if raw_winner == 1 else 1

    return MatchResult(
        winner=logical_winner,
        moves=moves,
        scores=songo_ai_game.scores(state),
        think_ms=(round(think[0], 2), round(think[1], 2)),
        choose_fallbacks=(0, 0),
        opening_plies=tuple(int(m) for m in move_history[:8]),
        first_move_agent_a=(None if first_move_by_logical_agent.get(0) is None else int(first_move_by_logical_agent[0])),
        first_move_agent_b=(None if first_move_by_logical_agent.get(1) is None else int(first_move_by_logical_agent[1])),
        reason="finished" if songo_ai_game.is_terminal(state) else (end_reason if end_reason != "finished" else f"max_moves_reached:{max_moves}"),
        starter=starter,
    )
