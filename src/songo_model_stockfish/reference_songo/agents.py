from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

from songo_model_stockfish.reference_songo.levels import get_config, get_mcts_config
from songo_model_stockfish.reference_songo.mcts import choose_move as choose_move_mcts
from songo_model_stockfish.reference_songo.minimax import choose_move as choose_move_minimax

MoveResult = Tuple[int, Dict[str, Any]]


class Agent(Protocol):
    @property
    def display_name(self) -> str:
        ...

    def choose(self, state) -> MoveResult:
        ...


class MinimaxAgent:
    def __init__(self, level: str) -> None:
        self._level = level
        self._name = f"Minimax beginner" if level == "beginner" else f"IA minimax {level}"

    @property
    def display_name(self) -> str:
        return self._name

    def choose(self, state) -> MoveResult:
        return choose_move_minimax(state, get_config(self._level))


class MCTSAgent:
    def __init__(self, level: str) -> None:
        self._level = level
        self._name = f"IA MCTS {level}"

    @property
    def display_name(self) -> str:
        return self._name

    def choose(self, state) -> MoveResult:
        return choose_move_mcts(state, get_mcts_config(self._level))

