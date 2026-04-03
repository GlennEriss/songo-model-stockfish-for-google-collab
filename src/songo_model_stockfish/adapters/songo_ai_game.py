from __future__ import annotations

from typing import Any

from songo_model_stockfish.reference_songo.engine import create_state as _create_state
from songo_model_stockfish.reference_songo.game import clone_state as _clone_state
from songo_model_stockfish.reference_songo.game import legal_moves as _legal_moves
from songo_model_stockfish.reference_songo.game import simulate_move as _simulate_move


def _ensure_songo_ai_importable() -> None:
    return


def create_state() -> Any:
    return _create_state()


def clone_state(state: Any) -> Any:
    return _clone_state(state)


def legal_moves(state: Any) -> list[int]:
    return list(_legal_moves(state))


def simulate_move(state: Any, move: int) -> Any:
    return _simulate_move(state, move)


def is_terminal(state: Any) -> bool:
    return bool(state.get("finished", False))


def current_player(state: Any) -> int:
    return int(state["current_player"])


def scores(state: Any) -> tuple[int, int]:
    s = list(state.get("scores", [0, 0]))
    return int(s[0]), int(s[1])


def winner(state: Any) -> int | None:
    value = state.get("winner")
    return None if value is None else int(value)


def board_as_14(state: Any) -> list[int]:
    board = state["board"]
    south = list(board[0])
    north = list(board[1])
    return south + north


def to_raw_state(state: Any) -> dict[str, Any]:
    south_score, north_score = scores(state)
    return {
        "state_format_version": "v1",
        "board": board_as_14(state),
        "player_to_move": "south" if current_player(state) == 0 else "north",
        "scores": {
            "south": south_score,
            "north": north_score,
        },
        "turn_index": int(state.get("turn_index", 0)),
        "is_terminal": is_terminal(state),
    }
