from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SONGO_AI_READY = False


def _ensure_songo_ai_importable() -> None:
    global _SONGO_AI_READY
    if _SONGO_AI_READY:
        return

    project_root = Path(__file__).resolve().parents[3]
    songo_ai_root = project_root.parent / "songo-ai"
    if not songo_ai_root.exists():
        raise FileNotFoundError(f"`songo-ai` introuvable a {songo_ai_root}")

    songo_ai_root_str = str(songo_ai_root)
    if songo_ai_root_str not in sys.path:
        sys.path.insert(0, songo_ai_root_str)
    _SONGO_AI_READY = True


def create_state() -> Any:
    _ensure_songo_ai_importable()
    from src.songo.engine import create_state as _create_state

    return _create_state()


def clone_state(state: Any) -> Any:
    _ensure_songo_ai_importable()
    from src.songo.game import clone_state as _clone_state

    return _clone_state(state)


def legal_moves(state: Any) -> list[int]:
    _ensure_songo_ai_importable()
    from src.songo.game import legal_moves as _legal_moves

    return list(_legal_moves(state))


def simulate_move(state: Any, move: int) -> Any:
    _ensure_songo_ai_importable()
    from src.songo.game import simulate_move as _simulate_move

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
