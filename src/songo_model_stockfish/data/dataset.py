from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetSample:
    sample_id: str
    game_id: str
    ply: int
    state: dict[str, Any]
    player_to_move: str
    legal_moves: list[int]
    policy_target: dict[str, Any]
    value_target: float
    teacher_engine: str
    teacher_level: str
    seed: int
