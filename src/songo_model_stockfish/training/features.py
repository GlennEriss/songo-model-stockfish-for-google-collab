from __future__ import annotations

from typing import Any

import numpy as np


def encode_raw_state(raw_state: dict[str, Any], legal_moves: list[int]) -> tuple[np.ndarray, np.ndarray]:
    board = np.asarray(raw_state["board"], dtype=np.float32)
    player = 0.0 if raw_state["player_to_move"] == "south" else 1.0
    scores = raw_state["scores"]
    features = np.concatenate(
        [
            board,
            np.asarray([player, float(scores["south"]), float(scores["north"])], dtype=np.float32),
        ]
    )
    legal_mask = np.zeros(7, dtype=np.float32)
    for move in legal_moves:
        legal_mask[int(move) - 1] = 1.0
    return features, legal_mask
