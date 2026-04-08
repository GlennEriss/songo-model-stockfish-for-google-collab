from __future__ import annotations

from typing import Any

import numpy as np


TACTICAL_SUMMARY_KEYS = [
    "capture_moves_count",
    "safe_moves_count",
    "risky_moves_count",
    "max_immediate_score_gain",
    "min_opponent_best_immediate_gain",
    "best_move_has_immediate_capture",
    "best_move_exposes_to_immediate_capture",
    "best_move_net_score_swing_next_ply",
]

TACTICAL_PER_MOVE_KEYS = [
    "teacher_score",
    "immediate_score_gain",
    "immediate_score_loss",
    "has_immediate_capture",
    "opponent_legal_moves_count",
    "opponent_capture_moves_count",
    "opponent_best_immediate_gain",
    "exposes_to_immediate_capture",
    "net_score_swing_next_ply",
]


def _normalize_tactical_value(key: str, value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    numeric = float(value)
    if key == "teacher_score":
        return float(np.tanh(numeric / 200.0))
    if key in {"immediate_score_gain", "immediate_score_loss", "opponent_best_immediate_gain", "net_score_swing_next_ply"}:
        return float(np.tanh(numeric / 12.0))
    if key in {"capture_moves_count", "safe_moves_count", "risky_moves_count", "opponent_legal_moves_count", "opponent_capture_moves_count"}:
        return numeric / 7.0
    if key == "min_opponent_best_immediate_gain":
        return float(np.tanh(numeric / 12.0))
    return numeric


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


def encode_tactical_analysis(tactical_analysis: dict[str, Any] | None) -> np.ndarray:
    if not tactical_analysis:
        return np.zeros(len(TACTICAL_SUMMARY_KEYS) + (7 * len(TACTICAL_PER_MOVE_KEYS)), dtype=np.float32)

    summary = tactical_analysis.get("summary", {}) if isinstance(tactical_analysis, dict) else {}
    per_move = tactical_analysis.get("per_move", {}) if isinstance(tactical_analysis, dict) else {}
    values: list[float] = []

    for key in TACTICAL_SUMMARY_KEYS:
        values.append(_normalize_tactical_value(key, summary.get(key, 0.0)))

    for move in range(1, 8):
        move_payload = per_move.get(str(move), {})
        for key in TACTICAL_PER_MOVE_KEYS:
            values.append(_normalize_tactical_value(key, move_payload.get(key, 0.0)))

    return np.asarray(values, dtype=np.float32)


def encode_model_features(
    raw_state: dict[str, Any],
    legal_moves: list[int],
    *,
    tactical_analysis: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_features, legal_mask = encode_raw_state(raw_state, legal_moves)
    tactical_features = encode_tactical_analysis(tactical_analysis)
    return np.concatenate([base_features, tactical_features]).astype(np.float32, copy=False), legal_mask


def adapt_feature_dim(features: np.ndarray, expected_dim: int) -> np.ndarray:
    if int(features.shape[0]) == int(expected_dim):
        return features
    if int(features.shape[0]) > int(expected_dim):
        return features[:expected_dim].astype(np.float32, copy=False)
    padding = np.zeros(int(expected_dim) - int(features.shape[0]), dtype=np.float32)
    return np.concatenate([features.astype(np.float32, copy=False), padding]).astype(np.float32, copy=False)
