from __future__ import annotations

from typing import Any

import numpy as np
from songo_model_stockfish.adapters import songo_ai_game


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


def _raw_state_to_runtime_state(raw_state: dict[str, Any]) -> Any:
    board_flat = list(raw_state["board"])
    current_player = 0 if raw_state["player_to_move"] == "south" else 1
    scores = raw_state["scores"]
    return {
        "board": [board_flat[:7], board_flat[7:]],
        "scores": [int(scores["south"]), int(scores["north"])],
        "current_player": current_player,
        "finished": bool(raw_state.get("is_terminal", False)),
        "winner": None,
        "reason": "",
        "turn_index": int(raw_state.get("turn_index", 0)),
    }


def build_inference_tactical_analysis(raw_state: dict[str, Any], legal_moves: list[int]) -> dict[str, Any]:
    runtime_state = _raw_state_to_runtime_state(raw_state)
    root_player = int(runtime_state["current_player"])
    opponent = 1 - root_player
    root_scores = list(runtime_state["scores"])
    root_player_score = int(root_scores[root_player])
    root_opponent_score = int(root_scores[opponent])

    per_move: dict[str, dict[str, Any]] = {}
    capture_moves_count = 0
    safe_moves_count = 0
    risky_moves_count = 0
    max_immediate_score_gain = 0
    min_opponent_best_gain: int | None = None
    best_move: int | None = legal_moves[0] if legal_moves else None
    best_move_net_score_swing = -10**9

    for move in legal_moves:
        next_state = songo_ai_game.simulate_move(runtime_state, int(move))
        next_scores = list(next_state["scores"])
        immediate_score_gain = int(next_scores[root_player]) - root_player_score
        immediate_score_loss = int(next_scores[opponent]) - root_opponent_score
        opponent_legal_moves = list(songo_ai_game.legal_moves(next_state))

        opponent_best_immediate_gain = 0
        opponent_capture_moves = 0
        for opponent_move in opponent_legal_moves:
            reply_state = songo_ai_game.simulate_move(next_state, int(opponent_move))
            reply_scores = list(reply_state["scores"])
            gain = int(reply_scores[opponent]) - int(next_scores[opponent])
            if gain > 0:
                opponent_capture_moves += 1
            if gain > opponent_best_immediate_gain:
                opponent_best_immediate_gain = gain

        net_score_swing = int(immediate_score_gain) - int(opponent_best_immediate_gain)
        has_immediate_capture = immediate_score_gain > 0
        exposes_to_immediate_capture = opponent_best_immediate_gain > 0
        if has_immediate_capture:
            capture_moves_count += 1
        if exposes_to_immediate_capture:
            risky_moves_count += 1
        else:
            safe_moves_count += 1
        if net_score_swing > best_move_net_score_swing:
            best_move_net_score_swing = net_score_swing
            best_move = int(move)

        max_immediate_score_gain = max(max_immediate_score_gain, int(immediate_score_gain))
        if min_opponent_best_gain is None or opponent_best_immediate_gain < min_opponent_best_gain:
            min_opponent_best_gain = int(opponent_best_immediate_gain)

        per_move[str(move)] = {
            "immediate_score_gain": int(immediate_score_gain),
            "immediate_score_loss": int(immediate_score_loss),
            "has_immediate_capture": bool(has_immediate_capture),
            "opponent_legal_moves_count": int(len(opponent_legal_moves)),
            "opponent_capture_moves_count": int(opponent_capture_moves),
            "opponent_best_immediate_gain": int(opponent_best_immediate_gain),
            "exposes_to_immediate_capture": bool(exposes_to_immediate_capture),
            "net_score_swing_next_ply": int(net_score_swing),
        }

    best_move_stats = per_move.get(str(best_move), {})
    return {
        "summary": {
            "capture_moves_count": int(capture_moves_count),
            "safe_moves_count": int(safe_moves_count),
            "risky_moves_count": int(risky_moves_count),
            "max_immediate_score_gain": int(max_immediate_score_gain),
            "min_opponent_best_immediate_gain": int(min_opponent_best_gain or 0),
            "best_move_has_immediate_capture": bool(best_move_stats.get("has_immediate_capture", False)),
            "best_move_exposes_to_immediate_capture": bool(best_move_stats.get("exposes_to_immediate_capture", False)),
            "best_move_net_score_swing_next_ply": int(best_move_stats.get("net_score_swing_next_ply", 0)),
        },
        "per_move": per_move,
    }


def encode_model_features(
    raw_state: dict[str, Any],
    legal_moves: list[int],
    *,
    tactical_analysis: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base_features, legal_mask = encode_raw_state(raw_state, legal_moves)
    if tactical_analysis is None and legal_moves:
        tactical_analysis = build_inference_tactical_analysis(raw_state, legal_moves)
    tactical_features = encode_tactical_analysis(tactical_analysis)
    return np.concatenate([base_features, tactical_features]).astype(np.float32, copy=False), legal_mask


def adapt_feature_dim(features: np.ndarray, expected_dim: int) -> np.ndarray:
    if int(features.shape[0]) == int(expected_dim):
        return features
    if int(features.shape[0]) > int(expected_dim):
        return features[:expected_dim].astype(np.float32, copy=False)
    padding = np.zeros(int(expected_dim) - int(features.shape[0]), dtype=np.float32)
    return np.concatenate([features.astype(np.float32, copy=False), padding]).astype(np.float32, copy=False)
