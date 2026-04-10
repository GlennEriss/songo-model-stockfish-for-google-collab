from __future__ import annotations

from typing import Any

import numpy as np

from songo_model_stockfish.adapters import songo_ai_game
from songo_model_stockfish.reference_songo.engine import (
    NUM_PITS,
    can_capture_pit7_of_side,
    consume_single_seed_from_pit7_if_only_seed,
    evaluate_end_of_turn,
    move_can_transmit_with_selected_pit,
    non_pit7_total,
    other_player,
    pit7_index,
    pit_number_to_index,
    side_total,
    sow,
    steps_to_reach_opponent,
    validate_or_finish,
    capture,
    clockwise_ring,
)


TACTICAL_SUMMARY_KEYS = [
    "capture_moves_count",
    "safe_moves_count",
    "risky_moves_count",
    "max_immediate_score_gain",
    "min_opponent_best_immediate_gain",
    "best_move_has_immediate_capture",
    "best_move_exposes_to_immediate_capture",
    "best_move_net_score_swing_next_ply",
    "solidarity_active",
    "solidarity_optimal_moves_count",
    "interdit_solo_active",
    "interdit_duo_active",
    "interdit_sec_active",
    "zt_moves_count",
    "ts_moves_count",
    "tr_moves_count",
    "tm_moves_count",
    "best_move_is_solidarity_move",
    "best_move_is_zt",
    "best_move_is_ts",
    "best_move_is_tr",
    "best_move_is_tm",
    "bidoua_self",
    "bidoua_opp",
    "bidoua_advantage",
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
    "selected_seeds",
    "transmission_count",
    "is_legal_move",
    "is_solidarity_move",
    "is_interdit_solo_candidate",
    "is_interdit_duo_candidate",
    "is_interdit_sec_candidate",
    "is_zt",
    "is_ts",
    "is_tr",
    "is_tm",
]

_TRANSFER_STATE_WEIGHTS = {
    "ZT": 1.0,
    "TS": 2.0,
    "TR": 3.0,
    "TM": 4.0,
}


def _normalize_tactical_value(key: str, value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    numeric = float(value)
    if key == "teacher_score":
        return float(np.tanh(numeric / 200.0))
    if key in {
        "immediate_score_gain",
        "immediate_score_loss",
        "opponent_best_immediate_gain",
        "net_score_swing_next_ply",
        "max_immediate_score_gain",
        "min_opponent_best_immediate_gain",
    }:
        return float(np.tanh(numeric / 12.0))
    if key in {
        "capture_moves_count",
        "safe_moves_count",
        "risky_moves_count",
        "opponent_legal_moves_count",
        "opponent_capture_moves_count",
        "solidarity_optimal_moves_count",
        "zt_moves_count",
        "ts_moves_count",
        "tr_moves_count",
        "tm_moves_count",
        "transmission_count",
    }:
        return numeric / 7.0
    if key == "selected_seeds":
        return float(np.tanh(numeric / 20.0))
    if key in {"bidoua_self", "bidoua_opp", "bidoua_advantage"}:
        return float(np.tanh(numeric / 20.0))
    return numeric


def _runtime_state_for_player(runtime_state: dict[str, Any], player: int) -> dict[str, Any]:
    state = songo_ai_game.clone_state(runtime_state)
    state["current_player"] = int(player)
    state["finished"] = False
    state["winner"] = None
    state["reason"] = ""
    return state


def _classify_transfer_state(player: int, pit_index: int, seeds: int) -> str:
    if seeds <= 0:
        return "ZT"
    steps = steps_to_reach_opponent(player, pit_index)
    if seeds < steps:
        return "ZT"
    remaining = seeds - steps
    if remaining < NUM_PITS:
        return "TS"
    if remaining < (2 * NUM_PITS):
        return "TR"
    return "TM"


def _initial_opponent_transmission_count(player: int, pit_index: int, seeds: int) -> int:
    if seeds <= 0:
        return 0
    steps = steps_to_reach_opponent(player, pit_index)
    if seeds < steps:
        return 0
    return int(min(NUM_PITS, seeds - steps + 1))


def _capture_path(board: list[list[int]], player: int, last_pos: int | None) -> tuple[list[tuple[int, int]], bool]:
    if last_pos is None:
        return [], False

    ring = clockwise_ring()
    opponent = other_player(player)
    captured_positions: list[tuple[int, int]] = []
    blocked_by_pit7_rule = False
    pos = last_pos

    while True:
        row, col = ring[pos]
        if row != opponent:
            break

        seeds = int(board[row][col])
        if seeds < 2 or seeds > 4:
            break

        if col == pit7_index(opponent) and not can_capture_pit7_of_side(board, opponent):
            blocked_by_pit7_rule = True
            break

        captured_positions.append((row, col))
        pos = (pos - 1) % len(ring)

    return captured_positions, blocked_by_pit7_rule


def _state_flags_from_transfer_state(transfer_state: str) -> dict[str, bool]:
    return {
        "is_zt": transfer_state == "ZT",
        "is_ts": transfer_state == "TS",
        "is_tr": transfer_state == "TR",
        "is_tm": transfer_state == "TM",
    }


def _analyze_move(runtime_state: dict[str, Any], move: int) -> dict[str, Any]:
    state_before = _runtime_state_for_player(runtime_state, int(runtime_state["current_player"]))
    board_before = state_before["board"]
    scores_before = list(state_before["scores"])
    player = int(state_before["current_player"])
    opponent = other_player(player)
    pit_index = pit_number_to_index(player, int(move))
    selected_seeds = int(board_before[player][pit_index])
    transmission_count = _initial_opponent_transmission_count(player, pit_index, selected_seeds)
    transfer_state = _classify_transfer_state(player, pit_index, selected_seeds)
    interdit_solo_candidate = bool(
        selected_seeds == 1 and pit_index == pit7_index(player) and non_pit7_total(board_before, player) != 0
    )

    result = {
        "selected_seeds": int(selected_seeds),
        "transmission_count": int(transmission_count),
        "transfer_state": transfer_state,
        "is_legal_move": False,
        "is_solidarity_move": False,
        "is_interdit_solo_candidate": interdit_solo_candidate,
        "is_interdit_duo_candidate": False,
        "is_interdit_sec_candidate": False,
        "immediate_score_gain": 0,
        "immediate_score_loss": 0,
        "has_immediate_capture": False,
        "opponent_legal_moves_count": 0,
        "opponent_capture_moves_count": 0,
        "opponent_best_immediate_gain": 0,
        "exposes_to_immediate_capture": False,
        "net_score_swing_next_ply": 0,
        "capture_path_length": 0,
        "capture_blocked_by_pit7_rule": False,
        **_state_flags_from_transfer_state(transfer_state),
    }

    state_after = _runtime_state_for_player(runtime_state, player)
    ok, _message, validated_pit_index = validate_or_finish(state_after, int(move))
    if not ok:
        return result

    assert validated_pit_index is not None
    result["is_legal_move"] = True

    if consume_single_seed_from_pit7_if_only_seed(state_after["board"], player, validated_pit_index):
        state_after["board"][player][validated_pit_index] = 0
        state_after["scores"][player] += 1
        evaluate_end_of_turn(state_after, player)
        captured_total = 1
        capture_path_length = 1
        blocked_by_pit7_rule = False
    else:
        last_pos, captured_on_start, ended_by_start_capture = sow(state_after["board"], player, validated_pit_index)
        capture_positions, blocked_by_pit7_rule = (
            _capture_path(state_after["board"], player, last_pos) if not ended_by_start_capture else ([], False)
        )
        if captured_on_start > 0:
            state_after["scores"][player] += captured_on_start
        captured_total = int(captured_on_start)
        if not ended_by_start_capture:
            captured_total += int(capture(state_after["board"], state_after["scores"], player, last_pos))
        evaluate_end_of_turn(state_after, player)
        capture_path_length = len(capture_positions)

    immediate_score_gain = int(state_after["scores"][player]) - int(scores_before[player])
    immediate_score_loss = int(state_after["scores"][opponent]) - int(scores_before[opponent])
    opponent_legal_moves = list(songo_ai_game.legal_moves(state_after))

    opponent_best_immediate_gain = 0
    opponent_capture_moves = 0
    for opponent_move in opponent_legal_moves:
        reply_state = songo_ai_game.simulate_move(state_after, int(opponent_move))
        reply_scores = list(reply_state["scores"])
        gain = int(reply_scores[opponent]) - int(state_after["scores"][opponent])
        if gain > 0:
            opponent_capture_moves += 1
        if gain > opponent_best_immediate_gain:
            opponent_best_immediate_gain = int(gain)

    has_immediate_capture = captured_total > 0
    exposes_to_immediate_capture = opponent_best_immediate_gain > 0
    net_score_swing = int(immediate_score_gain) - int(opponent_best_immediate_gain)
    interdit_duo_candidate = bool(
        selected_seeds == 2
        and validated_pit_index == pit7_index(player)
        and non_pit7_total(board_before, player) != 0
        and transmission_count == 2
        and not has_immediate_capture
    )
    interdit_sec_candidate = capture_path_length >= NUM_PITS

    result.update(
        {
            "is_interdit_duo_candidate": interdit_duo_candidate,
            "is_interdit_sec_candidate": interdit_sec_candidate,
            "immediate_score_gain": int(immediate_score_gain),
            "immediate_score_loss": int(immediate_score_loss),
            "has_immediate_capture": bool(has_immediate_capture),
            "opponent_legal_moves_count": int(len(opponent_legal_moves)),
            "opponent_capture_moves_count": int(opponent_capture_moves),
            "opponent_best_immediate_gain": int(opponent_best_immediate_gain),
            "exposes_to_immediate_capture": bool(exposes_to_immediate_capture),
            "net_score_swing_next_ply": int(net_score_swing),
            "capture_path_length": int(capture_path_length),
            "capture_blocked_by_pit7_rule": bool(blocked_by_pit7_rule),
        }
    )
    return result


def _build_player_move_analyses(
    runtime_state: dict[str, Any],
    player: int,
) -> tuple[dict[str, dict[str, Any]], list[int], dict[str, Any]]:
    player_state = _runtime_state_for_player(runtime_state, player)
    analyses = {str(move): _analyze_move(player_state, move) for move in range(1, NUM_PITS + 1)}
    legal_moves = [move for move in range(1, NUM_PITS + 1) if bool(analyses[str(move)]["is_legal_move"])]

    opponent = other_player(player)
    solidarity_active = side_total(player_state["board"], opponent) == 0
    solidarity_required_transmission = 0
    if solidarity_active:
        max_transmission = max(
            (int(item["transmission_count"]) for item in analyses.values() if int(item["selected_seeds"]) > 0),
            default=0,
        )
        solidarity_required_transmission = int(NUM_PITS if max_transmission >= NUM_PITS else max_transmission)
        if solidarity_required_transmission > 0:
            for item in analyses.values():
                item["is_solidarity_move"] = bool(
                    item["is_legal_move"] and int(item["transmission_count"]) == solidarity_required_transmission
                )

    return analyses, legal_moves, {
        "solidarity_active": bool(solidarity_active),
        "solidarity_required_transmission": int(solidarity_required_transmission),
    }


def _move_bidoua_score(move_payload: dict[str, Any]) -> float:
    if not bool(move_payload.get("is_legal_move", False)):
        return 0.0
    transfer_state = str(move_payload.get("transfer_state", "ZT"))
    score = float(_TRANSFER_STATE_WEIGHTS.get(transfer_state, 0.0))
    if bool(move_payload.get("is_solidarity_move", False)):
        score += 0.5
    if bool(move_payload.get("has_immediate_capture", False)):
        score += 0.5
    if bool(move_payload.get("exposes_to_immediate_capture", False)):
        score -= 0.5
    return max(score, 0.0)


def build_runtime_tactical_analysis(
    runtime_state: dict[str, Any],
    legal_moves: list[int],
    *,
    move_scores: dict[int, float] | None = None,
    best_move: int | None = None,
) -> dict[str, Any]:
    root_player = int(runtime_state["current_player"])
    root_analyses, computed_legal_moves, root_context = _build_player_move_analyses(runtime_state, root_player)
    effective_legal_moves = [int(move) for move in legal_moves if int(move) in computed_legal_moves] or computed_legal_moves

    if best_move is None and effective_legal_moves:
        if move_scores:
            best_move = max(effective_legal_moves, key=lambda move: float(move_scores.get(int(move), float("-inf"))))
        else:
            best_move = max(
                effective_legal_moves,
                key=lambda move: int(root_analyses[str(move)].get("net_score_swing_next_ply", 0)),
            )

    opponent_state = _runtime_state_for_player(runtime_state, other_player(root_player))
    opponent_analyses, _opponent_legal_moves, _opponent_context = _build_player_move_analyses(
        opponent_state,
        int(opponent_state["current_player"]),
    )

    capture_moves_count = sum(
        1 for move in effective_legal_moves if bool(root_analyses[str(move)].get("has_immediate_capture", False))
    )
    safe_moves_count = sum(
        1 for move in effective_legal_moves if not bool(root_analyses[str(move)].get("exposes_to_immediate_capture", False))
    )
    risky_moves_count = sum(
        1 for move in effective_legal_moves if bool(root_analyses[str(move)].get("exposes_to_immediate_capture", False))
    )
    max_immediate_score_gain = max(
        (int(root_analyses[str(move)].get("immediate_score_gain", 0)) for move in effective_legal_moves),
        default=0,
    )
    min_opponent_best_gain = min(
        (int(root_analyses[str(move)].get("opponent_best_immediate_gain", 0)) for move in effective_legal_moves),
        default=0,
    )
    zt_moves_count = sum(1 for move in effective_legal_moves if bool(root_analyses[str(move)].get("is_zt", False)))
    ts_moves_count = sum(1 for move in effective_legal_moves if bool(root_analyses[str(move)].get("is_ts", False)))
    tr_moves_count = sum(1 for move in effective_legal_moves if bool(root_analyses[str(move)].get("is_tr", False)))
    tm_moves_count = sum(1 for move in effective_legal_moves if bool(root_analyses[str(move)].get("is_tm", False)))
    solidarity_optimal_moves_count = sum(
        1 for move in effective_legal_moves if bool(root_analyses[str(move)].get("is_solidarity_move", False))
    )

    bidoua_self = float(sum(_move_bidoua_score(root_analyses[str(move)]) for move in range(1, NUM_PITS + 1)))
    bidoua_opp = float(sum(_move_bidoua_score(opponent_analyses[str(move)]) for move in range(1, NUM_PITS + 1)))
    bidoua_advantage = float(bidoua_self - bidoua_opp)

    best_move_stats = root_analyses.get(str(best_move), {})
    per_move: dict[str, dict[str, Any]] = {}
    for move in range(1, NUM_PITS + 1):
        item = dict(root_analyses[str(move)])
        if move_scores is not None:
            item["teacher_score"] = float(move_scores.get(int(move), 0.0))
        per_move[str(move)] = item

    return {
        "summary": {
            "capture_moves_count": int(capture_moves_count),
            "safe_moves_count": int(safe_moves_count),
            "risky_moves_count": int(risky_moves_count),
            "max_immediate_score_gain": int(max_immediate_score_gain),
            "min_opponent_best_immediate_gain": int(min_opponent_best_gain),
            "best_move_has_immediate_capture": bool(best_move_stats.get("has_immediate_capture", False)),
            "best_move_exposes_to_immediate_capture": bool(best_move_stats.get("exposes_to_immediate_capture", False)),
            "best_move_net_score_swing_next_ply": int(best_move_stats.get("net_score_swing_next_ply", 0)),
            "solidarity_active": bool(root_context.get("solidarity_active", False)),
            "solidarity_optimal_moves_count": int(solidarity_optimal_moves_count),
            "solidarity_required_transmission": int(root_context.get("solidarity_required_transmission", 0)),
            "interdit_solo_active": any(
                bool(item.get("is_interdit_solo_candidate", False)) for item in root_analyses.values()
            ),
            "interdit_duo_active": any(
                bool(item.get("is_interdit_duo_candidate", False)) for item in root_analyses.values()
            ),
            "interdit_sec_active": any(
                bool(item.get("is_interdit_sec_candidate", False)) for item in root_analyses.values()
            ),
            "zt_moves_count": int(zt_moves_count),
            "ts_moves_count": int(ts_moves_count),
            "tr_moves_count": int(tr_moves_count),
            "tm_moves_count": int(tm_moves_count),
            "best_move_is_solidarity_move": bool(best_move_stats.get("is_solidarity_move", False)),
            "best_move_is_zt": bool(best_move_stats.get("is_zt", False)),
            "best_move_is_ts": bool(best_move_stats.get("is_ts", False)),
            "best_move_is_tr": bool(best_move_stats.get("is_tr", False)),
            "best_move_is_tm": bool(best_move_stats.get("is_tm", False)),
            "best_move_transfer_state": str(best_move_stats.get("transfer_state", "")),
            "bidoua_self": float(bidoua_self),
            "bidoua_opp": float(bidoua_opp),
            "bidoua_advantage": float(bidoua_advantage),
        },
        "per_move": per_move,
    }


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
    return build_runtime_tactical_analysis(runtime_state, legal_moves)


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
