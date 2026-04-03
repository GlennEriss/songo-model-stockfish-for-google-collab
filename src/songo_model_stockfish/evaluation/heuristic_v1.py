from __future__ import annotations

from songo_model_stockfish.adapters import songo_ai_game


def evaluate_state(state) -> float:
    player = songo_ai_game.current_player(state)
    opponent = 1 - player
    scores = songo_ai_game.scores(state)
    score_diff = scores[player] - scores[opponent]

    legal_count = len(songo_ai_game.legal_moves(state))

    board = songo_ai_game.board_as_14(state)
    start = 0 if player == 0 else 7
    own_side = board[start : start + 7]
    opp_start = 7 if player == 0 else 0
    opp_side = board[opp_start : opp_start + 7]

    own_mass = sum(own_side)
    opp_mass = sum(opp_side)
    mass_diff = own_mass - opp_mass

    capture_potential = sum(1 for seeds in opp_side if 2 <= seeds <= 4)
    risk = sum(1 for seeds in own_side if 2 <= seeds <= 4)

    return (
        100.0 * float(score_diff)
        + 5.0 * float(legal_count)
        + 2.0 * float(mass_diff)
        + 20.0 * float(capture_potential)
        - 18.0 * float(risk)
    )
