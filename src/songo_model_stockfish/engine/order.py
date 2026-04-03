from __future__ import annotations

from songo_model_stockfish.adapters import songo_ai_game


def order_moves(state, moves: list[int]) -> list[int]:
    scored: list[tuple[float, int]] = []
    root_player = songo_ai_game.current_player(state)
    root_scores = songo_ai_game.scores(state)
    root_score = root_scores[root_player]
    for move in moves:
        next_state = songo_ai_game.simulate_move(state, move)
        next_scores = songo_ai_game.scores(next_state)
        delta = next_scores[root_player] - root_score
        scored.append((float(delta), move))
    scored.sort(reverse=True)
    return [move for _score, move in scored]
