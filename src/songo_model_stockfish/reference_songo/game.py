from __future__ import annotations

import copy
from typing import List

from songo_model_stockfish.reference_songo.engine import (
    NUM_PITS,
    State,
    can_transmit,
    move_can_transmit_with_selected_pit,
    non_pit7_total,
    other_player,
    pit7_index,
    pit_number_to_index,
    play_turn,
    side_total,
)


def clone_state(state: State) -> State:
    return copy.deepcopy(state)


def legal_moves(state: State) -> List[int]:
    board = state["board"]  # type: ignore[assignment]
    player = int(state["current_player"])
    opponent = other_player(player)

    opponent_empty = side_total(board, opponent) == 0
    if opponent_empty and (not can_transmit(board, player)):
        return []

    idx7 = pit7_index(player)
    moves: List[int] = []
    for pit_number in range(1, NUM_PITS + 1):
        pit_index = pit_number_to_index(player, pit_number)
        seeds = board[player][pit_index]
        if seeds == 0:
            continue
        if seeds == 1 and pit_index == idx7 and non_pit7_total(board, player) != 0:
            continue
        if opponent_empty and (not move_can_transmit_with_selected_pit(seeds, player, pit_index)):
            continue
        moves.append(pit_number)

    return moves


def simulate_move(state: State, pit: int) -> State:
    s2 = clone_state(state)
    play_turn(s2, pit)
    return s2


def terminal_winner_utility(state: State, root_player: int) -> float:
    if not bool(state["finished"]):
        raise ValueError("State is not terminal")
    winner = state["winner"]
    if winner is None:
        return 0.0
    return 1e9 if int(winner) == root_player else -1e9

