from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Board = List[List[int]]
State = Dict[str, object]
Pos = Tuple[int, int]

NUM_PITS = 7
START_SEEDS = 5
WIN_SCORE = 35


def create_board() -> Board:
    return [[START_SEEDS for _ in range(NUM_PITS)] for _ in range(2)]


def create_state() -> State:
    return {
        "board": create_board(),
        "scores": [0, 0],
        "current_player": 0,
        "finished": False,
        "winner": None,
        "reason": "",
    }


def other_player(player: int) -> int:
    return 1 - player


def pit_number_to_index(player: int, pit_number: int) -> int:
    if player == 0:
        return NUM_PITS - pit_number
    return pit_number - 1


def pit7_index(player: int) -> int:
    return pit_number_to_index(player, 7)


def non_pit7_total(board: Board, player: int) -> int:
    idx7 = pit7_index(player)
    return sum(board[player][idx] for idx in range(NUM_PITS) if idx != idx7)


def clockwise_ring() -> List[Pos]:
    top_row = [(1, idx) for idx in range(NUM_PITS)]
    bottom_row = [(0, idx) for idx in range(NUM_PITS - 1, -1, -1)]
    return top_row + bottom_row


def side_total(board: Board, player: int) -> int:
    return sum(board[player])


def steps_to_reach_opponent(player: int, pit_index: int) -> int:
    ring = clockwise_ring()
    start_pos = ring.index((player, pit_index))
    pos = start_pos
    steps = 0
    while True:
        pos = (pos + 1) % len(ring)
        steps += 1
        row, _ = ring[pos]
        if row == other_player(player):
            return steps


def move_can_transmit_with_selected_pit(seeds: int, player: int, pit_index: int) -> bool:
    required_steps = steps_to_reach_opponent(player, pit_index)
    return seeds >= required_steps


def can_transmit(board: Board, player: int) -> bool:
    idx7 = pit7_index(player)
    for pit_index in range(NUM_PITS):
        seeds = board[player][pit_index]
        if seeds == 0:
            continue
        if pit_index == idx7 and seeds == 1:
            continue
        if move_can_transmit_with_selected_pit(seeds, player, pit_index):
            return True
    return False


def can_capture_pit7_of_side(board: Board, side_owner: int) -> bool:
    idx7 = pit7_index(side_owner)
    for idx in range(NUM_PITS):
        if idx == idx7:
            continue
        seeds = board[side_owner][idx]
        if seeds > 4 or seeds < 2:
            return True
    return False


def validate_or_finish(state: State, pit_number: int) -> Tuple[bool, str, Optional[int]]:
    board: Board = state["board"]  # type: ignore[assignment]
    player = int(state["current_player"])
    opponent = other_player(player)
    pit_index = pit_number_to_index(player, pit_number)
    idx7 = pit7_index(player)

    if pit_number < 1 or pit_number > NUM_PITS:
        return False, "Choisis une case entre 1 et 7.", None

    selected_seeds = board[player][pit_index]

    if side_total(board, opponent) == 0:
        if not can_transmit(board, player):
            state["finished"] = True
            state["winner"] = player
            state["reason"] = "L'adversaire est vide et aucune transmission n'est possible."
            return False, str(state["reason"]), None
        if not move_can_transmit_with_selected_pit(selected_seeds, player, pit_index):
            return False, "Coup invalide: cette case ne transmet pas de pions a l'adversaire.", None

    if selected_seeds == 0:
        return False, "Coup invalide: case vide.", None

    if selected_seeds == 1 and pit_index == idx7:
        if non_pit7_total(board, player) != 0:
            return False, "Coup invalide: impossible de jouer la case 7 avec un seul pion.", None

    return True, "", pit_index


def consume_single_seed_from_pit7_if_only_seed(board: Board, player: int, pit_index: int) -> bool:
    if pit_index != pit7_index(player):
        return False
    if board[player][pit_index] != 1:
        return False
    if non_pit7_total(board, player) != 0:
        return False
    return True


def sow(board: Board, player: int, pit_index: int) -> Tuple[Optional[int], int, bool]:
    ring = clockwise_ring()
    start_pos = ring.index((player, pit_index))
    opponent_start_index = pit_number_to_index(other_player(player), 1)
    opponent_start_pos = ring.index((other_player(player), opponent_start_index))
    pos = start_pos
    seeds = board[player][pit_index]
    board[player][pit_index] = 0
    last_pos: Optional[int] = None
    captured_on_start = 0
    ended_by_start_capture = False
    distributed = 0

    while seeds > 0:
        next_pos = (pos + 1) % len(ring)
        next_row, _ = ring[next_pos]
        if distributed >= 13 and next_row == player:
            if seeds == 1:
                captured_on_start = 1
                seeds -= 1
                ended_by_start_capture = True
                break
            next_pos = opponent_start_pos

        if next_pos == start_pos:
            if seeds == 1:
                captured_on_start = 1
                seeds -= 1
                ended_by_start_capture = True
                break
            next_pos = (next_pos + 1) % len(ring)

        row, col = ring[next_pos]
        board[row][col] += 1
        seeds -= 1
        distributed += 1
        last_pos = next_pos
        pos = next_pos

    return last_pos, captured_on_start, ended_by_start_capture


def capture(board: Board, scores: List[int], player: int, last_pos: Optional[int]) -> int:
    if last_pos is None:
        return 0

    ring = clockwise_ring()
    opponent = other_player(player)
    captured = 0
    pos = last_pos

    while True:
        row, col = ring[pos]
        if row != opponent:
            break

        seeds = board[row][col]
        if seeds < 2 or seeds > 4:
            break

        if col == pit7_index(opponent) and not can_capture_pit7_of_side(board, opponent):
            break

        captured += seeds
        board[row][col] = 0
        pos = (pos - 1) % len(ring)

    scores[player] += captured
    return captured


def evaluate_end_of_turn(state: State, mover: int) -> None:
    board: Board = state["board"]  # type: ignore[assignment]
    scores: List[int] = state["scores"]  # type: ignore[assignment]
    opponent = other_player(mover)

    if scores[0] > WIN_SCORE:
        state["finished"] = True
        state["winner"] = 0
        state["reason"] = "Joueur 1 a mange plus de 35 pions."
        return
    if scores[1] > WIN_SCORE:
        state["finished"] = True
        state["winner"] = 1
        state["reason"] = "Joueur 2 a mange plus de 35 pions."
        return
    if scores[0] == WIN_SCORE and scores[1] == WIN_SCORE:
        state["finished"] = True
        state["winner"] = None
        state["reason"] = "Match nul: 35 a 35."
        return
    if side_total(board, opponent) == 0:
        state["finished"] = True
        state["winner"] = mover
        state["reason"] = "Le camp adverse est vide."
        return
    state["current_player"] = opponent


def play_turn(state: State, pit_number: int) -> str:
    if bool(state["finished"]):
        return "La partie est deja terminee."

    player = int(state["current_player"])
    board: Board = state["board"]  # type: ignore[assignment]
    scores: List[int] = state["scores"]  # type: ignore[assignment]

    ok, message, pit_index = validate_or_finish(state, pit_number)
    if not ok:
        return message

    assert pit_index is not None

    if consume_single_seed_from_pit7_if_only_seed(board, player, pit_index):
        board[player][pit_index] = 0
        scores[player] += 1
        evaluate_end_of_turn(state, player)
        return f"Joueur {player + 1} mange son dernier pion de la case 7."

    last_pos, captured_on_start, ended_by_start_capture = sow(board, player, pit_index)
    if captured_on_start > 0:
        scores[player] += captured_on_start

    captured = 0
    if not ended_by_start_capture:
        captured = capture(board, scores, player, last_pos)
    evaluate_end_of_turn(state, player)

    parts = [f"Joueur {player + 1} joue case {pit_number}."]
    if captured_on_start:
        parts.append("Dernier pion revenu sur case de depart: mange.")
    if captured:
        parts.append(f"Pions manges: {captured}.")
    return " ".join(parts)

