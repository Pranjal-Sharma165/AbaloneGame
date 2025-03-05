import sys
import copy
import itertools
import time

from moves import (
    move_marbles_cmd,
    MoveError,
    PushNotAllowedError,
    are_marbles_aligned,
    get_move_direction,
    BoardBoundaryError,
    InvalidDirectionError,
    get_leading_marble,
    DIRECTION_VECTORS,
    transform_coordinate,
    are_coordinates_contiguous
)

from board_io import (
    import_current_text_to_board,
    WHITE_MARBLE,
    BLACK_MARBLE,
    NO_MARBLE
)


def board_to_canonical_string(board: dict, turn: str) -> str:
    """
    Changes board dictionary to a canonical string representation.
    """
    black_coords = []
    white_coords = []
    for coord, cell in board.items():
        if cell == NO_MARBLE:
            continue
        if cell == BLACK_MARBLE:
            black_coords.append(coord)
        elif cell == WHITE_MARBLE:
            white_coords.append(coord)

    def sort_key(c):
        return c[0], int(c[1:])

    black_coords.sort(key=sort_key)
    white_coords.sort(key=sort_key)
    black_tokens = [f"{c}b" for c in black_coords]
    white_tokens = [f"{c}w" for c in white_coords]
    tokens_line = ",".join(black_tokens + white_tokens)
    return tokens_line


def find_all_groups_of_size_1_2_3(board: dict, color: str) -> list:
    """
    Extracts groups of 1, 2, or 3 contiguous marbles of the specified color.
    """
    all_positions = [coord for coord, val in board.items() if val == color]
    groups = set()
    for size in [1, 2, 3]:
        for combo in itertools.combinations(all_positions, size):
            combo_list = list(combo)
            if are_coordinates_contiguous(combo_list):
                groups.add(tuple(sorted(combo_list)))
    return list(groups)


def generate_move_notation(marble_coords: list, direction: str) -> str:
    """
    Returns a move notation string for a given group of marbles moving in a direction.
    """
    source_str = "".join(sorted(marble_coords))
    dest_coords = []
    for c in marble_coords:
        dr, dc = DIRECTION_VECTORS[direction]
        new_row = chr(ord(c[0]) + dr)
        new_col = int(c[1:]) + dc
        dest_coords.append(f"{new_row}{new_col}")
    dest_str = "".join(sorted(dest_coords))
    return f"{source_str},{dest_str}"


def generate_all_next_moves(board: dict, color: str, turn: str) -> list:
    """
    Generates all legal next-ply moves by trying every contiguous group of 1-3 marbles in every direction.
    """
    results = []
    seen = set()
    groups = find_all_groups_of_size_1_2_3(board, color)
    directions = list(DIRECTION_VECTORS.keys())
    opponent_color = WHITE_MARBLE if color == BLACK_MARBLE else BLACK_MARBLE
    for group in groups:
        group_list = list(group)
        for d in directions:
            new_board = copy.deepcopy(board)
            try:
                move_marbles_cmd(new_board, group_list, d, color, opponent_color)
                canon = board_to_canonical_string(new_board, turn)
                if canon not in seen:
                    seen.add(canon)
                    notation = generate_move_notation(group_list, d)
                    results.append((notation, canon))
            except (MoveError, PushNotAllowedError, BoardBoundaryError, InvalidDirectionError):
                continue
    results.sort(key=lambda x: x[0])
    return results


def save_moves_to_file(moves: list, output_filename: str):
    """
    Saves the canonical board configurations, one per row, to the specified file.
    """
    with open(output_filename, "w") as f:
        for notation, canon in moves:
            f.write(canon + "\n")
    print(f"Output saved to {output_filename}")


def save_legal_move_notations(moves: list, output_filename: str) -> None:
    """
    Saves the legal move notations, one per row, to the specified file.
    """
    with open(output_filename, "w") as f:
        for notation, canon in moves:
            f.write(notation + "\n")


def save_board_configurations(moves: list, output_filename: str) -> None:
    """
    Saves the canonical board configurations resulting from each move
    """
    with open(output_filename, "w") as f:
        for notation, canon in moves:
            f.write(canon + "\n")


def generate_and_save_all_next_moves(board: dict, color: str, turn: str,
                                     moves_filename: str, boards_filename: str) -> list:
    """
    Generates all legal next-ply moves, then saves
    """
    moves = generate_all_next_moves(board, color, turn)
    save_legal_move_notations(moves, moves_filename)
    save_board_configurations(moves, boards_filename)
    return moves


if __name__ == "__main__":
    #test
    input_filename = sys.argv[1]
    output_filename = input_filename[:-6]
    print(output_filename)
    a = time.time()
    board, turn = import_current_text_to_board(input_filename)
    current_color = BLACK_MARBLE if turn == "Black" else WHITE_MARBLE

    moves_filename = f"{output_filename}.move"
    boards_filename = f"{output_filename}-Team3.board"

    moves = generate_and_save_all_next_moves(board, current_color, turn,
                                             moves_filename, boards_filename)
    b = time.time()
    print(f"time taken: {b - a} seconds")
