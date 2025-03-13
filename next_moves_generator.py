import sys
import itertools
import time

from moves import Move, MoveError, PushNotAllowedError, BoardBoundaryError, InvalidDirectionError
from board_io import BoardIO

class NoLegalMovesError(Exception):
    """Raised when no legal moves can be generated."""
    pass

class NextMove:
    DIRECTION_VECTORS = {
        "upper_left": (1, 0),
        "upper_right": (1, 1),
        "left": (0, -1),
        "right": (0, 1),
        "down_left": (-1, -1),
        "down_right": (-1, 0)
    }

    @staticmethod
    def board_to_canonical_string(_board: dict, _turn: str) -> str:
        black_coords = [coord for coord, cell in _board.items() if cell == BoardIO.BLACK_MARBLE]
        white_coords = [coord for coord, cell in _board.items() if cell == BoardIO.WHITE_MARBLE]
        sort_key = lambda c: (c[0], int(c[1:]))
        black_coords.sort(key=sort_key)
        white_coords.sort(key=sort_key)
        tokens_line = ",".join([f"{c}b" for c in black_coords] + [f"{c}w" for c in white_coords])
        return tokens_line

    @staticmethod
    def find_all_groups_of_size_1_2_3(_board: dict, color: str) -> list:
        all_positions = [coord for coord, val in _board.items() if val == color]
        groups = set()
        for size in (1, 2):
            for combo in itertools.combinations(all_positions, size):
                combo_sorted = tuple(sorted(combo))
                if size == 1:
                    groups.add(combo_sorted)
                elif size == 2:
                    if Move.are_coordinates_contiguous(combo_sorted):
                        groups.add(combo_sorted)

        for combo in itertools.combinations(all_positions, 3):
                if Move.are_marbles_in_allowed_pattern(combo):
                    groups.add(combo)
        return list(groups)

    @staticmethod
    def generate_move_notation(marble_coords: list, direction: str) -> str:
        source_str = "".join(sorted(marble_coords))
        dest_coords = [
            f"{chr(ord(c[0]) + Move.DIRECTION_VECTORS[direction][0])}{int(c[1:]) + Move.DIRECTION_VECTORS[direction][1]}"
            for c in marble_coords
        ]
        dest_str = "".join(sorted(dest_coords))
        return f"{source_str},{dest_str}"

    @staticmethod
    def generate_all_next_moves(_board: dict, color: str, _turn: str) -> list:
        results = []
        seen = set()
        groups = NextMove.find_all_groups_of_size_1_2_3(_board, color)
        directions = list(Move.DIRECTION_VECTORS.keys())
        opponent_color = BoardIO.WHITE_MARBLE if color == BoardIO.BLACK_MARBLE else BoardIO.BLACK_MARBLE
        for group in groups:
            group_list = list(group)
            for d in directions:
                new_board = _board.copy()
                try:
                    Move.move_marbles_cmd(new_board, group_list, d, color, opponent_color)
                    canon = NextMove.board_to_canonical_string(new_board, _turn)
                    if canon not in seen:
                        seen.add(canon)
                        notation = NextMove.generate_move_notation(group_list, d)
                        results.append((notation, canon))
                except (MoveError, PushNotAllowedError, BoardBoundaryError, InvalidDirectionError):
                    continue
        results.sort(key=lambda x: x[0])
        if not results:
            raise NoLegalMovesError("No legal moves found for the current board state.")
        return results

    @staticmethod
    def save_legal_move_notations(_moves: list, _output_filename: str) -> None:
        with open(_output_filename, "w") as f:
            f.write("\n".join(notation for notation, _ in _moves) + "\n")

    @staticmethod
    def save_board_configurations(_moves: list, _output_filename: str) -> None:
        with open(_output_filename, "w") as f:
            f.write("\n".join(canon for _, canon in _moves) + "\n")

    @staticmethod
    def generate_and_save_all_next_moves(_board: dict, color: str, _turn: str,
                                           _moves_filename: str, _boards_filename: str) -> list:
        moves = NextMove.generate_all_next_moves(_board, color, _turn)
        NextMove.save_legal_move_notations(moves, _moves_filename)
        NextMove.save_board_configurations(moves, _boards_filename)
        return moves

if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = input_filename[:-6]
    start_time = time.time()
    board, turn = BoardIO.import_current_text_to_board(input_filename)
    current_color = BoardIO.BLACK_MARBLE if turn == "Black" else BoardIO.WHITE_MARBLE

    moves_filename = f"{output_filename}.move"
    boards_filename = f"{output_filename}-Team3.board"

    moves = NextMove.generate_and_save_all_next_moves(board, current_color, turn,
                                                      moves_filename, boards_filename)
    print(f"time taken: {time.time() - start_time} seconds")
