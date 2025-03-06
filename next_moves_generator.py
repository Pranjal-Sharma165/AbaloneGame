import sys
import copy
import itertools
import time

from moves import Move, MoveError, PushNotAllowedError, BoardBoundaryError, InvalidDirectionError
from board_io import BoardIO

class NoLegalMovesError(Exception):
    """Raised when no legal moves can be generated."""
    pass

class NextMove:
    @staticmethod
    def board_to_canonical_string(_board: dict, _turn: str) -> str:

        black_coords = []
        white_coords = []
        for coord, cell in _board.items():
            if cell == BoardIO.NO_MARBLE:
                continue
            if cell == BoardIO.BLACK_MARBLE:
                black_coords.append(coord)
            elif cell == BoardIO.WHITE_MARBLE:
                white_coords.append(coord)

        def sort_key(c):
            return c[0], int(c[1:])
        black_coords.sort(key=sort_key)
        white_coords.sort(key=sort_key)
        black_tokens = [f"{c}b" for c in black_coords]
        white_tokens = [f"{c}w" for c in white_coords]
        tokens_line = ",".join(black_tokens + white_tokens)
        return tokens_line

    @staticmethod
    def find_all_groups_of_size_1_2_3(_board: dict, color: str) -> list:

        all_positions = [coord for coord, val in _board.items() if val == color]
        groups = set()
        for size in [1, 2, 3]:
            for combo in itertools.combinations(all_positions, size):
                combo_list = list(combo)
                if Move.are_coordinates_contiguous(combo_list):
                    groups.add(tuple(sorted(combo_list)))
        return list(groups)

    @staticmethod
    def generate_move_notation(marble_coords: list, direction: str) -> str:

        source_str = "".join(sorted(marble_coords))
        dest_coords = []
        for c in marble_coords:
            dr, dc = Move.DIRECTION_VECTORS[direction]
            new_row = chr(ord(c[0]) + dr)
            new_col = int(c[1:]) + dc
            dest_coords.append(f"{new_row}{new_col}")
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
                new_board = copy.deepcopy(_board)
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
            for notation, canon in _moves:
                f.write(notation + "\n")

    @staticmethod
    def save_board_configurations(_moves: list, _output_filename: str) -> None:

        with open(_output_filename, "w") as f:
            for notation, canon in _moves:
                f.write(canon + "\n")

    @staticmethod
    def generate_and_save_all_next_moves(_board: dict, color: str, _turn: str,
                                           _moves_filename: str, _boards_filename: str) -> list:

        _moves = NextMove.generate_all_next_moves(_board, color, _turn)
        NextMove.save_legal_move_notations(_moves, _moves_filename)
        NextMove.save_board_configurations(_moves, _boards_filename)
        return _moves

if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = input_filename[:-6]
    a = time.time()
    board, turn = BoardIO.import_current_text_to_board(input_filename)
    current_color = BoardIO.BLACK_MARBLE if turn == "Black" else BoardIO.WHITE_MARBLE

    moves_filename = f"{output_filename}.move"
    boards_filename = f"{output_filename}-Team3.board"

    moves = NextMove.generate_and_save_all_next_moves(board, current_color, turn,
                                                        moves_filename, boards_filename)
    b = time.time()
    print(f"time taken: {b - a} seconds")
