import re


class MoveError(Exception):
    """Base exception for all move-related errors."""
    pass


class MoveFormatError(MoveError):
    """Raised when the move command format is incorrect."""
    pass


class InvalidMarbleError(MoveError):
    """Raised when trying to move a marble that does not belong to the current player."""
    pass


class InvalidDirectionError(MoveError):
    """Raised when the direction of the move is invalid or inconsistent."""
    pass


class BoardBoundaryError(MoveError):
    """Raised when the move goes out of board boundaries."""
    pass


class OverlapError(MoveError):
    """Raised when marbles would overlap after a move."""
    pass


class PushNotAllowedError(MoveError):
    """Raised when a push move is not allowed due to insufficient pushing power."""
    pass


DIRECTION_VECTORS = {
    "upper_left": (1, 0),
    "upper_right": (1, 1),
    "left": (0, -1),
    "right": (0, 1),
    "down_left": (-1, -1),
    "down_right": (-1, 0)
}


def next_row(row: str) -> str:
    """Returns the next alphabetical row."""
    return chr(ord(row) + 1)


def prev_row(row: str) -> str:
    """Returns the previous alphabetical row."""
    return chr(ord(row) - 1)


def validate_move_directions(source_list: list[str], dest_list: list[str]) -> str:
    """
    Validates that multiple marble moves have consistent directions.

    :param source_list: list of source coordinates.
    :param dest_list: list of destination coordinates.
    :return: the common move direction if valid.
    """
    if len(source_list) != len(dest_list):
        raise MoveFormatError("The number of source and destination coordinates do not match")

    direction = get_move_direction(source_list[0], dest_list[0])
    for s, d in zip(source_list[1:], dest_list[1:]):
        if get_move_direction(s, d) != direction:
            raise InvalidDirectionError("Inconsistent move directions for multiple marbles")
    return direction


def parse_move_input(move_str: str) -> tuple[list[str], list[str]]:
    """
    Parses the move command input into source and destination lists.

    :param move_str: a string in the format
    :return: a tuple
    """
    move_str = move_str.replace(" ", "").upper()
    parts = move_str.split(",")
    if len(parts) != 2:
        raise MoveFormatError("Invalid move format")
    source_str, dest_str = parts
    source_list = re.findall(r"[A-Z]\d+", source_str)
    dest_list = re.findall(r"[A-Z]\d+", dest_str)
    if not source_list or not dest_list:
        raise MoveFormatError("Coordinates are not in the correct format")
    return source_list, dest_list


def get_move_direction(source: str, dest: str) -> str:
    """
    Determines the move direction based on the source and destination coordinates.

    :param source: starting coordinate (e.g., "E5").
    :param dest: destination coordinate (e.g., "F5").
    :return: one of the valid move directions.
    """
    s_row, s_col = source[0], int(source[1:])
    d_row, d_col = dest[0], int(dest[1:])
    if next_row(s_row) == d_row and s_col == d_col:
        return "upper_left"
    elif next_row(s_row) == d_row and s_col + 1 == d_col:
        return "upper_right"
    elif s_row == d_row and s_col - 1 == d_col:
        return "left"
    elif s_row == d_row and s_col + 1 == d_col:
        return "right"
    elif prev_row(s_row) == d_row and s_col - 1 == d_col:
        return "down_left"
    elif prev_row(s_row) == d_row and s_col == d_col:
        return "down_right"
    else:
        raise InvalidDirectionError("Cannot determine move direction.")


def transform_coordinate(coord: str, direction: str, board: dict) -> str:
    """
    Transforms a board coordinate in the given direction using DIRECTION_VECTORS.

    :param coord: starting coordinate (e.g., "E5").
    :param direction: one of the valid move directions.
    :param board: the board state dictionary.
    :return: new coordinate after applying the move.
    :raises BoardBoundaryError: if the new coordinate is not valid.
    """
    if direction not in DIRECTION_VECTORS:
        raise InvalidDirectionError("Illegal direction")
    row_delta, col_delta = DIRECTION_VECTORS[direction]
    row = coord[0]
    col = int(coord[1:])
    new_row = chr(ord(row) + row_delta)
    new_col = col + col_delta
    new_coord = f"{new_row}{new_col}"
    if new_coord not in board:
        raise BoardBoundaryError("Out of board boundaries.")
    return new_coord


def get_leading_marble(marble_coords: list[str], direction: str) -> str:
    """
    Returns the coordinate from marble_coords that is farthest in the given direction.

    :param marble_coords: list of coordinates.
    :param direction: the move direction.
    :return: the coordinate of the leading marble.
    """
    if direction not in DIRECTION_VECTORS:
        raise InvalidDirectionError("Wrong direction")
    vec = DIRECTION_VECTORS[direction]

    def coord_value(coord: str) -> int:
        row_val = ord(coord[0])
        col_val = int(coord[1:])
        return row_val * vec[0] + col_val * vec[1]

    return max(marble_coords, key=coord_value)


def move_marbles_cmd(board: dict, marble_coords: list[str], direction: str,
                     player_marble: str, opponent_marble: str) -> int:
    """
    Moves the marbles on the board in the specified direction. Handles both normal moves and inline push moves.

    :param board: current board state dictionary.
    :param marble_coords: list of coordinates for the marbles to move.
    :param direction: the move direction.
    :param player_marble: string representing the moving player's marble.
    :param opponent_marble: string representing the opponent's marble.
    :return: Score gained (0 if normal move, 1 if an opponent marble is pushed off).
    :raises OverlapError, PushNotAllowedError, MoveError: errors
    """
    score = 0
    leader = get_leading_marble(marble_coords, direction)
    try:
        next_cell = transform_coordinate(leader, direction, board)
    except BoardBoundaryError:
        next_cell = None

    if next_cell is None or board.get(next_cell, "Blank") == "Blank":
        new_coords = [transform_coordinate(coord, direction, board) for coord in marble_coords]
        for orig, new_coord in zip(marble_coords, new_coords):
            if new_coord not in marble_coords and board.get(new_coord, "Blank") != "Blank":
                raise OverlapError("Marbles cannot overlap!")
        for coord in marble_coords:
            board[coord] = "Blank"
        for new_coord in new_coords:
            board[new_coord] = player_marble
        return score
    else:
        if board[next_cell] == player_marble:
            raise OverlapError("Cannot push your own marble!")
        elif board[next_cell] == opponent_marble:
            opponent_coords = []
            current = next_cell
            while True:
                if board.get(current, "Blank") == opponent_marble:
                    opponent_coords.append(current)
                    try:
                        current = transform_coordinate(current, direction, board)
                    except BoardBoundaryError:
                        break
                else:
                    break
            if len(marble_coords) <= len(opponent_coords):
                raise PushNotAllowedError("Push not allowed.")
            try:
                final_cell = transform_coordinate(opponent_coords[-1], direction, board)
                if board.get(final_cell, "Blank") != "Blank":
                    raise BoardBoundaryError
            except BoardBoundaryError:
                board[opponent_coords[-1]] = "Blank"
                score = 1
                opponent_coords = opponent_coords[:-1]
            for coord in reversed(opponent_coords):
                try:
                    new_coord = transform_coordinate(coord, direction, board)
                    board[new_coord] = opponent_marble
                    board[coord] = "Blank"
                except BoardBoundaryError:
                    board[coord] = "Blank"
            new_coords = [transform_coordinate(coord, direction, board) for coord in marble_coords]
            for coord in marble_coords:
                board[coord] = "Blank"
            for new_coord in new_coords:
                board[new_coord] = player_marble
            return score
        else:
            raise MoveError("Unexpected Error")
