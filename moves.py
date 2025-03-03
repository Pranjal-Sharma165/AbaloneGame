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
    """Raised when a push move is not allowed."""
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
    """
    Returns the next alphabetical row.
    """
    return chr(ord(row) + 1)


def prev_row(row: str) -> str:
    """
    Returns the previous alphabetical row.
    """
    return chr(ord(row) - 1)


def get_neighbors(coord: str) -> list[str]:
    """
    Returns all neighbor coordinates for a given coordinate.
    """
    neighbors = []
    row = coord[0]
    col = int(coord[1:])
    for dr, dc in DIRECTION_VECTORS.values():
        new_row = chr(ord(row) + dr)
        new_col = col + dc
        neighbors.append(f"{new_row}{new_col}")
    return neighbors


def are_coordinates_contiguous(coords: list[str]) -> bool:
    """
    Checks if the given coordinates form a contiguous group.
    """
    if not coords:
        return True
    visited = set()
    def dfs(c):
        if c in visited:
            return
        visited.add(c)
        for nbr in get_neighbors(c):
            if nbr in coords and nbr not in visited:
                dfs(nbr)
    dfs(coords[0])
    return len(visited) == len(coords)


def are_marbles_aligned(marble_coords: list[str], direction: str) -> bool:
    """
    Checks if the marbles are on a straight line
    """
    if len(marble_coords) <= 1:
        return True
    dr, dc = DIRECTION_VECTORS[direction]
    ref_row = ord(marble_coords[0][0])
    ref_col = int(marble_coords[0][1:])
    multiples = []
    for coord in marble_coords:
        r = ord(coord[0])
        c = int(coord[1:])
        delta_r = r - ref_row
        delta_c = c - ref_col
        if dr != 0:
            k = delta_r / dr
        elif dc != 0:
            k = delta_c / dc
        else:
            k = 0
        if not k.is_integer() or (dc * int(k)) != delta_c:
            return False
        multiples.append(int(k))
    multiples.sort()
    for i in range(1, len(multiples)):
        if multiples[i] - multiples[i - 1] != 1:
            return False
    return True


def validate_move_directions(source_list: list[str], dest_list: list[str]) -> str:
    """
    Validates that multiple marble moves have consistent directions.

    :param source_list: list of source coordinates.
    :param dest_list: list of destination coordinates.
    :return: the common move direction if valid.
    """
    if len(source_list) != len(dest_list):
        raise MoveFormatError("The number of source and destination coordinates do not match.")
    if len(source_list) > 3:
        raise MoveFormatError("Cannot move more than 3 marbles at once.")
    if len(source_list) > 1 and not are_coordinates_contiguous(source_list):
        raise MoveError("Marbles must be contiguous.")
    direction = get_move_direction(source_list[0], dest_list[0])
    for s, d in zip(source_list[1:], dest_list[1:]):
        if get_move_direction(s, d) != direction:
            raise InvalidDirectionError("Inconsistent move directions for multiple marbles.")
    return direction


def parse_move_input(move_str: str) -> tuple[list[str], list[str]]:
    """
    Parses the move command input into source and destination lists.

    :param move_str: a string representing the move command.
    :return: a tuple (source_list, dest_list).
    """
    move_str = move_str.replace(" ", "").upper()
    parts = move_str.split(",")
    if len(parts) != 2:
        raise MoveFormatError("Invalid move format.")
    source_str, dest_str = parts
    source_list = re.findall(r"[A-Z]\d+", source_str)
    dest_list = re.findall(r"[A-Z]\d+", dest_str)
    if not source_list or not dest_list:
        raise MoveFormatError("Coordinates are not in the correct format.")
    return source_list, dest_list


def get_move_direction(source: str, dest: str) -> str:
    """
    Determines the move direction based on the source and destination coordinates.

    :param source: starting coordinate
    :param dest: destination coordinate
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

    :param coord: starting coordinate
    :param direction: one of the valid move directions.
    :param board: the board state dictionary.
    :return: new coordinate after applying the move.
    :raises BoardBoundaryError: if the new coordinate is not valid.
    """
    if direction not in DIRECTION_VECTORS:
        raise InvalidDirectionError("Illegal direction")
    dr, dc = DIRECTION_VECTORS[direction]
    row = coord[0]
    col = int(coord[1:])
    new_row = chr(ord(row) + dr)
    new_col = col + dc
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
        return ord(coord[0]) * vec[0] + int(coord[1:]) * vec[1]
    return max(marble_coords, key=coord_value)

def are_marbles_collinear(marble_coords: list[str]) -> bool:
    """
    Returns True if the given three marbles are collinear.

    """
    if len(marble_coords) <= 2:
        return True
    for d in DIRECTION_VECTORS.keys():
        if are_marbles_aligned(marble_coords, d):
            return True
    return False

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

    if len(marble_coords) == 3 and not are_marbles_collinear(marble_coords):
        raise InvalidDirectionError("Three marbles must be collinear to move.")

    destinations = {}
    for coord in marble_coords:
        try:
            destinations[coord] = transform_coordinate(coord, direction, board)
        except BoardBoundaryError:
            destinations[coord] = None

    lead = get_leading_marble(marble_coords, direction)
    if destinations[lead] is None:
        raise BoardBoundaryError("Leading marble cannot move off-board in a normal move.")

    inline = are_marbles_aligned(marble_coords, direction)

    if board.get(destinations[lead], "Blank") == opponent_marble:
        if not inline:
            raise InvalidDirectionError("Push moves require an inline formation of marbles.")
        opponent_chain = []
        current = destinations[lead]
        while True:
            if board.get(current, "Blank") == opponent_marble:
                opponent_chain.append(current)
                try:
                    current = transform_coordinate(current, direction, board)
                except BoardBoundaryError:
                    break
            else:
                break
        if len(marble_coords) <= len(opponent_chain):
            raise PushNotAllowedError("Insufficient numbers for push move.")
        try:
            final_cell = transform_coordinate(opponent_chain[-1], direction, board)
            if board.get(final_cell, "Blank") != "Blank":
                raise PushNotAllowedError("Push blocked: final cell is not empty.")
        except BoardBoundaryError:
            board[opponent_chain[-1]] = "Blank"
            score = 1
            opponent_chain = opponent_chain[:-1]
        for opp in reversed(opponent_chain):
            try:
                new_opp = transform_coordinate(opp, direction, board)
                board[new_opp] = opponent_marble
                board[opp] = "Blank"
            except BoardBoundaryError:
                board[opp] = "Blank"
        for coord in marble_coords:
            if destinations[coord] not in marble_coords and board.get(destinations[coord], "Blank") != "Blank":
                raise OverlapError("Destination cell is blocked for push move.")
        for coord in marble_coords:
            board[coord] = "Blank"
        for coord in marble_coords:
            board[destinations[coord]] = player_marble
        return score
    else:
        for coord in marble_coords:
            if destinations[coord] is None:
                raise BoardBoundaryError("Cannot move off-board.")
            if destinations[coord] not in marble_coords and board.get(destinations[coord], "Blank") != "Blank":
                raise OverlapError("Destination cell is blocked.")
        for coord in marble_coords:
            board[coord] = "Blank"
        for coord in marble_coords:
            board[destinations[coord]] = player_marble
        return score