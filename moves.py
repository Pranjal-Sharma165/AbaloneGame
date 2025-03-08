import functools
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

class NonContiguousError(MoveError):
    """Raised when marbles are not contiguous."""
    pass

class NonCollinearError(MoveError):
    """Raised when three marbles are not collinear to move."""
    pass

class InlineMovementError(MoveError):
    """Raised when three marbles push marbles, not being in-line state"""
    pass

class ThreeMarblesError(MoveError):
    """Three marbles' special rule"""
    pass

class Move:
    DIRECTION_VECTORS = {
        "upper_left": (1, 0),
        "upper_right": (1, 1),
        "left": (0, -1),
        "right": (0, 1),
        "down_left": (-1, -1),
        "down_right": (-1, 0)
    }

    @staticmethod
    def next_row(row: str) -> str:
        """
        Returns the next alphabetical row.
        """
        return chr(ord(row) + 1)

    @staticmethod
    def prev_row(row: str) -> str:
        """
        Returns the previous alphabetical row.
        """
        return chr(ord(row) - 1)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_neighbors(coord: str) -> list:
        """
        Returns all neighbor coordinates for a given coordinate.
        """
        row = coord[0]
        col = int(coord[1:])
        return [f"{chr(ord(row) + dr)}{col + dc}" for dr, dc in Move.DIRECTION_VECTORS.values()]

    @staticmethod
    def are_coordinates_contiguous(coords: list) -> bool:
        """
        Checks if the given coordinates form a contiguous group.
        """
        n = len(coords)
        if n <= 1:
            return True
        if n == 2:
            a, b = coords
            return b in Move.get_neighbors(a)
        if n == 3:
            a, b, c = coords
            neigh_a = set(Move.get_neighbors(a))
            neigh_b = set(Move.get_neighbors(b))
            count = (b in neigh_a) + (c in neigh_a) + (c in neigh_b)
            return count >= 2
        visited = set()
        def dfs(_c):
            if _c in visited:
                return
            visited.add(_c)
            for nbr in Move.get_neighbors(_c):
                if nbr in coords and nbr not in visited:
                    dfs(nbr)
        dfs(coords[0])
        return len(visited) == len(coords)

    @staticmethod
    def are_marbles_aligned(marble_coords: list, direction: str) -> bool:
        """
        Checks if the marbles are on a straight line
        """
        if len(marble_coords) <= 1:
            return True
        dr, dc = Move.DIRECTION_VECTORS[direction]
        ref_row = ord(marble_coords[0][0])
        ref_col = int(marble_coords[0][1:])
        multiples = []
        for coord in marble_coords:
            r = ord(coord[0])
            c = int(coord[1:])
            delta_r = r - ref_row
            delta_c = c - ref_col
            if dr != 0:
                if delta_r % dr != 0:
                    return False
                k = delta_r // dr
            elif dc != 0:
                if delta_c % dc != 0:
                    return False
                k = delta_c // dc
            else:
                k = 0
            if dc * k != delta_c:
                return False
            multiples.append(k)
        multiples.sort()
        for i in range(1, len(multiples)):
            if multiples[i] - multiples[i - 1] != 1:
                return False
        return True

    @staticmethod
    def are_marbles_in_allowed_pattern(marble_coords: list) -> bool:
        """
        Checks if three marbles are in one of the allowed patterns.
        """
        if len(marble_coords) != 3:
            return True
        def parse_coord(coord: str):
            return coord[0].upper(), int(coord[1:])
        coords = sorted([parse_coord(c) for c in marble_coords], key=lambda x: (x[0], x[1]))
        m0, m1, m2 = coords
        if (m0[0] == m1[0] and m0[1] == m1[1] - 1 and
            m2[0] == chr(ord(m1[0]) + 1) and m2[1] == m1[1]):
            return True
        if (m0[0] == chr(ord(m1[0]) - 1) and m0[1] == m1[1] and
            m2[0] == chr(ord(m1[0]) + 1) and m2[1] == m1[1]):
            return True
        if (m0[0] == chr(ord(m1[0]) - 1) and m0[1] == m1[1] - 1 and
            m2[0] == chr(ord(m1[0]) + 1) and m2[1] == m1[1] + 1):
            return True
        return False

    @staticmethod
    def validate_move_directions(source_list: list, dest_list: list) -> str:
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
        if len(source_list) > 1 and not Move.are_coordinates_contiguous(source_list):
            raise NonContiguousError("Marbles must be contiguous.")
        if len(source_list) == 3 and not Move.are_marbles_in_allowed_pattern(source_list):
            raise ThreeMarblesError("Three marbles must be the allowed patterns")
        direction = Move.get_move_direction(source_list[0], dest_list[0])
        for s, d in zip(source_list[1:], dest_list[1:]):
            if Move.get_move_direction(s, d) != direction:
                raise InvalidDirectionError("Inconsistent move directions for multiple marbles.")
        return direction

    @staticmethod
    def parse_move_input(move_str: str) -> tuple:
        """
        Parses the move command input into source and destination lists.
        """
        move_str = move_str.replace(" ", "").upper()
        parts = move_str.split(",")
        if len(parts) != 2:
            raise MoveFormatError("Invalid move format.")
        source_list = re.findall(r"[A-Z]\d+", parts[0])
        dest_list = re.findall(r"[A-Z]\d+", parts[1])
        if not source_list or not dest_list:
            raise MoveFormatError("Coordinates are not in the correct format.")
        return source_list, dest_list

    @staticmethod
    def get_move_direction(source: str, dest: str) -> str:
        """
        Determines the move direction based on the source and destination coordinates.

        :param source: starting coordinate
        :param dest: destination coordinate
        :return: one of the valid move directions.
        """
        s_row, s_col = source[0], int(source[1:])
        d_row, d_col = dest[0], int(dest[1:])
        if Move.next_row(s_row) == d_row and s_col == d_col:
            return "upper_left"
        elif Move.next_row(s_row) == d_row and s_col + 1 == d_col:
            return "upper_right"
        elif s_row == d_row and s_col - 1 == d_col:
            return "left"
        elif s_row == d_row and s_col + 1 == d_col:
            return "right"
        elif Move.prev_row(s_row) == d_row and s_col - 1 == d_col:
            return "down_left"
        elif Move.prev_row(s_row) == d_row and s_col == d_col:
            return "down_right"
        else:
            raise InvalidDirectionError("Cannot determine move direction.")

    @staticmethod
    def transform_coordinate(coord: str, direction: str, board: dict) -> str:
        """
        Transforms a board coordinate in the given direction using DIRECTION_VECTORS.

        :param coord: starting coordinate
        :param direction: one of the valid move directions.
        :param board: the board state dictionary.
        :return: new coordinate after applying the move.
        :raises BoardBoundaryError: if the new coordinate is not valid.
        """
        if direction not in Move.DIRECTION_VECTORS:
            raise InvalidDirectionError("Illegal direction")
        dr, dc = Move.DIRECTION_VECTORS[direction]
        row = coord[0]
        col = int(coord[1:])
        new_coord = f"{chr(ord(row) + dr)}{col + dc}"
        if new_coord not in board:
            raise BoardBoundaryError("Out of board boundaries.")
        return new_coord

    @staticmethod
    def get_leading_marble(marble_coords: list, direction: str) -> str:
        """
        Returns the coordinate from marble_coords that is farthest in the given direction.

        :param marble_coords: list of coordinates.
        :param direction: the move direction.
        :return: the coordinate of the leading marble.
        """
        if direction not in Move.DIRECTION_VECTORS:
            raise InvalidDirectionError("Wrong direction")
        vec = Move.DIRECTION_VECTORS[direction]
        def coord_value(coord: str) -> int:
            return ord(coord[0]) * vec[0] + int(coord[1:]) * vec[1]
        return max(marble_coords, key=coord_value)

    @staticmethod
    def are_marbles_collinear(marble_coords: list) -> bool:
        """
        Returns True if the given marbles are collinear.
        """
        if len(marble_coords) <= 2:
            return True
        for d in Move.DIRECTION_VECTORS.keys():
            if Move.are_marbles_aligned(marble_coords, d):
                return True
        return False

    @staticmethod
    def move_marbles_cmd(board: dict, marble_coords: list, direction: str,
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
        if len(marble_coords) == 3 and not Move.are_marbles_collinear(marble_coords):
            raise NonCollinearError("Three marbles must be collinear to move.")

        destinations = {}
        for coord in marble_coords:
            try:
                destinations[coord] = Move.transform_coordinate(coord, direction, board)
            except BoardBoundaryError:
                destinations[coord] = None

        lead = Move.get_leading_marble(marble_coords, direction)
        if destinations[lead] is None:
            raise BoardBoundaryError("Leading marble cannot move off-board in a normal move.")

        inline = Move.are_marbles_aligned(marble_coords, direction)

        if board.get(destinations[lead], "Blank") == opponent_marble:
            if not inline:
                raise InlineMovementError("Push moves require an inline formation of marbles.")
            opponent_chain = []
            current = destinations[lead]
            while board.get(current, "Blank") == opponent_marble:
                opponent_chain.append(current)
                try:
                    current = Move.transform_coordinate(current, direction, board)
                except BoardBoundaryError:
                    break
            if len(marble_coords) <= len(opponent_chain):
                raise PushNotAllowedError("Insufficient numbers for push move.")
            try:
                final_cell = Move.transform_coordinate(opponent_chain[-1], direction, board)
                if board.get(final_cell, "Blank") != "Blank":
                    raise PushNotAllowedError("Push blocked: final cell is not empty.")
            except BoardBoundaryError:
                board[opponent_chain[-1]] = "Blank"
                score = 1
                opponent_chain = opponent_chain[:-1]
            for opp in reversed(opponent_chain):
                try:
                    new_opp = Move.transform_coordinate(opp, direction, board)
                    board[new_opp] = opponent_marble
                    board[opp] = "Blank"
                except BoardBoundaryError:
                    board[opp] = "Blank"
            for coord in marble_coords:
                if destinations[coord] not in marble_coords and board.get(destinations[coord], "Blank") != "Blank":
                    raise PushNotAllowedError("Destination cell is blocked for push move.")
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
                    raise PushNotAllowedError("Destination cell is blocked.")
            for coord in marble_coords:
                board[coord] = "Blank"
            for coord in marble_coords:
                board[destinations[coord]] = player_marble
            return score
