# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=True
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True

import time
import numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport malloc, free

ctypedef np.int8_t INT8_t
ctypedef np.uint8_t UINT8_t
ctypedef np.int32_t INT32_t

WHITE_MARBLE = "#D9D9D9"
BLACK_MARBLE = "#8A8A8A"
NO_MARBLE = "Blank"

DIRECTION_VECTORS = {
    "upper_left": (1, 0),
    "upper_right": (1, 1),
    "left": (0, -1),
    "right": (0, 1),
    "down_left": (-1, -1),
    "down_right": (-1, 0)
}

VALID_COORDS = {
    (9, 5), (9, 6), (9, 7), (9, 8), (9, 9),
    (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9),
    (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9),
    (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
    (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)
}

VALID_COORDS_ARRAY = np.array(list(VALID_COORDS), dtype=np.int32)

VALID_COORDS_LOOKUP = np.zeros((10, 10), dtype=np.uint8)
for coord in VALID_COORDS:
    VALID_COORDS_LOOKUP[coord[0], coord[1]] = 1

cdef UINT8_t[:, :] VALID_COORDS_VIEW = VALID_COORDS_LOOKUP

cdef int[:, :] DIRECTION_VECTORS_ARRAY = np.zeros((6, 2), dtype=np.int32)

cdef int dir_idx = 0
for direction, vector in DIRECTION_VECTORS.items():
    DIRECTION_VECTORS_ARRAY[dir_idx, 0] = vector[0]
    DIRECTION_VECTORS_ARRAY[dir_idx, 1] = vector[1]
    dir_idx += 1

VALID_COORDS_VIEW_EXPORT = VALID_COORDS_VIEW
DIRECTION_VECTORS_ARRAY_EXPORT = DIRECTION_VECTORS_ARRAY

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_valid_coord_fast(int row, int col):
    if 0 <= row < 10 and 0 <= col < 10:
        return VALID_COORDS_VIEW[row, col] == 1
    return False

cpdef bint is_valid_coord(tuple coord):
    cdef int row = coord[0]
    cdef int col = coord[1]
    return is_valid_coord_fast(row, col)

cpdef list convert_board_format(dict board_dict):
    cdef list black_marbles = []
    cdef list white_marbles = []
    cdef str coord, marble_type
    cdef int row, col

    for coord, marble_type in board_dict.items():
        row = ord(coord[0]) - ord('A') + 1
        col = int(coord[1:])

        if marble_type == BLACK_MARBLE:
            black_marbles.append([row, col])
        elif marble_type == WHITE_MARBLE:
            white_marbles.append([row, col])

    black_marbles.sort()
    white_marbles.sort()

    return [black_marbles, white_marbles]

cpdef list parse_move_input(str move_text):
    cdef list source_coords = []
    cdef list dest_coords = []
    cdef str letter, number
    cdef int row, col, i
    cdef list parts

    move_text = move_text.upper().strip()

    if ',' in move_text:
        parts = [part.strip() for part in move_text.split(',')]
    else:
        parts = [move_text[:len(move_text) // 2], move_text[len(move_text) // 2:]]

        if len(move_text) >= 4:
            for i in range(2, len(move_text), 2):
                if move_text[i - 2] != move_text[i]:
                    parts = [move_text[:i], move_text[i:]]
                    break

    cdef str source_text = parts[0].replace(" ", "")
    for i in range(0, len(source_text), 2):
        if i + 1 < len(source_text):
            letter = source_text[i]
            number = source_text[i + 1]
            row = ord(letter) - ord('A') + 1
            col = int(number)
            source_coords.append([row, col])

    cdef str dest_text = parts[1].replace(" ", "")
    for i in range(0, len(dest_text), 2):
        if i + 1 < len(dest_text):
            letter = dest_text[i]
            number = dest_text[i + 1]
            row = ord(letter) - ord('A') + 1
            col = int(number)
            dest_coords.append([row, col])

    return [source_coords, dest_coords]

cpdef list lists_to_sets(list board):
    cdef set black_set = {tuple(coord) for coord in board[0]}
    cdef set white_set = {tuple(coord) for coord in board[1]}
    return [black_set, white_set]

def move_validation(list source_coords, list dest_coords, list board, str current_player_color):
    source_coords_tuples = [tuple(coord) for coord in source_coords]
    dest_coords_tuples = [tuple(coord) for coord in dest_coords]

    black_marbles_set = {tuple(marble) for marble in board[0]}
    white_marbles_set = {tuple(marble) for marble in board[1]}

    if current_player_color == "BLACK":
        player_marbles_set = black_marbles_set
        opponent_marbles_set = white_marbles_set
    else:
        player_marbles_set = white_marbles_set
        opponent_marbles_set = black_marbles_set

    for coord_tuple in source_coords_tuples:
        if coord_tuple not in player_marbles_set:
            return False, "only your marbles"

    num_marbles = len(source_coords)
    if num_marbles < 1 or num_marbles > 3:
        return False, "move 1, 2, or 3 marbles at a time"

    if len(source_coords) != len(dest_coords):
        return False, "Number of source and destination coordinates must match"

    for coord_tuple in dest_coords_tuples:
        if coord_tuple not in VALID_COORDS:
            return False, "Destination coordinates must be on the board"

    if num_marbles == 2:
        diff_vector = (source_coords[1][0] - source_coords[0][0], source_coords[1][1] - source_coords[0][1])

        is_adjacent = False
        for vector in DIRECTION_VECTORS.values():
            if diff_vector == vector or diff_vector == (-vector[0], -vector[1]):
                is_adjacent = True
                break

        if not is_adjacent:
            return False, "source marbles must be adjacent"

    if num_marbles == 3:
        sorted_coords = sorted(source_coords)

        diff1 = (sorted_coords[1][0] - sorted_coords[0][0], sorted_coords[1][1] - sorted_coords[0][1])

        is_adjacent1 = False
        for vector in DIRECTION_VECTORS.values():
            if diff1 == vector or diff1 == (-vector[0], -vector[1]):
                is_adjacent1 = True
                break

        if not is_adjacent1:
            return False, "marbles must be adjacent to each other"

        diff2 = (sorted_coords[2][0] - sorted_coords[1][0], sorted_coords[2][1] - sorted_coords[1][1])

        is_adjacent2 = False
        for vector in DIRECTION_VECTORS.values():
            if diff2 == vector or diff2 == (-vector[0], -vector[1]):
                is_adjacent2 = True
                break

        if not is_adjacent2:
            return False, "marbles must be adjacent to each other"

        if diff1 != diff2:
            return False, "three marbles must be in a straight line"

    if num_marbles == 1:
        diff = (dest_coords[0][0] - source_coords[0][0], dest_coords[0][1] - source_coords[0][1])
        move_direction = None

        for direction, vector in DIRECTION_VECTORS.items():
            if vector == diff:
                move_direction = direction
                break

        if move_direction is None:
            return False, "invalid movement direction for a single marble"

        dest_tuple = tuple(dest_coords[0])
        if dest_tuple in black_marbles_set or dest_tuple in white_marbles_set:
            return False, "destination position must be empty"

    elif num_marbles == 2 or num_marbles == 3:
        directions = []
        for i in range(num_marbles):
            diff = (dest_coords[i][0] - source_coords[i][0], dest_coords[i][1] - source_coords[i][1])
            move_direction = None

            for direction, vector in DIRECTION_VECTORS.items():
                if vector == diff:
                    move_direction = direction
                    break

            if move_direction is None:
                return False, "invalid movement direction"

            directions.append(move_direction)

        if len(set(directions)) != 1:
            return False, "all marbles must move in the same direction"

        move_direction_str = directions[0]
        direction_vector = DIRECTION_VECTORS[move_direction_str]

        is_line_push = True
        if direction_vector[0] > 0 or (direction_vector[0] == 0 and direction_vector[1] > 0):
            sorted_source = sorted(source_coords, reverse=True)
        else:
            sorted_source = sorted(source_coords)

        for i in range(1, len(sorted_source)):
            prev_pos = sorted_source[i - 1]
            curr_pos = sorted_source[i]
            diff = (prev_pos[0] - curr_pos[0], prev_pos[1] - curr_pos[1])

            if diff != direction_vector and diff != (-direction_vector[0], -direction_vector[1]):
                is_line_push = False
                break

        for i, dest in enumerate(dest_coords):
            dest_tuple = tuple(dest)

            if dest_tuple in player_marbles_set or dest_tuple in opponent_marbles_set:
                if dest_tuple in player_marbles_set:
                    is_moving_source = False
                    for src in source_coords:
                        if tuple(src) == dest_tuple:
                            is_moving_source = True
                            break

                    if not is_moving_source:
                        return False, "cannot move onto your own stationary marble"

                elif dest_tuple in opponent_marbles_set:
                    if not is_line_push:
                        return False, "cannot move onto an opponent marble unless performing a valid line push"

                    front_marble = sorted_source[0]
                    front_pos = tuple(front_marble)

                    push_pos = (front_marble[0] + direction_vector[0], front_marble[1] + direction_vector[1])

                    if push_pos != dest_tuple:
                        return False, "push opponent marbles with the front marble of your line"

                    opponent_line = [push_pos]
                    next_pos = push_pos

                    while True:
                        next_pos = (next_pos[0] + direction_vector[0], next_pos[1] + direction_vector[1])
                        if next_pos in opponent_marbles_set:
                            opponent_line.append(next_pos)
                        else:
                            break

                    if len(opponent_line) >= num_marbles:
                        return False, "cannot push unless you have more marbles than opponent"

                    last_opponent = opponent_line[-1]
                    push_dest = (last_opponent[0] + direction_vector[0], last_opponent[1] + direction_vector[1])

                    if push_dest not in VALID_COORDS:
                        pass
                    elif push_dest in black_marbles_set or push_dest in white_marbles_set:
                        return False, "cannot push marble into another marble"

        for i, dest in enumerate(dest_coords):
            dest_tuple = tuple(dest)
            source_found = False

            for src in source_coords:
                expected_dest = (src[0] + direction_vector[0], src[1] + direction_vector[1])
                if tuple(expected_dest) == dest_tuple:
                    source_found = True
                    break

            if not source_found and dest_tuple not in opponent_marbles_set:
                return False, "invalid destination for marble movement"

    return True, "it works"

cpdef tuple move_marbles(list source_coords, list dest_coords, list board, str current_player_color):
    cdef bint is_valid
    cdef str reason
    cdef list new_board, black_marbles, white_marbles, player_marbles, opponent_marbles
    cdef list source_tuples, front_marble, push_coord, opponent_line, marble
    cdef set black_marbles_set, white_marbles_set, player_marbles_set, opponent_marbles_set
    cdef tuple diff, direction_vector, vector, push_tuple, next_coord, new_pos, marble_tuple, src_tuple
    cdef int marbles_pushed_off, front_marble_idx, i, j
    cdef bint is_front

    is_valid, reason = move_validation(source_coords, dest_coords, board, current_player_color)
    if not is_valid:
        return None, 0

    new_board = [[], []]
    new_board[0] = [coord.copy() for coord in board[0]]
    new_board[1] = [coord.copy() for coord in board[1]]

    black_marbles = new_board[0]
    white_marbles = new_board[1]

    black_marbles_set = {tuple(marble) for marble in black_marbles}
    white_marbles_set = {tuple(marble) for marble in white_marbles}

    if current_player_color == "BLACK":
        player_marbles = black_marbles
        opponent_marbles = white_marbles
        player_marbles_set = black_marbles_set
        opponent_marbles_set = white_marbles_set
    else:
        player_marbles = white_marbles
        opponent_marbles = black_marbles
        player_marbles_set = white_marbles_set
        opponent_marbles_set = black_marbles_set

    diff = (dest_coords[0][0] - source_coords[0][0], dest_coords[0][1] - source_coords[0][1])
    direction_vector = None

    for _, vector in DIRECTION_VECTORS.items():
        if vector == diff:
            direction_vector = vector
            break

    marbles_pushed_off = 0
    front_marble_idx = -1

    for i, src in enumerate(source_coords):
        is_front = True
        for other_src in source_coords:
            if src != other_src:
                next_pos = [src[0] + direction_vector[0], src[1] + direction_vector[1]]
                if next_pos == other_src:
                    is_front = False
                    break
        if is_front:
            front_marble_idx = i
            break

    if front_marble_idx != -1:
        front_marble = source_coords[front_marble_idx]
        push_coord = [front_marble[0] + direction_vector[0], front_marble[1] + direction_vector[1]]
        push_tuple = tuple(push_coord)

        if push_tuple in opponent_marbles_set:
            opponent_line = [push_tuple]
            next_coord = push_tuple

            while True:
                next_pos = (next_coord[0] + direction_vector[0], next_coord[1] + direction_vector[1])
                if next_pos in opponent_marbles_set:
                    opponent_line.append(next_pos)
                    next_coord = next_pos
                else:
                    break

            for marble_tuple in opponent_line:
                new_pos = (marble_tuple[0] + direction_vector[0], marble_tuple[1] + direction_vector[1])

                for i, marble in enumerate(opponent_marbles):
                    if tuple(marble) == marble_tuple:
                        opponent_marbles.pop(i)
                        break

                if new_pos in VALID_COORDS:
                    opponent_marbles.append(list(new_pos))
                else:
                    marbles_pushed_off += 1

    source_tuples = [tuple(src) for src in source_coords]

    for i, src_tuple in enumerate(source_tuples):
        for j, marble in enumerate(player_marbles):
            if tuple(marble) == src_tuple:
                player_marbles.pop(j)
                player_marbles.append(dest_coords[i])
                break

    black_marbles.sort()
    white_marbles.sort()

    return new_board, marbles_pushed_off

cpdef dict convert_to_dictionary(list board_list, str no_marble_value="Blank", str black_marble_value="#8A8A8A",
                                str white_marble_value="#D9D9D9"):
    cdef list valid_coords_list = [
        [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
        [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9],
        [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9],
        [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9],
        [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],
        [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
        [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
        [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
        [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]
    ]

    cdef list black_marbles = board_list[0]
    cdef list white_marbles = board_list[1]

    cdef set black_set = {tuple(marble) for marble in black_marbles}
    cdef set white_set = {tuple(marble) for marble in white_marbles}

    cdef dict board_dict = {}
    cdef dict letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}
    cdef str key
    cdef tuple pos
    cdef int row, col

    for row, col in valid_coords_list:
        key = f"{letter_map[row]}{col}"

        pos = (row, col)
        if pos in black_set:
            board_dict[key] = black_marble_value
        elif pos in white_set:
            board_dict[key] = white_marble_value
        else:
            board_dict[key] = no_marble_value

    return board_dict