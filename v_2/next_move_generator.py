import itertools
import time
import numpy as np
from numba import njit, jit
import functools
from collections import defaultdict

# Import optimized functions from move.py
from move import move_validation, move_marbles, DIRECTION_VECTORS, VALID_COORDS


def memoize(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


def read_board_from_text(filename):

    with open(filename, 'r') as file:
        lines = file.readlines()

    current_player = "BLACK" if lines[0].strip().lower() == "b" else "WHITE"

    if len(lines) < 2:
        return current_player, [[], []]

    board_str = lines[1].strip()
    marbles = board_str.split(',')

    black_marbles = []
    white_marbles = []

    for marble_str in marbles:
        if len(marble_str) < 3:
            continue

        letter = marble_str[0].upper()

        num_end = 1
        while num_end < len(marble_str) and marble_str[num_end].isdigit():
            num_end += 1

        if num_end == 1:
            continue

        row = ord(letter) - ord('A') + 1
        col = int(marble_str[1:num_end])

        color = marble_str[-1].lower()

        if color == 'b':
            black_marbles.append([row, col])
        elif color == 'w':
            white_marbles.append([row, col])

    black_marbles.sort()
    white_marbles.sort()

    return current_player, [black_marbles, white_marbles]

@memoize
def check_marbles(marble_tuple):

    if len(marble_tuple) == 1:
        return True

    elif len(marble_tuple) == 2:

        marble1, marble2 = marble_tuple
        diff = (marble2[0] - marble1[0], marble2[1] - marble1[1])

        for vector in DIRECTION_VECTORS.values():
            if diff == vector or diff == (-vector[0], -vector[1]):
                return True
        return False

    elif len(marble_tuple) == 3:
        sorted_marbles = sorted(marble_tuple, key=lambda x: (x[0], x[1]))

        diff1 = (sorted_marbles[1][0] - sorted_marbles[0][0], sorted_marbles[1][1] - sorted_marbles[0][1])
        diff2 = (sorted_marbles[2][0] - sorted_marbles[1][0], sorted_marbles[2][1] - sorted_marbles[1][1])

        return diff1 == diff2 and diff1 != (0, 0)

    else:
        return False


def find_all_groups_of_size_1_2_3(board, color):

    all_positions = board[0] if color == "BLACK" else board[1]

    all_positions_tuples = [tuple(pos) for pos in all_positions]

    adjacency_map = defaultdict(list)
    for i, pos1 in enumerate(all_positions_tuples):
        for j, pos2 in enumerate(all_positions_tuples):
            if i != j:
                diff = (pos2[0] - pos1[0], pos2[1] - pos1[1])
                for vector in DIRECTION_VECTORS.values():
                    if diff == vector or diff == (-vector[0], -vector[1]):
                        adjacency_map[pos1].append(pos2)
                        break

    groups = []
    for pos in all_positions:
        groups.append([pos])

    for pos1 in all_positions_tuples:
        for pos2 in adjacency_map[pos1]:
            if pos1 < pos2:
                groups.append([list(pos1), list(pos2)])

    for pos1 in all_positions_tuples:
        for pos2 in adjacency_map[pos1]:
            for pos3 in adjacency_map[pos2]:
                if pos1 < pos2 < pos3:
                    marble_tuple = (pos1, pos2, pos3)
                    if check_marbles(marble_tuple):
                        groups.append([list(pos1), list(pos2), list(pos3)])

    return groups


def generate_all_directions(marble_list):
    result = {}

    for direction, vector in DIRECTION_VECTORS.items():

        new_positions = []

        for marble in marble_list:

            new_pos = [marble[0] + vector[0], marble[1] + vector[1]]
            new_positions.append(new_pos)

        result[direction] = new_positions

    return result


def generate_all_next_moves(board, color):

    marble_groups = find_all_groups_of_size_1_2_3(board, color)

    all_sources = []
    all_directions = []

    for source in marble_groups:
        directions = generate_all_directions(source)
        for direction_name, dest in directions.items():
            all_sources.append(source)
            all_directions.append(dest)

    result = []

    for i in range(len(all_sources)):
        source = all_sources[i]
        dest = all_directions[i]

        if any(tuple(d) not in VALID_COORDS for d in dest):
            continue

        is_valid, _ = move_validation(source, dest, board, color)

        if is_valid:
            new_board, _ = move_marbles(source, dest, board, color)
            if new_board is not None:
                result.append(new_board)

    return result


def save_board_states_to_file(board_states, filename, next_player_color):

    if next_player_color == "BLACK":
        black_marker = "b"
        white_marker = "w"
    else:
        black_marker = "w"
        white_marker = "b"

    letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}

    with open(filename, 'w') as file:
        for board in board_states:
            black_marbles = board[0]
            white_marbles = board[1]

            black_strings = []
            for row, col in black_marbles:
                letter = letter_map[row]
                black_strings.append(f"{letter}{col}{black_marker}")

            white_strings = []
            for row, col in white_marbles:
                letter = letter_map[row]
                white_strings.append(f"{letter}{col}{white_marker}")

            file.write(','.join(black_strings + white_strings) + '\n')


def format_coords_to_string(coords):

    letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}

    result = []
    for row, col in coords:
        letter = letter_map[row]
        result.append(f"{letter}{col}")

    return ''.join(result)


if __name__ == "__main__":
    a = time.time()
    sample_board = [
        [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [3, 3], [3, 4],
         [3, 5]],
        [[7, 5], [7, 6], [7, 7], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [9, 5], [9, 6], [9, 7], [9, 8],
         [9, 9]]]

    input_file = "./output/Test2.input"
    current_player, board_list = read_board_from_text(input_file)

    next_boards = generate_all_next_moves(board_list, current_player)
    # for next_board in next_boards:
    #     print(next_board)

    save_board_states_to_file(next_boards, "./output/team3_test.board", current_player)
    b = time.time()
    print(b - a)