# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=True
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True

import time
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset
from libc.stdint cimport uint64_t, uint32_t, int32_t, int8_t, uint8_t
import cython

from move_cy import move_validation, move_marbles, DIRECTION_VECTORS, VALID_COORDS, is_valid_coord

DEF MAX_MARBLES = 14  
DEF MAX_BOARD_SIZE = 10  
DEF MAX_GROUPS = 100  
DEF NUM_DIRECTIONS = 6  
DEF MAX_MOVES = 500  
DEF VALID_COORDS_COUNT = 61  

cdef struct Marble:
    int row
    int col

cdef struct MarbleGroup:
    Marble marbles[3]
    int size
    int direction

cdef struct Move:
    Marble source[3]
    Marble dest[3]
    int size
    int direction


cdef struct GroupCache:
    int initialized
    int count
    MarbleGroup groups[MAX_GROUPS]


cdef int DIRECTIONS[NUM_DIRECTIONS][2]
cdef void init_directions():
    
    DIRECTIONS[0][0] = 1
    DIRECTIONS[0][1] = 0
    
    DIRECTIONS[1][0] = 1
    DIRECTIONS[1][1] = 1
    
    DIRECTIONS[2][0] = 0
    DIRECTIONS[2][1] = -1
    
    DIRECTIONS[3][0] = 0
    DIRECTIONS[3][1] = 1
    
    DIRECTIONS[4][0] = -1
    DIRECTIONS[4][1] = -1
    
    DIRECTIONS[5][0] = -1
    DIRECTIONS[5][1] = 0


cdef int VALID_COORDS_C[VALID_COORDS_COUNT][2]
cdef int VALID_COORDS_LOOKUP[MAX_BOARD_SIZE][MAX_BOARD_SIZE]
cdef void init_valid_coords():
    cdef int row, col
    cdef tuple coord
    cdef int idx = 0

    for row in range(MAX_BOARD_SIZE):
        for col in range(MAX_BOARD_SIZE):
            VALID_COORDS_LOOKUP[row][col] = 0

    for coord in VALID_COORDS:
        VALID_COORDS_C[idx][0] = coord[0]
        VALID_COORDS_C[idx][1] = coord[1]
        
        VALID_COORDS_LOOKUP[coord[0]][coord[1]] = 1
        idx += 1

cdef struct Board:
    int black_count
    int white_count
    Marble black_marbles[MAX_MARBLES]
    Marble white_marbles[MAX_MARBLES]
    int board_array[MAX_BOARD_SIZE][MAX_BOARD_SIZE]  


init_directions()
init_valid_coords()

cdef GroupCache BLACK_GROUP_CACHE
cdef GroupCache WHITE_GROUP_CACHE
BLACK_GROUP_CACHE.initialized = 0
WHITE_GROUP_CACHE.initialized = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_valid_coord_c(int row, int col):
    if 0 <= row < MAX_BOARD_SIZE and 0 <= col < MAX_BOARD_SIZE:
        return VALID_COORDS_LOOKUP[row][col] == 1
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Board create_c_board(list py_board):
    cdef Board c_board
    cdef int i, row, col

    
    for row in range(MAX_BOARD_SIZE):
        for col in range(MAX_BOARD_SIZE):
            c_board.board_array[row][col] = 0

    
    c_board.black_count = len(py_board[0])
    for i in range(c_board.black_count):
        c_board.black_marbles[i].row = py_board[0][i][0]
        c_board.black_marbles[i].col = py_board[0][i][1]
        c_board.board_array[c_board.black_marbles[i].row][c_board.black_marbles[i].col] = 1

    
    c_board.white_count = len(py_board[1])
    for i in range(c_board.white_count):
        c_board.white_marbles[i].row = py_board[1][i][0]
        c_board.white_marbles[i].col = py_board[1][i][1]
        c_board.board_array[c_board.white_marbles[i].row][c_board.white_marbles[i].col] = 2

    return c_board

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint are_marbles_adjacent(Marble m1, Marble m2):
    cdef int dr = m2.row - m1.row
    cdef int dc = m2.col - m1.col
    cdef int i

    for i in range(NUM_DIRECTIONS):
        if (dr == DIRECTIONS[i][0] and dc == DIRECTIONS[i][1]) or \
                (dr == -DIRECTIONS[i][0] and dc == -DIRECTIONS[i][1]):
            return True

    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int get_direction(Marble m1, Marble m2):
    cdef int dr = m2.row - m1.row
    cdef int dc = m2.col - m1.col
    cdef int i

    for i in range(NUM_DIRECTIONS):
        if dr == DIRECTIONS[i][0] and dc == DIRECTIONS[i][1]:
            return i

    return -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint are_three_in_line(Marble m1, Marble m2, Marble m3):
    cdef int dir1 = get_direction(m1, m2)
    cdef int dir2 = get_direction(m2, m3)

    if dir1 == -1 or dir2 == -1:
        return False

    return dir1 == dir2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_groups_c(Marble * marbles, int count, MarbleGroup * groups, int * group_count):
    cdef int i, j, k, dir_i, dir1, dir2
    cdef bint is_adj

    group_count[0] = 0

    for i in range(count):
        if group_count[0] < MAX_GROUPS:
            groups[group_count[0]].marbles[0].row = marbles[i].row
            groups[group_count[0]].marbles[0].col = marbles[i].col
            groups[group_count[0]].size = 1
            groups[group_count[0]].direction = -1
            group_count[0] += 1

    for i in range(count):
        for j in range(i + 1, count):
            is_adj = are_marbles_adjacent(marbles[i], marbles[j])

            if is_adj and group_count[0] < MAX_GROUPS:
                dir_i = get_direction(marbles[i], marbles[j])
                if dir_i == -1:
                    dir_i = get_direction(marbles[j], marbles[i])

                groups[group_count[0]].marbles[0].row = marbles[i].row
                groups[group_count[0]].marbles[0].col = marbles[i].col
                groups[group_count[0]].marbles[1].row = marbles[j].row
                groups[group_count[0]].marbles[1].col = marbles[j].col
                groups[group_count[0]].size = 2
                groups[group_count[0]].direction = dir_i
                group_count[0] += 1

    for i in range(count):
        for j in range(i + 1, count):
            if not are_marbles_adjacent(marbles[i], marbles[j]):
                continue

            dir1 = get_direction(marbles[i], marbles[j])
            if dir1 == -1:
                dir1 = get_direction(marbles[j], marbles[i])
                if dir1 == -1:
                    continue

            for k in range(j + 1, count):
                if not are_marbles_adjacent(marbles[j], marbles[k]):
                    continue

                dir2 = get_direction(marbles[j], marbles[k])
                if dir2 == -1:
                    continue

                if dir1 == dir2 and group_count[0] < MAX_GROUPS:
                    groups[group_count[0]].marbles[0].row = marbles[i].row
                    groups[group_count[0]].marbles[0].col = marbles[i].col
                    groups[group_count[0]].marbles[1].row = marbles[j].row
                    groups[group_count[0]].marbles[1].col = marbles[j].col
                    groups[group_count[0]].marbles[2].row = marbles[k].row
                    groups[group_count[0]].marbles[2].col = marbles[k].col
                    groups[group_count[0]].size = 3
                    groups[group_count[0]].direction = dir1
                    group_count[0] += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef MarbleGroup * get_groups(Board * board, str color, int * group_count):
    cdef GroupCache * cache
    cdef Marble * marbles
    cdef int marble_count

    if color == "BLACK":
        cache = &BLACK_GROUP_CACHE
        marbles = board.black_marbles
        marble_count = board.black_count
    else:
        cache = &WHITE_GROUP_CACHE
        marbles = board.white_marbles
        marble_count = board.white_count

    if not cache.initialized:
        cache.initialized = 1
        cache.count = 0
        find_groups_c(marbles, marble_count, cache.groups, &cache.count)

    group_count[0] = cache.count
    return cache.groups

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint generate_move_in_direction(MarbleGroup group, int dir_idx, Move * move):
    cdef int i
    cdef int drow = DIRECTIONS[dir_idx][0]
    cdef int dcol = DIRECTIONS[dir_idx][1]

    for i in range(group.size):
        move.source[i].row = group.marbles[i].row
        move.source[i].col = group.marbles[i].col

        move.dest[i].row = group.marbles[i].row + drow
        move.dest[i].col = group.marbles[i].col + dcol

        if not is_valid_coord_c(move.dest[i].row, move.dest[i].col):
            return False

    move.size = group.size
    move.direction = dir_idx
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void move_to_python_lists(Move move, list source_coords, list dest_coords):
    cdef int i
    source_coords.clear()
    dest_coords.clear()

    for i in range(move.size):
        source_coords.append([move.source[i].row, move.source[i].col])
        dest_coords.append([move.dest[i].row, move.dest[i].col])

cpdef void reset_group_caches():
    global BLACK_GROUP_CACHE, WHITE_GROUP_CACHE
    BLACK_GROUP_CACHE.initialized = 0
    WHITE_GROUP_CACHE.initialized = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list move_marbles_fast(list source_coords, list dest_coords, list board, str current_player_color):
    cdef list new_board, black_marbles, white_marbles, player_marbles, opponent_marbles
    cdef list front_marble, push_coord, opponent_line, marble
    cdef set opponent_marbles_set
    cdef tuple diff, direction_vector, vector, push_tuple, next_coord, new_pos, marble_tuple
    cdef int marbles_pushed_off, front_marble_idx, i, j
    cdef bint is_front

    new_board = [[], []]
    new_board[0] = [coord.copy() for coord in board[0]]
    new_board[1] = [coord.copy() for coord in board[1]]

    black_marbles = new_board[0]
    white_marbles = new_board[1]

    if current_player_color == "BLACK":
        player_marbles = black_marbles
        opponent_marbles = white_marbles
    else:
        player_marbles = white_marbles
        opponent_marbles = black_marbles

    opponent_marbles_set = {tuple(marble) for marble in opponent_marbles}

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

    for i, src in enumerate(source_coords):
        for j, marble in enumerate(player_marbles):
            if marble[0] == src[0] and marble[1] == src[1]:
                player_marbles.pop(j)
                player_marbles.append(dest_coords[i])
                break

    black_marbles.sort()
    white_marbles.sort()

    return new_board

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_all_next_moves(list py_board, str color):
    cdef Board c_board = create_c_board(py_board)

    reset_group_caches()

    cdef int group_count = 0
    cdef MarbleGroup * groups = get_groups(&c_board, color, &group_count)

    cdef dict result_dict = {}

    cdef Move move

    cdef list source_coords = []
    cdef list dest_coords = []

    cdef int i, dir_idx, j
    cdef bint valid_move
    cdef bint valid_validation
    cdef str reason
    cdef list new_board
    cdef tuple source_tuple, dest_tuple
    cdef tuple result_tuple

    for i in range(group_count):
        for dir_idx in range(NUM_DIRECTIONS):
            valid_move = generate_move_in_direction(groups[i], dir_idx, &move)

            if valid_move:
                move_to_python_lists(move, source_coords, dest_coords)

                result_tuple = move_validation(source_coords, dest_coords, py_board, color)
                valid_validation = result_tuple[0]
                reason = result_tuple[1]

                if valid_validation:
                    new_board = move_marbles_fast(source_coords, dest_coords, py_board, color)

                    if new_board is not None:
                        source_tuple = tuple(tuple(pos) for pos in source_coords)
                        dest_tuple = tuple(tuple(pos) for pos in dest_coords)

                        result_dict[(source_tuple, dest_tuple)] = new_board

    return result_dict

@cython.boundscheck(False)
@cython.wraparound(False)
def find_all_groups_of_size_1_2_3(list py_board, str color):
    cdef Board c_board = create_c_board(py_board)

    reset_group_caches()

    cdef int group_count = 0
    cdef MarbleGroup * groups = get_groups(&c_board, color, &group_count)

    cdef list result = []
    cdef list group_marbles
    cdef int i, j

    for i in range(group_count):
        group_marbles = []
        for j in range(groups[i].size):
            group_marbles.append([groups[i].marbles[j].row, groups[i].marbles[j].col])
        result.append(group_marbles)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def format_coords_to_string(list coords):
    cdef dict letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}
    cdef list result = []
    cdef int i, n, row, col

    n = len(coords)
    for i in range(n):
        row = coords[i][0]
        col = coords[i][1]
        result.append(f"{letter_map[row]}{col}")

    return ''.join(result)

@cython.boundscheck(False)
@cython.wraparound(False)
def read_board_from_text(str filename):
    cdef str current_player
    cdef list black_marbles = []
    cdef list white_marbles = []
    cdef list lines, marbles
    cdef str board_str, marble_str, letter, color
    cdef int row, col, num_end, i

    with open(filename, 'r') as file:
        lines = file.readlines()

    current_player = "BLACK" if lines[0].strip().lower() == "b" else "WHITE"

    if len(lines) < 2:
        return current_player, [[], []]

    board_str = lines[1].strip()
    marbles = board_str.split(',')

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

@cython.boundscheck(False)
@cython.wraparound(False)
def save_board_states_to_file(object board_states, str filename, str next_player_color):
    cdef str black_marker = "b" if next_player_color == "BLACK" else "w"
    cdef str white_marker = "w" if next_player_color == "BLACK" else "b"
    cdef dict letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}
    cdef list black_strings, white_strings
    cdef list board_list
    cdef list black_marbles, white_marbles
    cdef int row, col

    with open(filename, 'w') as file:
        if isinstance(board_states, dict):
            board_list = list(board_states.values())
        else:
            board_list = board_states

        for board in board_list:
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