# cython: language_level=3
# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

import time
import numpy as np
cimport numpy as np
import cython
import math
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset
from libc.math cimport sqrt, exp
from collections import OrderedDict

ctypedef np.int64_t INT64_t
ctypedef np.float64_t FLOAT64_t
ctypedef np.uint8_t UINT8_t

DTYPE = np.int64
FLOAT_DTYPE = np.float64

cdef int DIRECTIONS_C[6][2]
DIRECTIONS_C= [
    [1, 0],
    [1, 1],
    [0, -1],
    [0, 1],
    [-1, -1],
    [-1, 0]
]

DIRECTIONS = np.array(DIRECTIONS_C, dtype=DTYPE)
cdef INT64_t[:, :] DIRECTIONS_VIEW = DIRECTIONS

WEIGHTS = {
    "marble_diff": 1.0,
    "centrality": 0.18,
    "push_ability": 0.35,
    "formation": 0.027,
    "connectivity": 1.35
}

VALID_COORDS = np.array([
    [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
    [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9],
    [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9],
    [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9],
    [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],
    [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
    [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
    [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
    [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]
], dtype=DTYPE)

cdef INT64_t[:, :] VALID_COORDS_VIEW = VALID_COORDS

VALID_COORDS_LOOKUP = np.zeros((10, 10), dtype=np.uint8)
for coord in VALID_COORDS:
    VALID_COORDS_LOOKUP[coord[0], coord[1]] = 1
cdef UINT8_t[:, :] VALID_COORDS_LOOKUP_VIEW = VALID_COORDS_LOOKUP

VALID_COORDS_SET = {tuple(coord) for coord in VALID_COORDS}

COORD_TO_INDEX_MAP = {}
for i in range(VALID_COORDS.shape[0]):
    COORD_TO_INDEX_MAP[tuple(VALID_COORDS[i])] = i

CENTER_COORD = np.array([5, 5], dtype=DTYPE)

CENTER_COORDS = np.array([(5, 5)], dtype=DTYPE)

RING1_COORDS = np.array([
    (6, 5), (6, 6), (5, 4), (5, 6), (4, 4), (4, 5)
], dtype=DTYPE)

RING2_COORDS = np.array([
    (7, 5), (7, 6), (7, 7), (6, 4), (6, 7), (5, 3),
    (5, 7), (4, 3), (4, 6), (3, 3), (3, 4), (3, 5)
], dtype=DTYPE)

RING3_COORDS = np.array([
    (8, 5), (8, 6), (8, 7), (8, 8), (7, 4), (7, 8),
    (6, 3), (6, 8), (5, 2), (5, 8), (4, 2), (4, 7),
    (3, 2), (3, 6), (2, 2), (2, 3), (2, 4), (2, 5)
], dtype=DTYPE)

RING4_COORDS = np.array([
    (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (8, 4),
    (8, 9), (7, 3), (7, 9), (6, 2), (6, 9), (5, 1),
    (5, 9), (4, 1), (4, 8), (3, 1), (3, 7), (2, 1),
    (2, 6), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)
], dtype=DTYPE)

cdef double CENTER_SCORE = 7.0
cdef double RING1_SCORE = 4.5
cdef double RING2_SCORE = 2.8
cdef double RING3_SCORE = 1.3
cdef double RING4_SCORE = -1.5

CENTRALITY_MAP = {}
for coord in CENTER_COORDS:
    CENTRALITY_MAP[tuple(coord)] = CENTER_SCORE
for coord in RING1_COORDS:
    CENTRALITY_MAP[tuple(coord)] = RING1_SCORE
for coord in RING2_COORDS:
    CENTRALITY_MAP[tuple(coord)] = RING2_SCORE
for coord in RING3_COORDS:
    CENTRALITY_MAP[tuple(coord)] = RING3_SCORE
for coord in RING4_COORDS:
    CENTRALITY_MAP[tuple(coord)] = RING4_SCORE

cdef int BOARD_SIZE = 10
FRIEND_MASK = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
cdef np.int8_t[:, :] FRIEND_MASK_VIEW = FRIEND_MASK
ENEMY_MASK = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
cdef np.int8_t[:, :] ENEMY_MASK_VIEW = ENEMY_MASK

cdef set outer_ring = {(int(coord[0]), int(coord[1])) for coord in RING4_COORDS}
cdef set middle_ring = {(int(coord[0]), int(coord[1])) for coord in RING3_COORDS}
cdef set inner_ring = {(int(coord[0]), int(coord[1])) for coord in RING2_COORDS}

NEIGHBOR_CACHE = {}
for i in range(VALID_COORDS.shape[0]):
    coord = tuple(VALID_COORDS[i])
    neighbors = []
    for j in range(DIRECTIONS.shape[0]):
        neighbor = (VALID_COORDS[i, 0] + DIRECTIONS[j, 0], VALID_COORDS[i, 1] + DIRECTIONS[j, 1])
        if neighbor in VALID_COORDS_SET:
            neighbors.append(neighbor)
    NEIGHBOR_CACHE[coord] = neighbors

ZOBRIST_TABLE = np.random.randint(
    0, 2 ** 64 - 1, size=(2, VALID_COORDS.shape[0]), dtype=np.uint64
)

MAX_TABLE_SIZE = 200000
transposition_table = OrderedDict()
history_table = {}
killer_moves = {}

MAX_CACHE_SIZE = 5000
group_cache = OrderedDict()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void manage_cache_size(object cache, int max_size):
    """Manage cache size by removing oldest entries if needed"""
    if len(cache) > max_size:
        remove_count = max_size // 5
        for _ in range(remove_count):
            if cache:
                cache.popitem(last=False)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_valid_coord(tuple coord):
    """Fast check if a coordinate is valid"""
    cdef int row = coord[0]
    cdef int col = coord[1]
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return VALID_COORDS_LOOKUP_VIEW[row, col] == 1
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline list get_neighbors(tuple coord):
    """Get all valid neighboring coordinates"""
    return NEIGHBOR_CACHE.get(coord, [])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int coord_to_index(tuple coord):
    """Convert coordinate to index for Zobrist hashing"""
    return COORD_TO_INDEX_MAP.get(coord, -1)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_in_array(tuple coord, list arr):
    """Check if coordinate is in an array, optimized for different array sizes"""
    cdef int i
    cdef tuple item_tuple

    if len(arr) > 10:
        arr_set = {tuple(pos) for pos in arr}
        return coord in arr_set

    for i in range(len(arr)):
        item_tuple = tuple(arr[i]) if not isinstance(arr[i], tuple) else arr[i]
        if item_tuple[0] == coord[0] and item_tuple[1] == coord[1]:
            return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.uint64_t compute_zobrist_hash(list board):
    """Compute Zobrist hash for board position"""
    cdef np.uint64_t hash_value = 0
    cdef int idx
    cdef tuple marble_tuple

    for marble in board[0]:
        marble_tuple = tuple(marble)
        idx = coord_to_index(marble_tuple)
        if idx >= 0:
            hash_value ^= ZOBRIST_TABLE[0, idx]

    for marble in board[1]:
        marble_tuple = tuple(marble)
        idx = coord_to_index(marble_tuple)
        if idx >= 0:
            hash_value ^= ZOBRIST_TABLE[1, idx]

    return hash_value

@cython.boundscheck(False)
@cython.wraparound(False)
def find_groups_fast(marbles):
    """
    Find all possible groups of marbles (singles, pairs, triplets in line)
    Optimized with caching for frequent marble configurations
    """
    if not marbles:
        return []

    cdef list marbles_tuple = [tuple(m) for m in marbles]
    cdef tuple cache_key = tuple(sorted(marbles_tuple))

    if cache_key in group_cache:
        value = group_cache.pop(cache_key)
        group_cache[cache_key] = value
        return value

    cdef list result = []
    cdef int i, j, max_process, max_groups
    cdef tuple marble1, neighbor, ext
    cdef list group
    cdef tuple diff

    for i in range(len(marbles)):
        result.append([marbles[i]])

    if len(marbles) < 2:
        group_cache[cache_key] = result
        manage_cache_size(group_cache, MAX_CACHE_SIZE)
        return result

    cdef set marbles_set = set(marbles_tuple)

    max_process = min(len(marbles), 50)

    for i in range(max_process):
        marble1 = marbles_tuple[i]
        for neighbor in get_neighbors(marble1):
            if neighbor in marbles_set:
                result.append([list(marble1), list(neighbor)])

    max_groups = min(len(result), 50)

    for i in range(min(max_groups, len(result))):
        group = result[i]
        if len(group) == 2:
            diff = (group[1][0] - group[0][0], group[1][1] - group[0][1])
            ext = (group[1][0] + diff[0], group[1][1] + diff[1])
            if ext in marbles_set:
                result.append([group[0], group[1], list(ext)])

    group_cache[cache_key] = result
    manage_cache_size(group_cache, MAX_CACHE_SIZE)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_connectivity(list positions, bint friend_side):
    """
    Evaluate connectivity of marbles - how well they support each other
    Uses pre-filled mask arrays for fast lookups
    """
    cdef double conn_score = 0.0
    cdef int connection_count, row, col, nr, nc
    cdef int dr, dc, i

    if len(positions) < 3:
        return 0.0

    for position in positions:
        row = position[0]
        col = position[1]
        connection_count = 0

        for i in range(DIRECTIONS_VIEW.shape[0]):
            dr = <int> DIRECTIONS_VIEW[i, 0]
            dc = <int> DIRECTIONS_VIEW[i, 1]
            nr = row + dr
            nc = col + dc

            if 0 <= nr < 10 and 0 <= nc < 10:

                if friend_side:
                    if FRIEND_MASK_VIEW[nr, nc] == 1:
                        connection_count += 1
                else:
                    if ENEMY_MASK_VIEW[nr, nc] == 1:
                        connection_count += 1

        if connection_count == 0:
            conn_score -= 7.0
        elif connection_count == 1:
            conn_score -= 2.3

    return conn_score

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_hexagon_formation(list positions):
    """
    Evaluate formation of marbles - rewards hexagon and other strong formations
    """
    cdef double hexagon_score = 0.0
    cdef int neighbor_count, total_neighbors = 0, positions_with_neighbors = 0
    cdef int row, col, nr, nc, dr, dc, i, dir1_idx, dir2_idx
    cdef double avg_connections, rectangle_score = 0.0
    cdef int dr2
    cdef int dc2
    cdef int r3, c3
    cdef int r4, c4

    for (row, col) in [(pos[0], pos[1]) for pos in positions]:
        neighbor_count = 0
        for i in range(DIRECTIONS_VIEW.shape[0]):
            dr = <int> DIRECTIONS_VIEW[i, 0];
            dc = <int> DIRECTIONS_VIEW[i, 1]
            nr = row + dr;
            nc = col + dc
            if 0 <= nr < 10 and 0 <= nc < 10 and FRIEND_MASK_VIEW[nr, nc] == 1:
                neighbor_count += 1

        total_neighbors += neighbor_count
        if neighbor_count > 0:
            positions_with_neighbors += 1

        if neighbor_count == 6:
            hexagon_score += 1.5
        elif neighbor_count >= 4:
            hexagon_score += neighbor_count * 0.2

    avg_connections = 0.0
    if positions_with_neighbors > 0:
        avg_connections = total_neighbors / float(positions_with_neighbors)

    if avg_connections > 4.0:
        hexagon_score -= (avg_connections - 4.0) * 2.0

    for (row, col) in [(p[0], p[1]) for p in positions]:
        for dir1_idx in range(DIRECTIONS_VIEW.shape[0]):
            dr = <int> DIRECTIONS_VIEW[dir1_idx, 0];
            dc = <int> DIRECTIONS_VIEW[dir1_idx, 1]
            nr = row + dr;
            nc = col + dc

            if 0 <= nr < 10 and 0 <= nc < 10 and FRIEND_MASK_VIEW[nr, nc] == 1:

                dir2_idx = (dir1_idx + 2) % 6
                dr2 = <int> DIRECTIONS_VIEW[dir2_idx, 0]
                dc2 = <int> DIRECTIONS_VIEW[dir2_idx, 1]
                r3 = row + dr2
                c3 = col + dc2
                r4 = nr + dr2
                c4 = nc + dc2

                if 0 <= r3 < 10 and 0 <= c3 < 10 and 0 <= r4 < 10 and 0 <= c4 < 10:
                    if FRIEND_MASK_VIEW[r3, c3] == 1 and FRIEND_MASK_VIEW[r4, c4] == 1:
                        rectangle_score += 1.1

    rectangle_score = min(rectangle_score, 6.0)
    return hexagon_score + rectangle_score

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_push_ability_strength(list groups):
    """
    Evaluate the push potential of marble groups
    """
    cdef double strength = 0.0
    cdef int push_count
    cdef int r1, c1, r2, c2, r3, c3, push_r, push_c, final_r, final_c, next_r, next_c
    cdef list group, pos1, pos2, pos3

    if len(groups) <= 1:
        return 0.0

    cdef list three_groups = []
    for group in groups:
        if len(group) == 3:
            pos1, pos2, pos3 = group
            r1, c1 = pos1[0], pos1[1]
            r2, c2 = pos2[0], pos2[1]
            r3, c3 = pos3[0], pos3[1]

            if (r2 - r1) == (r3 - r2) and (c2 - c1) == (c3 - c2):
                three_groups.append(group)
                if len(three_groups) >= 15:
                    break

    for group in three_groups:
        pos1, pos2, pos3 = group
        r1, c1 = pos1[0], pos1[1]
        r2, c2 = pos2[0], pos2[1]
        r3, c3 = pos3[0], pos3[1]

        push_r = r3 + (r2 - r1)
        push_c = c3 + (c2 - c1)

        if 0 <= push_r < BOARD_SIZE and 0 <= push_c < BOARD_SIZE and ENEMY_MASK_VIEW[push_r, push_c] == 1:
            push_count = 1
            next_r = push_r + (r2 - r1)
            next_c = push_c + (c2 - c1)

            if 0 <= next_r < BOARD_SIZE and 0 <= next_c < BOARD_SIZE and ENEMY_MASK_VIEW[next_r, next_c] == 1:
                push_count += 1

            final_r = push_r + (r2 - r1) * push_count
            final_c = push_c + (c2 - c1) * push_count

            if not is_valid_coord((final_r, final_c)):
                strength += 30.0 * push_count
            elif FRIEND_MASK_VIEW[final_r, final_c] == 0 and ENEMY_MASK_VIEW[final_r, final_c] == 0:

                for m in range(push_count):
                    next_r = push_r + (r2 - r1) * m
                    next_c = push_c + (c2 - c1) * m
                    if (next_r, next_c) in outer_ring:
                        strength += 10.0
                    elif (next_r, next_c) in middle_ring:
                        strength += 6.0
                    elif (next_r, next_c) in inner_ring:
                        strength += 3.0

    cdef list two_groups = []
    for group in groups:
        if len(group) == 2:
            two_groups.append(group)
            if len(two_groups) >= 20:
                break

    for group in two_groups:
        pos1, pos2 = group
        r1, c1 = pos1[0], pos1[1]
        r2, c2 = pos2[0], pos2[1]
        push_r = r2 + (r2 - r1)
        push_c = c2 + (c2 - c1)

        if 0 <= push_r < BOARD_SIZE and 0 <= push_c < BOARD_SIZE and ENEMY_MASK_VIEW[push_r, push_c] == 1:
            final_r = push_r + (r2 - r1)
            final_c = push_c + (c2 - c1)

            if not is_valid_coord((final_r, final_c)):
                strength += 20.0
            elif FRIEND_MASK_VIEW[final_r, final_c] == 0 and ENEMY_MASK_VIEW[final_r, final_c] == 0:

                if (push_r, push_c) in outer_ring:
                    strength += 10.0
                elif (push_r, push_c) in middle_ring:
                    strength += 6.0
                elif (push_r, push_c) in inner_ring:
                    strength += 3.0

    return strength

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double calculate_centrality(list friend_positions, list enemy_positions, set friend_set=None, set enemy_set=None):
    """
    Calculate centrality score for both players
    """
    cdef double friend_centrality = 0.0
    cdef double enemy_centrality = 0.0
    cdef tuple pos
    cdef double score
    cdef double scale_factor = 0.075

    for pos in friend_positions:
        score = CENTRALITY_MAP.get(pos, 1.0) * scale_factor
        friend_centrality += score

    for pos in enemy_positions:
        score = CENTRALITY_MAP.get(pos, 1.0) * scale_factor
        enemy_centrality += score

    cdef int friend_marbles_left = len(friend_positions)
    cdef int enemy_marbles_left = len(enemy_positions)
    cdef double friend_weight = 1.0
    cdef double enemy_weight = 1.0

    return friend_centrality - enemy_centrality

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_marble_difference(int friend_count, int enemy_count):
    """
    Evaluate the difference in marble count between players
    """
    cdef double marble_diff_score = 0.0
    cdef int friend_off, enemy_off
    cdef double enemy_off_score, friend_off_score

    enemy_off = 14 - enemy_count
    enemy_off_score = 600.0 * enemy_off 

    friend_off = 14 - friend_count
    friend_off_score = -650.0 * friend_off


    if friend_count <= 8:
        return -10000.0
    if enemy_count <= 8:
        return 10000.0

    marble_diff_score += enemy_off_score + friend_off_score
    return marble_diff_score

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_board(list board, str player):
    """
    Complete board evaluation function
    Combines multiple evaluation factors with weights
    """
    cdef int friend_idx = 0 if player.lower() == "black" else 1
    cdef int enemy_idx = 1 if player.lower() == "black" else 0

    cdef list friend_marbles = board[friend_idx]
    cdef list enemy_marbles = board[enemy_idx]

    cdef int friend_count = len(friend_marbles)
    cdef int enemy_count = len(enemy_marbles)

    cdef double marble_diff_score, centrality_score, push_ability_score
    cdef double hexagon_score, connectivity_score, total_score

    cdef int i, j
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            FRIEND_MASK_VIEW[i, j] = 0
            ENEMY_MASK_VIEW[i, j] = 0

    for i in range(len(friend_marbles)):
        FRIEND_MASK_VIEW[friend_marbles[i][0], friend_marbles[i][1]] = 1
    for i in range(len(enemy_marbles)):
        ENEMY_MASK_VIEW[enemy_marbles[i][0], enemy_marbles[i][1]] = 1

    cdef list friend_positions = [tuple(pos) for pos in friend_marbles]
    cdef list enemy_positions = [tuple(pos) for pos in enemy_marbles]

    centrality_score = calculate_centrality(friend_positions, enemy_positions)

    cdef list friend_groups = find_groups_fast(friend_marbles)
    cdef list enemy_groups = find_groups_fast(enemy_marbles)

    push_ability_score = evaluate_push_ability_strength(friend_groups)
    hexagon_score = evaluate_hexagon_formation(friend_positions) * 0.75
    connectivity_score = evaluate_connectivity(friend_positions, True) - evaluate_connectivity(enemy_positions,
                                                                                               False) * 0.9
    marble_diff_score = evaluate_marble_difference(friend_count, enemy_count)

    total_score = 0.0
    total_score += WEIGHTS.get("marble_diff", 1.0) * marble_diff_score
    total_score += WEIGHTS.get("centrality", 1.0) * centrality_score
    total_score += WEIGHTS.get("push_ability", 1.0) * push_ability_score
    total_score += WEIGHTS.get("formation", 1.0) * hexagon_score
    total_score += WEIGHTS.get("connectivity", 1.0) * connectivity_score

    return total_score

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple alpha_beta_with_time_check(list board, int depth, double alpha, double beta, str player,
                                      str maximizing_player,
                                      object move_generator, double time_start, double time_limit):
    """
    Alpha-beta search algorithm with time limit checking
    """
    cdef double current_time
    cdef np.uint64_t board_hash
    cdef str tt_key
    cdef tuple result, value
    cdef double eval_score, best_score, score
    cdef dict next_boards_dict
    cdef list next_boards, scores
    cdef str next_player, current_color
    cdef object best_move = None
    cdef np.uint64_t move_hash
    cdef str move_hash_str
    cdef int i
    cdef tuple key_tuple

    if depth % 2 == 0:
        current_time = time.time()
        if current_time - time_start > time_limit:
            raise TimeoutError("Search time limit exceeded")

    board_hash = compute_zobrist_hash(board)
    tt_key = f"{board_hash}:{player}:{depth}"

    if tt_key in transposition_table:
        value = transposition_table.pop(tt_key)
        transposition_table[tt_key] = value
        return value

    if depth <= 0:
        eval_score = evaluate_board(board, maximizing_player)
        result = (eval_score, None)
        transposition_table[tt_key] = result
        manage_cache_size(transposition_table, MAX_TABLE_SIZE)
        return result

    current_color = "BLACK" if player.lower() == "black" else "WHITE"
    next_boards_dict = move_generator(board, current_color)

    if not next_boards_dict:

        eval_score = evaluate_board(board, maximizing_player)
        result = (eval_score, None)
        transposition_table[tt_key] = result
        manage_cache_size(transposition_table, MAX_TABLE_SIZE)
        return result

    next_boards = list(next_boards_dict.values())

    scores = [(board, evaluate_board(board, player)) for board in next_boards]
    is_maximizer = player.lower() == maximizing_player.lower()

    if is_maximizer:
        scores.sort(key=lambda x: x[1], reverse=True)
    else:
        scores.sort(key=lambda x: x[1])

    next_boards = [board for board, _ in scores]
    next_player = "White" if player.lower() == "black" else "Black"

    board_to_key = {id(board): key for key, board in next_boards_dict.items()}

    if is_maximizer:
        best_score = float('-inf')
        for i, move in enumerate(next_boards):
            try:
                score, _ = alpha_beta_with_time_check(
                    move, depth - 1, alpha, beta, next_player, maximizing_player,
                    move_generator, time_start, time_limit
                )

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)

                if beta <= alpha:

                    move_hash = compute_zobrist_hash(move)
                    move_hash_str = str(move_hash)
                    if depth not in killer_moves:
                        killer_moves[depth] = {}
                    killer_moves[depth][move_hash_str] = killer_moves[depth].get(move_hash_str, 0) + depth * depth
                    history_table[move_hash_str] = history_table.get(move_hash_str, 0) + depth * depth
                    break

            except TimeoutError:
                if best_move is None and next_boards:
                    best_move = next_boards[0]
                raise

    else:
        best_score = float('inf')
        for i, move in enumerate(next_boards):
            try:
                score, _ = alpha_beta_with_time_check(
                    move, depth - 1, alpha, beta, next_player, maximizing_player,
                    move_generator, time_start, time_limit
                )

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, best_score)

                if beta <= alpha:

                    move_hash = compute_zobrist_hash(move)
                    move_hash_str = str(move_hash)
                    if depth not in killer_moves:
                        killer_moves[depth] = {}
                    killer_moves[depth][move_hash_str] = killer_moves[depth].get(move_hash_str, 0) + depth * depth
                    history_table[move_hash_str] = history_table.get(move_hash_str, 0) + depth * depth
                    break

            except TimeoutError:
                if best_move is None and next_boards:
                    best_move = next_boards[0]
                raise

    result = (best_score, best_move)
    transposition_table[tt_key] = result
    manage_cache_size(transposition_table, MAX_TABLE_SIZE)
    return result

def get_move_string_from_key(key_tuple):
    """
    Convert a move key tuple to a readable string representation
    """
    source_tuple, dest_tuple = key_tuple
    source_str = ""
    dest_str = ""

    for src in source_tuple:
        row, col = src
        letter = chr(ord('A') + row - 1)
        source_str += f"{letter}{col}"

    for dst in dest_tuple:
        row, col = dst
        letter = chr(ord('A') + row - 1)
        dest_str += f"{letter}{col}"

    return f"{source_str},{dest_str}"

def find_best_move(list board, str player, int depth=4, double time_limit=5.0, object from_move_generator=None):
    """
    Find the best move using iterative deepening alpha-beta search
    Returns: (best board state, move string, empty features dict, total search time)
    """
    cdef int min_depth = 2
    cdef double start_time, current_time, elapsed, max_search_time, depth_start_time, remaining_time
    cdef double best_score, score
    cdef int current_depth
    cdef tuple result
    cdef list last_best_move = None
    cdef double last_best_score = 0.0
    cdef str color
    cdef dict next_boards_dict
    cdef tuple best_move_key = None
    cdef str move_str = ""

    if depth < min_depth:
        depth = min_depth

    if from_move_generator is None:
        try:
            from next_move_generator import generate_all_next_moves
            from_move_generator = generate_all_next_moves
        except ImportError:
            raise ImportError("Move generator not provided or not found.")

    start_time = time.time()
    max_search_time = time_limit * 0.95

    transposition_table.clear()
    killer_moves.clear()
    history_table.clear()

    for current_depth in range(min_depth, depth + 1):
        current_time = time.time()
        elapsed = current_time - start_time

        if elapsed > max_search_time:
            print(f"Time limit approaching, stopping at depth {current_depth - 1}")
            break

        print(f"Searching depth {current_depth}...")

        try:
            depth_start_time = time.time()
            remaining_time = max_search_time - elapsed

            score, move = alpha_beta_with_time_check(
                board, current_depth, float('-inf'), float('inf'),
                player, player, from_move_generator, depth_start_time, remaining_time
            )

            if move is not None:
                last_best_move = move
                last_best_score = score

                color = "BLACK" if player.lower() == "black" else "WHITE"
                next_boards_dict = from_move_generator(board, color)
                for key, value in next_boards_dict.items():
                    if value == move:
                        best_move_key = key
                        move_str = get_move_string_from_key(key)
                        break

                print(f"Depth {current_depth} best move found (score: {score:.2f}, move: {move_str})")

                current_time = time.time()
                elapsed = current_time - start_time

        except TimeoutError:
            print(f"Time limit reached during depth {current_depth} search")
            break

    if last_best_move is None:
        try:
            current_time = time.time()
            remaining_time = max(time_limit * 0.1, time_limit - (current_time - start_time))
            print(f"No move found. emergency search with {remaining_time:.4f}s")

            score, last_best_move = alpha_beta_with_time_check(
                board, 1, float('-inf'), float('inf'),
                player, player, from_move_generator, current_time, remaining_time
            )

            if last_best_move is not None:
                color = "BLACK" if player.lower() == "black" else "WHITE"
                next_boards_dict = from_move_generator(board, color)
                for key, value in next_boards_dict.items():
                    if value == last_best_move:
                        best_move_key = key
                        move_str = get_move_string_from_key(key)
                        break

        except TimeoutError:
            print("Emergency search timed out")
            color = "BLACK" if player.lower() == "black" else "WHITE"
            next_boards_dict = from_move_generator(board, color)
            if next_boards_dict and len(next_boards_dict) > 0:
                best_move_key = next(iter(next_boards_dict.keys()))
                last_best_move = next_boards_dict[best_move_key]
                move_str = get_move_string_from_key(best_move_key)

    end_time = time.time()
    cdef double total_time = end_time - start_time
    print(f"Search completed in {total_time:.4f}s - {len(transposition_table)} positions searched")

    return last_best_move, move_str, total_time