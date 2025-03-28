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

from collections import OrderedDict
from libc.math cimport sqrt, exp

ctypedef np.int64_t INT64_t
ctypedef np.float64_t FLOAT64_t

DTYPE = np.int64
FLOAT_DTYPE = np.float64

DIRECTIONS = np.array([
    [1, 0],
    [1, 1],
    [0, -1],
    [0, 1],
    [-1, -1],
    [-1, 0]
], dtype=DTYPE)

cdef INT64_t[:, :] DIRECTIONS_VIEW = DIRECTIONS

WEIGHTS = {
    "marble_diff": 1.0,
    "centrality": 0.16,
    "push_ability": 0.35,
    "formation": 0.030,
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
    if len(cache) > max_size:
        remove_count = max_size // 5
        for _ in range(remove_count):
            if cache:
                cache.popitem(last=False)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_valid_coord(tuple coord):
    return coord in VALID_COORDS_SET

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline list get_neighbors(tuple coord):
    return NEIGHBOR_CACHE.get(coord, [])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int coord_to_index(tuple coord):
    return COORD_TO_INDEX_MAP.get(coord, -1)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_in_array(tuple coord, list arr):
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
cdef inline double single_marble_penalty(tuple move_key):
    cdef int length

    if isinstance(move_key[0], tuple):
        length = len(move_key[0])
    elif isinstance(move_key[0], list):
        length = len(move_key[0])
    else:
        return 0.0

    if length == 1:
        return -5.0
    elif length == 2:
        return 3.3
    elif length == 3:
        return 4.4


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_connectivity(list friend_positions, set friend_set):
    cdef double conn_score = 0.0
    cdef tuple pos, neighbor
    cdef int connection_count

    if len(friend_positions) < 3:
        return 0.0

    for pos in friend_positions:
        connection_count = 0

        for neighbor in get_neighbors(pos):
            if neighbor in friend_set:
                connection_count += 1

        if connection_count == 0:
            conn_score -= 6.0
        elif connection_count == 1:
            conn_score -= 2.5

    return conn_score


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_hexagon_formation(list friend_positions, set friend_set):
    cdef double hexagon_score = 0.0
    cdef tuple center_pos, neighbor
    cdef int neighbor_count
    cdef INT64_t[:, :] directions = DIRECTIONS_VIEW
    cdef int total_neighbors = 0
    cdef int positions_with_neighbors = 0

    for center_pos in friend_positions:
        neighbor_count = 0

        for neighbor in get_neighbors(center_pos):
            if neighbor in friend_set:
                neighbor_count += 1

        total_neighbors += neighbor_count

        if neighbor_count > 0:
            positions_with_neighbors += 1

        if neighbor_count == 6:
            hexagon_score += 1.5

        elif neighbor_count >= 4:
            hexagon_score += neighbor_count * 0.2

    cdef double avg_connections = 0.0
    if positions_with_neighbors > 0:
        avg_connections = total_neighbors / float(positions_with_neighbors)

    if avg_connections > 4.0:
        hexagon_score -= (avg_connections - 4.0) * 2.0

    cdef double rectangle_score = 0.0
    cdef tuple pos1, pos2, pos3, pos4
    cdef tuple dir1, dir2
    cdef int dir1_idx, dir2_idx

    for pos1 in friend_positions:
        for dir1_idx in range(directions.shape[0]):
            dir1 = (directions[dir1_idx, 0], directions[dir1_idx, 1])
            pos2 = (pos1[0] + dir1[0], pos1[1] + dir1[1])

            if pos2 in friend_set:
                dir2_idx = (dir1_idx + 2) % 6
                dir2 = (directions[dir2_idx, 0], directions[dir2_idx, 1])

                pos3 = (pos1[0] + dir2[0], pos1[1] + dir2[1])
                pos4 = (pos2[0] + dir2[0], pos2[1] + dir2[1])

                if pos3 in friend_set and pos4 in friend_set:
                    rectangle_score += 1.1

    rectangle_score = min(rectangle_score, 6.0)

    return hexagon_score + rectangle_score

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_push_ability_strength(list groups, set player_set, set opponent_set):
    cdef double strength = 0.0
    cdef list group
    cdef list pos1, pos2, pos3
    cdef tuple diff1, diff2, push_pos, next_pos, final_pos
    cdef int push_count, i

    if len(groups) <= 1:
        return 0.0

    cdef set outer_ring = {
        (9, 5), (9, 6), (9, 7), (9, 8), (9, 9),
        (8, 4), (8, 9), (7, 3), (7, 9), (6, 2),
        (6, 9), (5, 1), (5, 9), (4, 1), (4, 8),
        (3, 1), (3, 7), (2, 1), (2, 6), (1, 1),
        (1, 2), (1, 3), (1, 4), (1, 5)
    }

    cdef set middle_ring = {
        (8, 5), (8, 6), (8, 7), (8, 8), (7, 4),
        (7, 8), (6, 3), (6, 8), (5, 2), (5, 8),
        (4, 2), (4, 7), (3, 2), (3, 6), (2, 2),
        (2, 3), (2, 4), (2, 5)
    }

    cdef set inner_ring = {
        (7, 5), (7, 6), (7, 7), (6, 4), (6, 7),
        (5, 3), (5, 7), (4, 3), (4, 6), (3, 3),
        (3, 4), (3, 5)
    }

    cdef list three_groups = []
    for g in groups:
        if len(g) == 3:
            pos1, pos2, pos3 = g
            diff1 = (pos2[0] - pos1[0], pos2[1] - pos1[1])
            diff2 = (pos3[0] - pos2[0], pos3[1] - pos2[1])
            if diff1 == diff2:
                three_groups.append(g)
                if len(three_groups) >= 15:
                    break

    for group in three_groups:
        pos1, pos2, pos3 = group
        diff1 = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        push_pos = (pos3[0] + diff1[0], pos3[1] + diff1[1])

        if push_pos in opponent_set:
            push_count = 1
            next_pos = push_pos

            for _ in range(1):
                next_check_pos = (next_pos[0] + diff1[0], next_pos[1] + diff1[1])
                if next_check_pos in opponent_set:
                    push_count += 1
                    next_pos = next_check_pos
                else:
                    break

            final_pos = (push_pos[0] + diff1[0] * push_count, push_pos[1] + diff1[1] * push_count)

            if not is_valid_coord(final_pos):
                strength += 30.0 * push_count

            elif final_pos not in player_set and final_pos not in opponent_set:
                for marble_pos in range(push_count):
                    check_pos = (push_pos[0] + diff1[0] * marble_pos, push_pos[1] + diff1[1] * marble_pos)

                    if check_pos in outer_ring:
                        strength += 10.0
                    elif check_pos in middle_ring:
                        strength += 6.0
                    elif check_pos in inner_ring:
                        strength += 3.0

    cdef list two_groups = []
    for g in groups:
        if len(g) == 2:
            two_groups.append(g)
            if len(two_groups) >= 20:
                break

    for group in two_groups:
        pos1, pos2 = group
        diff1 = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        push_pos = (pos2[0] + diff1[0], pos2[1] + diff1[1])

        if push_pos in opponent_set:
            final_pos = (push_pos[0] + diff1[0], push_pos[1] + diff1[1])

            if not is_valid_coord(final_pos):
                strength += 20.0
            elif final_pos not in player_set and final_pos not in opponent_set:

                if push_pos in outer_ring:
                    strength += 10.0
                elif push_pos in middle_ring:
                    strength += 6.0
                elif push_pos in inner_ring:
                    strength += 3.0

    return strength

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double calculate_centrality(list friend_positions, list enemy_positions, set friend_set=None, set enemy_set=None):
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

    if friend_marbles_left < 14:
        friend_weight = 1.0 - (14 - friend_marbles_left) * 0.03

    if enemy_marbles_left < 14:
        enemy_weight = 1.0 - (14 - enemy_marbles_left) * 0.03

    if friend_marbles_left > 0:
        friend_centrality = friend_centrality * (friend_marbles_left * friend_weight)

    if enemy_marbles_left > 0:
        enemy_centrality = enemy_centrality * (enemy_marbles_left * enemy_weight)

    return friend_centrality - enemy_centrality

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_marble_difference(int friend_count, int enemy_count):
    cdef double marble_diff_score = 0.0
    cdef int friend_off, enemy_off
    cdef double enemy_off_score, friend_off_score

    enemy_off = 14 - enemy_count
    enemy_off_score = 600 * enemy_off
    friend_off = 14 - friend_count
    friend_off_score = -650 * friend_off

    if friend_count <= 8:
        return -10000.0
    if enemy_count <= 8:
        return 10000.0


    marble_diff_score += (enemy_off_score + friend_off_score)

    return marble_diff_score

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_board_with_features(list board, str player):
    cdef int friend_idx = 0 if player.lower() == "black" else 1
    cdef int enemy_idx = 1 if player.lower() == "black" else 0

    cdef list friend_marbles = board[friend_idx]
    cdef list enemy_marbles = board[enemy_idx]

    cdef int friend_count = len(friend_marbles)
    cdef int enemy_count = len(enemy_marbles)

    cdef double raw_marble_diff = friend_count - enemy_count
    cdef double marble_diff_score, centrality_score, push_ability_score
    cdef double hexagon_score, connectivity_score
    cdef double enemy_off_score, friend_off_score, total_score
    cdef int enemy_off, friend_off
    cdef dict features

    cdef list friend_positions = [tuple(pos) for pos in friend_marbles]
    cdef list enemy_positions = [tuple(pos) for pos in enemy_marbles]
    cdef set friend_set = set(friend_positions)
    cdef set enemy_set = set(enemy_positions)

    centrality_score = calculate_centrality(friend_positions, enemy_positions)

    cdef list friend_groups = find_groups_fast(friend_marbles)
    cdef list enemy_groups = find_groups_fast(enemy_marbles)

    push_ability_score = evaluate_push_ability_strength(friend_groups, friend_set, enemy_set)

    hexagon_score = evaluate_hexagon_formation(friend_positions, friend_set) * 0.75

    connectivity_score = evaluate_connectivity(friend_positions, friend_set) - \
                         evaluate_connectivity(enemy_positions, enemy_set) * 0.9

    marble_diff_score = evaluate_marble_difference(friend_count, enemy_count)

    features = {
        'marble_diff': marble_diff_score,
        'centrality': centrality_score,
        'push_ability': push_ability_score,
        'formation': hexagon_score,
        'connectivity': connectivity_score,
    }

    total_score = 0.0
    total_score += WEIGHTS.get("marble_diff", 1.0) * marble_diff_score
    total_score += WEIGHTS.get("centrality", 1.0) * centrality_score
    total_score += WEIGHTS.get("push_ability", 1.0) * push_ability_score
    total_score += WEIGHTS.get("formation", 1.0) * hexagon_score
    total_score += WEIGHTS.get("connectivity", 1.0) * connectivity_score

    return total_score, features

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_board(list board, str player):
    cdef double score
    score, _ = evaluate_board_with_features(board, player)
    return score

@cython.boundscheck(False)
@cython.wraparound(False)
def board_to_key(board):
    return compute_zobrist_hash(board)

@cython.boundscheck(False)
@cython.wraparound(False)
def move_ordering_score(list move, int depth, tuple prev_best=None):
    cdef double score = 0.0
    cdef np.uint64_t move_key = compute_zobrist_hash(move)
    cdef str move_key_str = str(move_key)

    if prev_best and move == prev_best[1]:
        return float('inf')

    if depth in killer_moves and move_key_str in killer_moves[depth]:
        score += 100.0 + float(killer_moves[depth][move_key_str])

    return score

cdef double _enhanced_move_ordering(tuple item, int depth, tuple prev_best):
    cdef object move_key, move
    move_key, move = item
    return move_ordering_score(move, depth, prev_best)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple alpha_beta_with_time_check(list board, int depth, double alpha, double beta, str player,
                                      str maximizing_player,
                                      object move_generator, double time_start, double time_limit,
                                      tuple prev_best=None):
    """
    alpha-beta search with time management
    """
    cdef double current_time
    cdef np.uint64_t board_hash
    cdef str tt_key
    cdef tuple result, value
    cdef double eval_score, best_score, score
    cdef list moves_items
    cdef str next_player, current_color
    cdef object best_move = None
    cdef object best_move_key = None
    cdef bint is_maximizer
    cdef np.uint64_t move_hash
    cdef str move_hash_str

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
        result = (eval_score, None, None)
        transposition_table[tt_key] = result
        manage_cache_size(transposition_table, MAX_TABLE_SIZE)
        return result

    current_color = "BLACK" if player.lower() == "black" else "WHITE"
    moves_dict = move_generator(board, current_color)

    if not moves_dict:
        eval_score = evaluate_board(board, maximizing_player)
        result = (eval_score, None, None)
        transposition_table[tt_key] = result
        manage_cache_size(transposition_table, MAX_TABLE_SIZE)
        return result

    moves_items = list(moves_dict.items())

    scores = [(item, _enhanced_move_ordering(item, depth, prev_best)) for item in moves_items]

    is_maximizer = player.lower() == maximizing_player.lower()
    if is_maximizer:
        scores.sort(key=lambda x: x[1], reverse=True)
    else:
        scores.sort(key=lambda x: x[1])

    moves_items = [item for item, _ in scores]

    next_player = "White" if player.lower() == "black" else "Black"

    if is_maximizer:
        best_score = float('-inf')
        for move_key, move in moves_items:

            try:
                score, _, _ = alpha_beta_with_time_check(
                    move, depth - 1, alpha, beta, next_player, maximizing_player,
                    move_generator, time_start, time_limit,
                    (best_move_key, best_move) if best_move else None
                )

                if score > best_score:
                    best_score = score
                    best_move = move
                    best_move_key = move_key

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
                if best_move is None and moves_items:
                    best_move_key, best_move = moves_items[0]
                raise
    else:
        best_score = float('inf')
        for move_key, move in moves_items:

            try:
                score, _, _ = alpha_beta_with_time_check(
                    move, depth - 1, alpha, beta, next_player, maximizing_player,
                    move_generator, time_start, time_limit,
                    (best_move_key, best_move) if best_move else None
                )

                if score < best_score:
                    best_score = score
                    best_move = move
                    best_move_key = move_key

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
                if best_move is None and moves_items:
                    best_move_key, best_move = moves_items[0]
                raise

    result = (best_score, best_move, best_move_key)
    transposition_table[tt_key] = result
    manage_cache_size(transposition_table, MAX_TABLE_SIZE)
    return result

def find_best_move(list board, str player, int depth=4, double time_limit=5.0, object from_move_generator=None):

    cdef int min_depth = 2
    cdef double start_time, current_time, elapsed, max_search_time, depth_start_time, remaining_time
    cdef str move_str
    cdef double best_score, score
    cdef int current_depth
    cdef tuple prev_best, result
    cdef list last_best_move = None
    cdef object last_best_move_key = None
    cdef double last_best_score = 0.0
    cdef dict last_best_features = {}

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

            prev_best = (last_best_move_key, last_best_move) if last_best_move else None

            score, move, move_key = alpha_beta_with_time_check(
                board, current_depth, float('-inf'), float('inf'),
                player, player, from_move_generator, depth_start_time, remaining_time,
                prev_best
            )

            if move is not None:
                last_best_move = move
                last_best_move_key = move_key
                last_best_score = score

                _, last_best_features = evaluate_board_with_features(last_best_move, player)

                move_str = get_move_string_from_key(move_key)
                print(f"Depth {current_depth} best move: {move_str} (score: {score:.2f})")

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

            score, last_best_move, last_best_move_key = alpha_beta_with_time_check(
                board, 1, float('-inf'), float('inf'),
                player, player, from_move_generator, current_time, remaining_time
            )

            if last_best_move:
                _, last_best_features = evaluate_board_with_features(last_best_move, player)

        except TimeoutError:
            print("Emergency search timed out")
            color = "BLACK" if player.lower() == "black" else "WHITE"
            moves_dict = from_move_generator(board, color)
            if moves_dict:
                last_best_move_key, last_best_move = next(iter(moves_dict.items()))

                if last_best_move:
                    _, last_best_features = evaluate_board_with_features(last_best_move, player)

    move_str = get_move_string_from_key(last_best_move_key) if last_best_move_key else "No move found"

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Search completed in {total_time:.4f}s - {len(transposition_table)} positions searched")

    print("\nFeature scores")
    for feature, value in last_best_features.items():
        print(f"  '{feature}': {value:.2f}")
    print("")

    return last_best_move, move_str, last_best_features, total_time

def get_move_string_from_key(move_key):
    if move_key is None:
        return "No move found"

    source_coords, dest_coords = move_key

    letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}

    from_str = ''.join(f"{letter_map[r]}{c}" for r, c in sorted(source_coords))
    to_str = ''.join(f"{letter_map[r]}{c}" for r, c in sorted(dest_coords))

    return f"{from_str},{to_str}"