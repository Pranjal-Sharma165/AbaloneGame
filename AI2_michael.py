"""
You can code your heuristic function here based on AI.py
after all the implementation of the heuristic function containing all the evaluation functions,
send it to iterative alpha-beta pruning functions below.

for the way to use, you can reference AI.py and process_move_command() in driver.py by typing "2".
and you can set the time and depth you want. make them compete each other

"""

import time
from collections import OrderedDict

import numpy as np

from move import DIRECTION_VECTORS

DIRECTIONS = np.array([
    [1, 0],
    [1, 1],
    [0, -1],
    [0, 1],
    [-1, -1],
    [-1, 0]
])

WEIGHTS = {
    "marble_diff": 0.581,
    "centrality": 0.701,
    "push_ability": 0.885,
    "edge_safety": -0.444,
    "formation": 0.104,
    "mobility": 0.109
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
])

VALID_COORDS_SET = {tuple(coord) for coord in VALID_COORDS}

CENTER_COORD = np.array([5, 5])
CENTRALITY_VALUES = np.zeros(VALID_COORDS.shape[0])

for i in range(VALID_COORDS.shape[0]):
    dist = np.abs(VALID_COORDS[i, 0] - CENTER_COORD[0]) + np.abs(VALID_COORDS[i, 1] - CENTER_COORD[1])
    CENTRALITY_VALUES[i] = -dist

EDGE_MAP = np.zeros(VALID_COORDS.shape[0])

for i in range(VALID_COORDS.shape[0]):
    neighbors = 0
    for j in range(DIRECTIONS.shape[0]):
        neighbor = VALID_COORDS[i] + DIRECTIONS[j]
        for k in range(VALID_COORDS.shape[0]):
            if VALID_COORDS[k, 0] == neighbor[0] and VALID_COORDS[k, 1] == neighbor[1]:
                neighbors += 1
                break
    EDGE_MAP[i] = (neighbors < 6)

NEIGHBOR_CACHE = {}
for i in range(VALID_COORDS.shape[0]):
    coord = tuple(VALID_COORDS[i])
    neighbors = []
    for j in range(DIRECTIONS.shape[0]):
        neighbor = (VALID_COORDS[i][0] + DIRECTIONS[j][0], VALID_COORDS[i][1] + DIRECTIONS[j][1])
        if neighbor in VALID_COORDS_SET:
            neighbors.append(neighbor)
    NEIGHBOR_CACHE[coord] = neighbors

EDGE_COORDS = np.array([VALID_COORDS[i] for i in range(len(VALID_COORDS)) if EDGE_MAP[i] == 1])

EDGE_DIRECTIONS = {}
for i in range(EDGE_COORDS.shape[0]):
    coord = tuple(EDGE_COORDS[i])
    edge_dirs = []
    row, col = coord

    if row == 1:
        edge_dirs.extend([(-1, -1), (-1, 0)])
    if row == 9:
        edge_dirs.extend([(1, 0), (1, 1)])
    if col == 1:
        edge_dirs.append((0, -1))
    if col == 9:
        edge_dirs.append((0, 1))

    EDGE_DIRECTIONS[coord] = edge_dirs

CRITICAL_EDGE_POSITIONS = {coord for coord, dirs in EDGE_DIRECTIONS.items() if len(dirs) > 1}

MAX_TABLE_SIZE = 200000
transposition_table = OrderedDict()
history_table = {}
killer_moves = {}

MAX_CACHE_SIZE = 5000
group_cache = OrderedDict()

def manage_cache_size(cache, max_size):
    if len(cache) > max_size:
        remove_count = max_size // 5
        for _ in range(remove_count):
            if cache:
                cache.popitem(last=False)

def is_valid_coord(coord):
    return coord in VALID_COORDS_SET

def get_neighbors(coord):
    return NEIGHBOR_CACHE.get(tuple(coord), [])




# changing time from 0.472 sec to 0.058 sec
COORD_INDEX_MAP = {tuple(coord): idx for idx, coord in enumerate(VALID_COORDS)}

def coord_to_index(coord):
    return COORD_INDEX_MAP.get(tuple(coord), -1)

def find_groups_fast(marbles):
    if not marbles:
        return []

    cache_key = tuple(sorted(tuple(m) for m in marbles))
    if cache_key in group_cache:
        value = group_cache.pop(cache_key)
        group_cache[cache_key] = value
        return value

    result = []
    for i in range(len(marbles)):
        result.append([marbles[i]])

    if len(marbles) < 2:
        group_cache[cache_key] = result
        manage_cache_size(group_cache, MAX_CACHE_SIZE)
        return result

    marbles_set = {tuple(m) for m in marbles}
    max_process = min(len(marbles), 15)

    for i in range(max_process):
        marble1 = tuple(marbles[i])
        for j in range(DIRECTIONS.shape[0]):
            dir_vec = DIRECTIONS[j]
            marble2 = (marble1[0] + dir_vec[0], marble1[1] + dir_vec[1])
            if marble2 in marbles_set:
                result.append([list(marble1), list(marble2)])

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

def single_marble_penalty(move_key):
    source_coords, dest_coords = move_key

    if len(source_coords) == 1:
        return -4.0
    elif len(source_coords) == 2:
        return 1.2
    elif len(source_coords) == 3:
        return 1.4

def evaluate_hexagon_formation(friend_positions):

    hexagon_score = 0.0
    friend_set = set(friend_positions)

    for center_pos in friend_positions:
        neighbor_count = 0

        for dir_vec in DIRECTIONS:
            neighbor = (center_pos[0] + dir_vec[0], center_pos[1] + dir_vec[1])
            if neighbor in friend_set:
                neighbor_count += 1

        if neighbor_count >= 4:
            hexagon_score += (neighbor_count - 3) * 5.0

            if neighbor_count == 6:
                hexagon_score += 10.0

    for pos1 in friend_positions:
        for dir1_idx, dir1 in enumerate(DIRECTIONS):
            pos2 = (pos1[0] + dir1[0], pos1[1] + dir1[1])
            if pos2 in friend_set:

                dir2_idx = (dir1_idx + 1) % 6
                dir2 = DIRECTIONS[dir2_idx]
                pos3 = (pos2[0] + dir2[0], pos2[1] + dir2[1])

                if pos3 in friend_set:

                    hexagon_score += 2.0

                    dir3_idx = (dir2_idx + 1) % 6
                    dir3 = DIRECTIONS[dir3_idx]
                    pos4 = (pos3[0] + dir3[0], pos3[1] + dir3[1])

                    if pos4 in friend_set:
                        hexagon_score += 3.0

    return hexagon_score



#changing to go deeper in the tree
def evaluate_push_ability_strength(groups, player_set, opponent_set):
    strength = 0.0

    if len(groups) <= 1:
        return 0.0

    # 3-marbles groups
    three_groups = [g for g in groups if len(g) == 3]
    for group in three_groups[:10]:
        pos1, pos2, pos3 = group
        diff1 = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        diff2 = (pos3[0] - pos2[0], pos3[1] - pos2[1])

        if diff1 == diff2:
            push_dir = diff1
            push_pos = (pos3[0] + push_dir[0], pos3[1] + push_dir[1])

            if push_pos in opponent_set:
                push_count = 1
                next_pos = push_pos
                for _ in range(2):
                    next_pos = (next_pos[0] + push_dir[0], next_pos[1] + push_dir[1])
                    if next_pos in opponent_set:
                        push_count += 1
                    else:
                        break

                if push_count <= 2:
                    strength += 6.0 * push_count

                last_pos = (push_pos[0] + push_dir[0] * push_count, push_pos[1] + push_dir[1] * push_count)
                if not is_valid_coord(last_pos):
                    strength += 15.0
            elif is_valid_coord(push_pos) and push_pos not in player_set:
                strength += 1.0

    # 2-marbles groups
    two_groups = [g for g in groups if len(g) == 2]
    for group in two_groups[:10]:
        pos1, pos2 = group
        diff = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        push_pos = (pos2[0] + diff[0], pos2[1] + diff[1])

        if push_pos in opponent_set:
            next_pos = (push_pos[0] + diff[0], push_pos[1] + diff[1])
            if next_pos not in opponent_set:
                strength += 4.0
                if not is_valid_coord(next_pos):
                    strength += 10.0

    return strength

def evaluate_edge_safety(positions, opponent_positions):
    if not positions:
        return 0.0

    safety_score = 0.0
    positions_set = set(tuple(pos) for pos in positions)
    opponent_set = {tuple(pos) for pos in opponent_positions}
    edge_marbles = [pos for pos in positions if tuple(pos) in EDGE_DIRECTIONS]

    if not edge_marbles:
        return 0.0

    max_edges = min(len(edge_marbles), 8)

    for i in range(min(max_edges, len(edge_marbles))):
        pos = edge_marbles[i]
        pos_tuple = tuple(pos)

        for edge_dir in EDGE_DIRECTIONS.get(pos_tuple, []):
            opposite_dir = (-edge_dir[0], -edge_dir[1])
            check_pos = (pos[0] + opposite_dir[0], pos[1] + opposite_dir[1])

            if check_pos in opponent_set:
                next_pos = (check_pos[0] + opposite_dir[0], check_pos[1] + opposite_dir[1])
                if next_pos in opponent_set:
                    safety_score -= 2.5
                else:
                    safety_score -= 1.2

    return safety_score

def evaluate_mobility(positions, friend_set, enemy_set):
    if len(positions) < 3:
        return len(positions) * 2

    mobility = 0
    max_positions = min(len(positions), 8)
    positions_to_check = positions[:max_positions]

    for pos in positions_to_check:
        for nbr in get_neighbors(pos):
            if nbr not in friend_set and nbr not in enemy_set:
                mobility += 1

    if len(positions) > max_positions:
        mobility = int(mobility * (len(positions) / max_positions))

    return mobility

def calculate_centrality(friend_positions, enemy_positions):
    friend_centrality = 0.0
    enemy_centrality = 0.0
    max_friend = min(len(friend_positions), 10)
    max_enemy = min(len(enemy_positions), 10)

    # Precompute coord_to_index only once using zip
    for pos in friend_positions[:max_friend]:
        idx = coord_to_index(pos)
        if idx >= 0:
            friend_centrality += CENTRALITY_VALUES[idx]

    for pos in enemy_positions[:max_enemy]:
        idx = coord_to_index(pos)
        if idx >= 0:
            enemy_centrality += CENTRALITY_VALUES[idx]

    # Scaling adjustment
    if len(friend_positions) > max_friend:
        friend_centrality *= len(friend_positions) / max_friend
    if len(enemy_positions) > max_enemy:
        enemy_centrality *= len(enemy_positions) / max_enemy

    return friend_centrality - enemy_centrality


def evaluate_board(board, starting_board, player):  # Michael - added starting_board everywhere evaluate_board is used
    friend_idx = 0 if player.lower() == "black" else 1
    enemy_idx = 1 if player.lower() == "black" else 0

    friend_marbles = board[friend_idx]
    enemy_marbles = board[enemy_idx]

    friend_count = len(friend_marbles)
    enemy_count = len(enemy_marbles)
    marble_diff = friend_count - enemy_count

    # Early termination for win/loss
    if friend_count <= 8:
        return -10000.0
    if enemy_count <= 8:
        return 10000.0
    if marble_diff >= 5:
        return 800.0
    if marble_diff <= -5:
        return -800.0

    # Precompute positions and sets
    friend_positions = list(map(tuple, friend_marbles))
    enemy_positions = list(map(tuple, enemy_marbles))
    friend_set = set(friend_positions)
    enemy_set = set(enemy_positions)

    total_lost = 28 - (friend_count + enemy_count)
    progress_factor = 1.0 + total_lost / 10.0
    marble_feature = marble_diff * progress_factor

    centrality_score = calculate_centrality(friend_positions, enemy_positions)

    friend_groups = find_groups_fast(friend_marbles)
    enemy_groups = find_groups_fast(enemy_marbles)

    # Calculating previous board's push_ability_score - Michael added comparison to cancel moves resulting in lower push_ability (lower strength)
    start_friend_marbles = starting_board[friend_idx]
    start_enemy_marbles = starting_board[enemy_idx]
    start_friend_count = len(start_friend_marbles)
    start_enemy_count = len(start_enemy_marbles)
    start_material_diff = start_friend_count - start_enemy_count
    #
    if start_material_diff > 0:
        start_friend_groups = find_groups_fast(start_friend_marbles)
        start_enemy_groups = find_groups_fast(start_enemy_marbles)
        start_friend_positions = list(map(tuple, start_friend_marbles))
        start_enemy_positions = list(map(tuple, start_enemy_marbles))
        start_friend_set = set(start_friend_positions)
        start_enemy_set = set(start_enemy_positions)
        start_push_ability_score = (
                evaluate_push_ability_strength(start_friend_groups, start_friend_set, start_enemy_set)
                - evaluate_push_ability_strength(start_enemy_groups, start_enemy_set, start_friend_set)
        )
        push_ability_score = (
            evaluate_push_ability_strength(friend_groups, friend_set, enemy_set)
            - evaluate_push_ability_strength(enemy_groups, enemy_set, friend_set)
        )
        if push_ability_score < start_push_ability_score:
            push_ability_score = 0
    else:
        push_ability_score = (
            evaluate_push_ability_strength(friend_groups, friend_set, enemy_set)
            - evaluate_push_ability_strength(enemy_groups, enemy_set, friend_set)
        )

    edge_safety_score = (
        evaluate_edge_safety(friend_positions, enemy_marbles)
        - evaluate_edge_safety(enemy_positions, friend_marbles)
    )

    hexagon_score = (
        evaluate_hexagon_formation(friend_positions)
        - 0.8 * evaluate_hexagon_formation(enemy_positions)
    )

    enemy_off = 14 - enemy_count
    friend_off = 14 - friend_count
    enemy_off_score = 600 * enemy_off
    friend_off_score = -650 * friend_off

    mobility_score = (
        evaluate_mobility(friend_positions, friend_set, enemy_set)
        - evaluate_mobility(enemy_positions, enemy_set, friend_set)
    )

    features = {
        'marble_diff': marble_feature,
        'centrality': centrality_score,
        'push_ability': push_ability_score,
        'edge_safety': edge_safety_score,
        'mobility': mobility_score,
        'formation': hexagon_score,
        'off_board': enemy_off_score + friend_off_score
    }

    weights = WEIGHTS.copy()

    total_score = sum(weights.get(name, 1) * val for name, val in features.items())
    return total_score

def board_to_key(board):
    parts = ["B"]
    for pos in sorted(board[0]):
        parts.append(f"{pos[0]},{pos[1]}")

    parts.append("W")
    for pos in sorted(board[1]):
        parts.append(f"{pos[0]},{pos[1]}")

    return ":".join(parts)

def move_ordering_score(move, depth, prev_best=None):
    score = 0
    move_key = board_to_key(move)

    if prev_best and move == prev_best:
        return float('inf')

    if depth in killer_moves and move_key in killer_moves[depth]:
        score += 1000000 + killer_moves[depth][move_key]

    if move_key in history_table:
        score += history_table[move_key]

    if len(move[0]) != 14 or len(move[1]) != 14:
        score += 500

    return score

def alpha_beta_with_time_check(board, starting_board, depth, alpha, beta, player, maximizing_player,
                               move_generator, time_start, time_limit, prev_best=None):
    current_time = time.time()
    if current_time - time_start > time_limit * 0.95:
        raise TimeoutError("Time limit reached")

    board_key = f"{board_to_key(board)}:{player}:{depth}:{maximizing_player}"
    if board_key in transposition_table:
        value = transposition_table.pop(board_key)
        transposition_table[board_key] = value
        return value

    if depth <= 0:
        eval_score = evaluate_board(board, starting_board, maximizing_player)
        result = (eval_score, None, None)
        transposition_table[board_key] = result
        manage_cache_size(transposition_table, MAX_TABLE_SIZE)
        return result

    current_color = "BLACK" if player.lower() == "black" else "WHITE"
    moves_dict = move_generator(board, current_color)

    if not moves_dict:
        eval_score = evaluate_board(board, starting_board, maximizing_player)
        result = (eval_score, None, None)
        transposition_table[board_key] = result
        manage_cache_size(transposition_table, MAX_TABLE_SIZE)
        return result

    moves_items = list(moves_dict.items())

    def enhanced_move_ordering(item):
        move_key, move = item
        base_score = move_ordering_score(move, depth, prev_best[1] if prev_best else None)

        single_score = single_marble_penalty(move_key)
        base_score += single_score * 1

        return base_score

    moves_items.sort(key=enhanced_move_ordering, reverse=True)

    best_move = None
    best_move_key = None
    next_player = "White" if player.lower() == "black" else "Black"

    if player.lower() == maximizing_player.lower():
        best_score = float('-inf')
        for move_key, move in moves_items:
            try:
                score, _, _ = alpha_beta_with_time_check(
                    move, starting_board, depth - 1, alpha, beta, next_player, maximizing_player,
                    move_generator, time_start, time_limit,
                    prev_best=(best_move_key, best_move) if best_move else None
                )

                if score > best_score:
                    best_score = score
                    best_move = move
                    best_move_key = move_key

                alpha = max(alpha, best_score)

                if beta <= alpha:
                    move_board_key = board_to_key(move)
                    if depth not in killer_moves:
                        killer_moves[depth] = {}
                    killer_moves[depth][move_board_key] = killer_moves[depth].get(move_board_key, 0) + depth * depth
                    history_table[move_board_key] = history_table.get(move_board_key, 0) + depth * depth
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
                    move, starting_board, depth - 1, alpha, beta, next_player, maximizing_player,
                    move_generator, time_start, time_limit,
                    prev_best=(best_move_key, best_move) if best_move else None
                )

                if score < best_score:
                    best_score = score
                    best_move = move
                    best_move_key = move_key

                beta = min(beta, best_score)

                if beta <= alpha:
                    move_board_key = board_to_key(move)
                    if depth not in killer_moves:
                        killer_moves[depth] = {}
                    killer_moves[depth][move_board_key] = killer_moves[depth].get(move_board_key, 0) + depth * depth
                    history_table[move_board_key] = history_table.get(move_board_key, 0) + depth * depth
                    break

            except TimeoutError:
                if best_move is None and moves_items:
                    best_move_key, best_move = moves_items[0]
                raise

    result = (best_score, best_move, best_move_key)
    transposition_table[board_key] = result
    manage_cache_size(transposition_table, MAX_TABLE_SIZE)
    return result

def find_best_move(board, player, depth=4, time_limit=5.0, from_move_generator=None):
    print(f"Finding best move for {player}")

    min_depth = 3
    if depth < min_depth:
        depth = min_depth

    print(f"Using search depth: {depth}")

    if from_move_generator is None:
        try:
            from next_move_generator import generate_all_next_moves
            from_move_generator = generate_all_next_moves
        except ImportError:
            raise ImportError("Move generator not provided or not found.")

    start_time = time.time()
    best_move = None
    best_move_key = None
    best_score = 0.0

    transposition_table.clear()
    killer_moves.clear()
    history_table.clear()

    for current_depth in range(min_depth, depth + 1):
        print(f"Searching depth {current_depth}...")

        try:
            score, move, move_key = alpha_beta_with_time_check(
                board, board, current_depth, float('-inf'), float('inf'),
                player, player, from_move_generator, start_time, time_limit,
                prev_best=(best_move_key, best_move) if best_move else None
            )

            if move is not None:
                best_move = move
                best_move_key = move_key
                best_score = score

                move_str = get_move_string_from_key(move_key)
                print(f"Current best move: {move_str} (score: {score:.2f})")

        except TimeoutError:
            print(f"Time limit reached during depth {current_depth} search")
            break

        remaining = time_limit - (time.time() - start_time)
        if remaining < time_limit * 0.05:
            print(f"Search terminated")
            break

    if best_move is None:
        try:
            _, best_move, best_move_key = alpha_beta_with_time_check(
                board, min_depth - 1, float('-inf'), float('inf'),
                player, player, from_move_generator, start_time, time_limit
            )
        except TimeoutError:
            color = "BLACK" if player.lower() == "black" else "WHITE"
            moves_dict = from_move_generator(board, color)
            if moves_dict:
                best_move_key, best_move = next(iter(moves_dict.items()))

    move_str = get_move_string_from_key(best_move_key) if best_move_key else "No move found"

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Search completed in {total_time:.2f}s - {len(transposition_table)} positions analyzed")

    return best_move, move_str

def get_move_string_from_key(move_key):
    if move_key is None:
        return "No move found"

    source_coords, dest_coords = move_key

    letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}

    from_str = ''.join(f"{letter_map[r]}{c}" for r, c in sorted(source_coords))
    to_str = ''.join(f"{letter_map[r]}{c}" for r, c in sorted(dest_coords))

    return f"{from_str},{to_str}"