import time
from copy import deepcopy
from move import move_marbles, move_validation
from next_move_generator import generate_all_next_moves

from AI import (
    find_groups_fast,
    evaluate_hexagon_formation,
    evaluate_push_ability_strength,
    evaluate_edge_safety,
    evaluate_mobility,
    calculate_centrality,
)

WEIGHTS = {
    "marble_diff": 0.7,
    "centrality": 0.4,
    "push_ability": 1.0,
    "edge_safety": -0.6,
    "mobility": 0.2,
    "formation": 0.3,
    "off_board": 0.8
}


def aggressive_board_evaluation(board, player):
    friend_idx = 0 if player.lower() == "black" else 1
    enemy_idx = 1 - friend_idx

    friend_marbles = board[friend_idx]
    enemy_marbles = board[enemy_idx]

    friend_count = len(friend_marbles)
    enemy_count = len(enemy_marbles)
    marble_diff = friend_count - enemy_count

    if friend_count <= 8:
        return -10000.0
    if enemy_count <= 8:
        return 10000.0
    if marble_diff >= 5:
        return 1000.0
    if marble_diff <= -5:
        return -1000.0

    friend_pos = [tuple(p) for p in friend_marbles]
    enemy_pos = [tuple(p) for p in enemy_marbles]
    friend_set = set(friend_pos)
    enemy_set = set(enemy_pos)

    total_lost = 28 - (friend_count + enemy_count)
    game_progress = 1.0 + total_lost / 10.0
    marble_score = marble_diff * game_progress

    center_score = calculate_centrality(friend_pos, enemy_pos)

    friend_groups = find_groups_fast(friend_marbles)
    enemy_groups = find_groups_fast(enemy_marbles)

    push_score = evaluate_push_ability_strength(friend_groups, friend_set, enemy_set)
    push_score -= 0.8 * evaluate_push_ability_strength(enemy_groups, enemy_set, friend_set)

    edge_score = evaluate_edge_safety(friend_pos, enemy_marbles) - evaluate_edge_safety(enemy_pos, friend_marbles)
    form_score = evaluate_hexagon_formation(friend_pos) - 0.7 * evaluate_hexagon_formation(enemy_pos)

    off_score = (14 - enemy_count) * 700 - (14 - friend_count) * 650
    move_score = evaluate_mobility(friend_pos, friend_set, enemy_set) - evaluate_mobility(enemy_pos, enemy_set, friend_set)

    features = {
        'marble_diff': marble_score,
        'centrality': center_score,
        'push_ability': push_score,
        'edge_safety': edge_score,
        'mobility': move_score,
        'formation': form_score,
        'off_board': off_score,
    }

    debug_score = {k: round(WEIGHTS[k] * features[k], 2) for k in features}
    print("[EVAL DEBUG]", debug_score)
    total = sum(WEIGHTS[k] * features[k] for k in features)
    print("[EVAL SCORE]", round(total, 2))

    return total


def alpha_beta_aggressive(board, depth, alpha, beta, maximizing, player, start_time, time_limit):
    if depth == 0 or time.time() - start_time > time_limit:
        score = aggressive_board_evaluation(board, player)
        print(f"[DEBUG] Depth 0 or time limit reached. Returning score: {score}")
        return score, None

    curr_player = player if maximizing else ("white" if player == "black" else "black")

    moves = generate_all_next_moves(board, curr_player.upper())
    print(f"[DEBUG] {curr_player.upper()} possible moves at depth {depth}: {len(moves)}")

    if isinstance(moves, dict):
        moves = list(moves.values())

    print(f"[DEBUG] {curr_player.upper()} possible moves at depth {depth}: {len(moves)}")

    if not moves:
        eval_score = aggressive_board_evaluation(board, player)
        print(f"[DEBUG] No moves for {curr_player.upper()}. Eval: {eval_score}")
        return eval_score, None

    best = None

    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            state = deepcopy(board)

            try:
                source = move[0]
                dest = move[1]

                if isinstance(source, tuple) and len(source) == 1 and isinstance(source[0], tuple):
                    source = source[0]
                if isinstance(dest, tuple) and len(dest) == 1 and isinstance(dest[0], tuple):
                    dest = dest[0]


            except IndexError:
                print(f"[DEBUG] Skipping malformed move (too short): {move}")
                continue

            is_valid, msg = move_validation(source, dest, state, curr_player.upper())
            if not is_valid:
                print(f"[DEBUG] Skipping invalid move: {move}, reason: {msg}")
                continue

            move_marbles(state, [source, dest])

            score, _ = alpha_beta_aggressive(state, depth - 1, alpha, beta, False, player, start_time, time_limit)
            print(f"[DEBUG] Depth {depth}, {curr_player.upper()} move: {move}, Score: {round(score, 2)}")

            if score > max_eval:
                max_eval = score
                best = deepcopy(move)
                print(f"[DEBUG] --> New BEST move at depth {depth}: {move}, Score: {round(score, 2)}")

            alpha = max(alpha, score)
            if beta <= alpha:
                print(f"[DEBUG] Alpha-Beta cutoff at depth {depth} for {curr_player.upper()}")
                break

        return max_eval, best

    else:
        min_eval = float('inf')
        for seq in moves:
            state = deepcopy(board)
            valid = all(move_validation(m[0], m[1], state, curr_player.upper())[0] for m in seq)

            if not valid:
                print(f"[DEBUG] Skipping invalid move for {curr_player.upper()}: {seq}")
                continue

            for m in seq:
                move_marbles(state, m)

            score, _ = alpha_beta_aggressive(state, depth - 1, alpha, beta, True, player, start_time, time_limit)

            if score < min_eval:
                min_eval = score
                best = deepcopy(seq)
                print(f"[DEBUG] {curr_player.upper()} NEW BEST: {seq} (score: {round(score, 2)})")

            beta = min(beta, score)
            if beta <= alpha:
                print("[DEBUG] Alpha-Beta cutoff (MIN)")
                break

        return min_eval, best


def find_best_move(board, player, max_depth=4, time_limit=9.5):
    start = time.time()
    best_move = None
    best_score = float('-inf')
    depth = 1

    while depth <= max_depth:
        if time.time() - start >= time_limit:
            print("Time limit reached during depth", depth)
            break

        print(f"Searching depth {depth}...")
        score, move = alpha_beta_aggressive(
            deepcopy(board), depth, float('-inf'), float('inf'), True, player, start, time_limit
        )

        if time.time() - start >= time_limit:
            print("Time limit reached during depth", depth)
            break

        if move and score > best_score:
            best_move = move
            best_score = score
            print(f"Current best move: {','.join(str(m) for m in move)} (score: {round(best_score, 2)})")

        depth += 1

    print(f"Search completed in {round(time.time() - start, 2)}s")
    return best_move

