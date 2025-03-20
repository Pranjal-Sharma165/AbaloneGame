"""
You can code your heuristic function here based on AI.py
after all the implementation of the heuristic function containing all the evaluation functions,
send it to iterative alpha-beta pruning functions below.

for the way to use, you can reference AI.py and process_move_command() in driver.py by typing "2".
and you can set the time and depth you want. make them compete each other

"""


# def board_to_key(board):
#     parts = ["B"]
#     for pos in sorted(board[0]):
#         parts.append(f"{pos[0]},{pos[1]}")
#
#     parts.append("W")
#     for pos in sorted(board[1]):
#         parts.append(f"{pos[0]},{pos[1]}")
#
#     return ":".join(parts)
#
# def move_ordering_score(move, depth, prev_best=None):
#     score = 0
#     move_key = board_to_key(move)
#
#     if prev_best and move == prev_best:
#         return float('inf')
#
#     if depth in killer_moves and move_key in killer_moves[depth]:
#         score += 1000000 + killer_moves[depth][move_key]
#
#     if move_key in history_table:
#         score += history_table[move_key]
#
#     if len(move[0]) != 14 or len(move[1]) != 14:
#         score += 500
#
#     return score
#
# def alpha_beta_with_time_check(board, depth, alpha, beta, player, maximizing_player,
#                                move_generator, time_start, time_limit, prev_best=None):
#     current_time = time.time()
#     if current_time - time_start > time_limit * 0.95:
#         raise TimeoutError("Time limit reached")
#
#     board_key = f"{board_to_key(board)}:{player}:{depth}:{maximizing_player}"
#     if board_key in transposition_table:
#         value = transposition_table.pop(board_key)
#         transposition_table[board_key] = value
#         return value
#
#     if depth <= 0:
#         eval_score = evaluate_board(board, maximizing_player)
#         result = (eval_score, None, None)
#         transposition_table[board_key] = result
#         manage_cache_size(transposition_table, MAX_TABLE_SIZE)
#         return result
#
#     current_color = "BLACK" if player.lower() == "black" else "WHITE"
#     moves_dict = move_generator(board, current_color)
#
#     if not moves_dict:
#         eval_score = evaluate_board(board, maximizing_player)
#         result = (eval_score, None, None)
#         transposition_table[board_key] = result
#         manage_cache_size(transposition_table, MAX_TABLE_SIZE)
#         return result
#
#     moves_items = list(moves_dict.items())
#
#     def enhanced_move_ordering(item):
#         move_key, move = item
#         base_score = move_ordering_score(move, depth, prev_best[1] if prev_best else None)
#
#         single_score = single_marble_penalty(move_key)
#         base_score += single_score * 1
#
#         return base_score
#
#     moves_items.sort(key=enhanced_move_ordering, reverse=True)
#
#     best_move = None
#     best_move_key = None
#     next_player = "White" if player.lower() == "black" else "Black"
#
#     if player.lower() == maximizing_player.lower():
#         best_score = float('-inf')
#         for move_key, move in moves_items:
#             try:
#                 score, _, _ = alpha_beta_with_time_check(
#                     move, depth - 1, alpha, beta, next_player, maximizing_player,
#                     move_generator, time_start, time_limit,
#                     prev_best=(best_move_key, best_move) if best_move else None
#                 )
#
#                 if score > best_score:
#                     best_score = score
#                     best_move = move
#                     best_move_key = move_key
#
#                 alpha = max(alpha, best_score)
#
#                 if beta <= alpha:
#                     move_board_key = board_to_key(move)
#                     if depth not in killer_moves:
#                         killer_moves[depth] = {}
#                     killer_moves[depth][move_board_key] = killer_moves[depth].get(move_board_key, 0) + depth * depth
#                     history_table[move_board_key] = history_table.get(move_board_key, 0) + depth * depth
#                     break
#
#             except TimeoutError:
#                 if best_move is None and moves_items:
#                     best_move_key, best_move = moves_items[0]
#                 raise
#     else:
#         best_score = float('inf')
#         for move_key, move in moves_items:
#             try:
#                 score, _, _ = alpha_beta_with_time_check(
#                     move, depth - 1, alpha, beta, next_player, maximizing_player,
#                     move_generator, time_start, time_limit,
#                     prev_best=(best_move_key, best_move) if best_move else None
#                 )
#
#                 if score < best_score:
#                     best_score = score
#                     best_move = move
#                     best_move_key = move_key
#
#                 beta = min(beta, best_score)
#
#                 if beta <= alpha:
#                     move_board_key = board_to_key(move)
#                     if depth not in killer_moves:
#                         killer_moves[depth] = {}
#                     killer_moves[depth][move_board_key] = killer_moves[depth].get(move_board_key, 0) + depth * depth
#                     history_table[move_board_key] = history_table.get(move_board_key, 0) + depth * depth
#                     break
#
#             except TimeoutError:
#                 if best_move is None and moves_items:
#                     best_move_key, best_move = moves_items[0]
#                 raise
#
#     result = (best_score, best_move, best_move_key)
#     transposition_table[board_key] = result
#     manage_cache_size(transposition_table, MAX_TABLE_SIZE)
#     return result
#
# def find_best_move(board, player, depth=4, time_limit=5.0, from_move_generator=None):
#     print(f"Finding best move for {player}")
#
#     min_depth = 3
#     if depth < min_depth:
#         depth = min_depth
#
#     print(f"Using search depth: {depth}")
#
#     if from_move_generator is None:
#         try:
#             from next_move_generator import generate_all_next_moves
#             from_move_generator = generate_all_next_moves
#         except ImportError:
#             raise ImportError("Move generator not provided or not found.")
#
#     start_time = time.time()
#     best_move = None
#     best_move_key = None
#     best_score = 0.0
#
#     transposition_table.clear()
#     killer_moves.clear()
#     history_table.clear()
#
#     for current_depth in range(min_depth, depth + 1):
#         print(f"Searching depth {current_depth}...")
#
#         try:
#             score, move, move_key = alpha_beta_with_time_check(
#                 board, current_depth, float('-inf'), float('inf'),
#                 player, player, from_move_generator, start_time, time_limit,
#                 prev_best=(best_move_key, best_move) if best_move else None
#             )
#
#             if move is not None:
#                 best_move = move
#                 best_move_key = move_key
#                 best_score = score
#
#                 move_str = get_move_string_from_key(move_key)
#                 print(f"Current best move: {move_str} (score: {score:.2f})")
#
#         except TimeoutError:
#             print(f"Time limit reached during depth {current_depth} search")
#             break
#
#         remaining = time_limit - (time.time() - start_time)
#         if remaining < time_limit * 0.05:
#             print(f"Search terminated")
#             break
#
#     if best_move is None:
#         try:
#             _, best_move, best_move_key = alpha_beta_with_time_check(
#                 board, min_depth - 1, float('-inf'), float('inf'),
#                 player, player, from_move_generator, start_time, time_limit
#             )
#         except TimeoutError:
#             color = "BLACK" if player.lower() == "black" else "WHITE"
#             moves_dict = from_move_generator(board, color)
#             if moves_dict:
#                 best_move_key, best_move = next(iter(moves_dict.items()))
#
#     move_str = get_move_string_from_key(best_move_key) if best_move_key else "No move found"
#
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Search completed in {total_time:.2f}s - {len(transposition_table)} positions analyzed")
#
#     return best_move, move_str
#
# def get_move_string_from_key(move_key):
#     if move_key is None:
#         return "No move found"
#
#     source_coords, dest_coords = move_key
#
#     letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}
#
#     from_str = ''.join(f"{letter_map[r]}{c}" for r, c in sorted(source_coords))
#     to_str = ''.join(f"{letter_map[r]}{c}" for r, c in sorted(dest_coords))
#
#     return f"{from_str},{to_str}"