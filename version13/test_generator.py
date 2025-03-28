# test_enhanced_ai.py
from next_move_generator import generate_all_next_moves
from AI import find_best_move

import time
from move import convert_board_format
from driver import STANDARD_BOARD_INIT
board = convert_board_format(STANDARD_BOARD_INIT)

start_time = time.time()

best_move, move_notation, _, _ = find_best_move(board,'Black',depth=4,time_limit=10,from_move_generator=generate_all_next_moves)

end_time = time.time()
search_time = end_time - start_time

print(f"best move: {move_notation}")
print(f"time: {search_time:.2f} seconds")


