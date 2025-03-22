import cProfile
import pstats
from move import convert_board_format
from driver import STANDARD_BOARD_INIT
from next_move_generator import generate_all_next_moves
from AI import find_best_move

def run_ai():
    board = convert_board_format(STANDARD_BOARD_INIT)
    find_best_move(
        board=board,
        player="Black",
        depth=4,
        time_limit=10.0,
        from_move_generator=generate_all_next_moves
    )

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_ai()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats(30)

