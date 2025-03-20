from move import convert_board_format
from AI import find_best_move, evaluate_board
import time
from driver import STANDARD_BOARD_INIT
from next_move_generator import generate_all_next_moves

board = convert_board_format(STANDARD_BOARD_INIT)

print(f"현재 보드 상태 평가: {evaluate_board(board, 'Black')}")


depths = [3, 4, 5]

for depth in depths:
    print(f"\n{depth}에서 최적 이동 찾는 중")
    start_time = time.time()

    try:
        best_move, move_notation = find_best_move(
            board,
            "Black",
            depth=depth,
            time_limit=120.0,
            from_move_generator=generate_all_next_moves
        )

        end_time = time.time()
        search_time = end_time - start_time

        print(f"최적 이동: {move_notation}")
        print(f"탐색 시간: {search_time:.2f}초")

    except Exception as e:
        print(f"깊이 {depth}에서 오류 발생: {e}")
