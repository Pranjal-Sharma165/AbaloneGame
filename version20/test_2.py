import time
import move_cy as mc
import next_move_generator_cy as nmgc
from IO import BoardIO

# Test board setup
STANDARD_BOARD_INIT = {
    "I5": "#D9D9D9", "I6": "#D9D9D9", "I7": "#D9D9D9", "I8": "#D9D9D9", "I9": "#D9D9D9",
    "H4": "#D9D9D9", "H5": "#D9D9D9", "H6": "#D9D9D9", "H7": "#D9D9D9", "H8": "#D9D9D9", "H9": "#D9D9D9",
    "G3": "Blank", "G4": "Blank", "G5": "#D9D9D9", "G6": "#D9D9D9", "G7": "#D9D9D9", "G8": "Blank", "G9": "Blank",
    "F2": "Blank", "F3": "Blank", "F4": "Blank", "F5": "Blank", "F6": "Blank", "F7": "Blank", "F8": "Blank",
    "F9": "Blank",
    "E1": "Blank", "E2": "Blank", "E3": "Blank", "E4": "Blank", "E5": "Blank", "E6": "Blank", "E7": "Blank",
    "E8": "Blank", "E9": "Blank",
    "D1": "Blank", "D2": "Blank", "D3": "Blank", "D4": "Blank", "D5": "Blank", "D6": "Blank", "D7": "Blank",
    "D8": "Blank",
    "C1": "Blank", "C2": "Blank", "C3": "#8A8A8A", "C4": "#8A8A8A", "C5": "#8A8A8A", "C6": "Blank", "C7": "Blank",
    "B1": "#8A8A8A", "B2": "#8A8A8A", "B3": "#8A8A8A", "B4": "#8A8A8A", "B5": "#8A8A8A", "B6": "#8A8A8A",
    "A1": "#8A8A8A", "A2": "#8A8A8A", "A3": "#8A8A8A", "A4": "#8A8A8A", "A5": "#8A8A8A"
}


a = BoardIO.import_current_text_to_board("./output/Test5.input")[0]
c = BoardIO.import_current_text_to_board("./output/Test5.input")[1]


def test_move_validation():
    board_list = mc.convert_board_format(STANDARD_BOARD_INIT)

    test_cases = [
        ([[3, 3]], [[4, 3]], "BLACK", True, "Valid single marble move to empty space"),
        ([[1, 1], [1, 3]], [[2, 1], [2, 3]], "BLACK", False, "Non-adjacent marbles"),
        ([[3, 3], [3, 4]], [[4, 3], [4, 4]], "BLACK", True, "Adjacent marbles to empty spaces")
    ]

    for i, (src, dst, color, expected, desc) in enumerate(test_cases):
        start = time.time()
        result, reason = mc.move_validation(src, dst, board_list, color)
        end = time.time()
        status = "PASS" if result == expected else "FAIL"
        print(f"Test {i + 1} ({desc}): {status} - {reason} ({(end - start) * 1000:.3f}ms)")


def test_move_execution():
    board_list = mc.convert_board_format(a)

    source_coords = [[3, 3]]
    dest_coords = [[4, 3]]

    start = time.time()
    new_board, pushed = mc.move_marbles(source_coords, dest_coords, board_list, c)
    end = time.time()

    print(f"Move execution: {pushed} marbles pushed off ({(end - start) * 1000:.3f}ms)")


def test_next_move_generation():
    board_list = mc.convert_board_format(a)

    start = time.time()
    next_moves = nmgc.generate_all_next_moves(board_list, c)
    end = time.time()

    print(f"Generated {len(next_moves)} possible moves in {end - start:.6f} seconds")


    print("\nSample moves:")
    sample_moves = list(next_moves.items())
    for i, (key, board) in enumerate(sample_moves):
        src, dst = key
        move_str = nmgc.format_coords_to_string([list(x) for x in src]) + "," + nmgc.format_coords_to_string(
            [list(x) for x in dst])
        print(f"Move {i + 1}: {move_str}")


def test_performance():
    board_list = mc.convert_board_format(a)

    source_coords = [[3, 3]]
    dest_coords = [[4, 3]]

    iterations = 10000
    start = time.time()
    for _ in range(iterations):
        mc.move_validation(source_coords, dest_coords, board_list, c)
    end = time.time()

    print(f"Move validation: {iterations} iterations in {end - start:.6f} seconds")
    print(f"Average time per validation: {(end - start) / iterations * 1000:.6f}ms")

    if hasattr(nmgc, 'test_next_move_generator_performance'):
        print("\nRunning built-in performance test...")
        nmgc.test_next_move_generator_performance()


if __name__ == "__main__":
    print("=== Testing move_c module ===")
    test_move_validation()
    print("\n=== Testing move execution ===")
    test_move_execution()
    print("\n=== Testing next move generation ===")
    test_next_move_generation()
    print("\n=== Performance Tests ===")
    test_performance()