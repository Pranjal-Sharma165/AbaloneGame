import time
import move_cy as mc
import next_move_generator_cy as nmgc
from IO import BoardIO

STANDARD_BOARD_INIT = {
    "I5": "#D9D9D9", "I6": "#D9D9D9", "I7": "#D9D9D9", "I8": "#D9D9D9", "I9": "#D9D9D9",
    "H4": "#D9D9D9", "H5": "#D9D9D9", "H6": "#D9D9D9", "H7": "#D9D9D9", "H8": "#D9D9D9", "H9": "#D9D9D9",
    "G3": "Blank", "G4": "Blank", "G5": "#D9D9D9", "G6": "#D9D9D9", "G7": "#D9D9D9", "G8": "Blank", "G9": "Blank",
    "F2": "Blank", "F3": "Blank", "F4": "Blank", "F5": "Blank", "F6": "Blank", "F7": "Blank", "F8": "Blank", "F9": "Blank",
    "E1": "Blank", "E2": "Blank", "E3": "Blank", "E4": "Blank", "E5": "Blank", "E6": "Blank", "E7": "Blank", "E8": "Blank", "E9": "Blank",
    "D1": "Blank", "D2": "Blank", "D3": "Blank", "D4": "Blank", "D5": "Blank", "D6": "Blank", "D7": "Blank", "D8": "Blank",
    "C1": "Blank", "C2": "Blank", "C3": "#8A8A8A", "C4": "#8A8A8A", "C5": "#8A8A8A", "C6": "Blank", "C7": "Blank",
    "B1": "#8A8A8A", "B2": "#8A8A8A", "B3": "#8A8A8A", "B4": "#8A8A8A", "B5": "#8A8A8A", "B6": "#8A8A8A",
    "A1": "#8A8A8A", "A2": "#8A8A8A", "A3": "#8A8A8A", "A4": "#8A8A8A", "A5": "#8A8A8A"
}



start = time.time()
board = mc.convert_board_format(BoardIO.import_current_text_to_board("./output/Test5.input")[0])
end = time.time()
print(f"Board conversion: {(end-start)*10000:.3f}ms")

start = time.time()
next_moves = nmgc.generate_all_next_moves(board, BoardIO.import_current_text_to_board("./output/Test5.input")[1])
end = time.time()
print(f"Generated {len(next_moves)} next moves in {end-start:.6f}s")

source_coords = [[1, 1]]
dest_coords = [[2, 1]]
start = time.time()
for _ in range(10000):
    mc.move_validation(source_coords, dest_coords, board, "BLACK")
end = time.time()
print(f"1000 move validations: {(end-start)*10000:.3f}ms")