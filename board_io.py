WHITE_MARBLE = "#D9D9D9"
BLACK_MARBLE = "#8A8A8A"
NO_MARBLE = "Blank"

STANDARD_BOARD_INIT = {
    "I5": WHITE_MARBLE, "I6": WHITE_MARBLE, "I7": WHITE_MARBLE, "I8": WHITE_MARBLE, "I9": WHITE_MARBLE,
    "H4": WHITE_MARBLE, "H5": WHITE_MARBLE, "H6": WHITE_MARBLE, "H7": WHITE_MARBLE, "H8": WHITE_MARBLE, "H9": WHITE_MARBLE,
    "G3": NO_MARBLE, "G4": NO_MARBLE, "G5": WHITE_MARBLE, "G6": WHITE_MARBLE, "G7": WHITE_MARBLE, "G8": NO_MARBLE, "G9": NO_MARBLE,
    "F2": NO_MARBLE, "F3": NO_MARBLE, "F4": NO_MARBLE, "F5": NO_MARBLE, "F6": NO_MARBLE, "F7": NO_MARBLE, "F8": NO_MARBLE, "F9": NO_MARBLE,
    "E1": NO_MARBLE, "E2": NO_MARBLE, "E3": NO_MARBLE, "E4": NO_MARBLE, "E5": NO_MARBLE, "E6": NO_MARBLE, "E7": NO_MARBLE, "E8": NO_MARBLE, "E9": NO_MARBLE,
    "D1": NO_MARBLE, "D2": NO_MARBLE, "D3": NO_MARBLE, "D4": NO_MARBLE, "D5": NO_MARBLE, "D6": NO_MARBLE, "D7": NO_MARBLE, "D8": NO_MARBLE,
    "C1": NO_MARBLE, "C2": NO_MARBLE, "C3": BLACK_MARBLE, "C4": BLACK_MARBLE, "C5": BLACK_MARBLE, "C6": NO_MARBLE, "C7": NO_MARBLE,
    "B1": BLACK_MARBLE, "B2": BLACK_MARBLE, "B3": BLACK_MARBLE, "B4": BLACK_MARBLE, "B5": BLACK_MARBLE, "B6": BLACK_MARBLE,
    "A1": BLACK_MARBLE, "A2": BLACK_MARBLE, "A3": BLACK_MARBLE, "A4": BLACK_MARBLE, "A5": BLACK_MARBLE
}

def export_current_board_to_text(board: dict, current_player: str, filename: str) -> None:
    turn_letter = "b" if current_player.lower() == "black" else "w"
    black_coords = []
    white_coords = []
    for coord, cell in board.items():
        if cell == "Blank":
            continue
        if cell == "#8A8A8A":
            black_coords.append(coord)
        elif cell == "#D9D9D9":
            white_coords.append(coord)

    black_coords.sort(key=lambda c: (c[0], int(c[1:])))
    white_coords.sort(key=lambda c: (c[0], int(c[1:])))
    black_tokens = [f"{c}b" for c in black_coords]
    white_tokens = [f"{c}w" for c in white_coords]
    token_line = ",".join(black_tokens + white_tokens)
    with open(filename, "w") as f:
        f.write(turn_letter + "\n")
        f.write(token_line)


def import_current_text_to_board(filename: str) -> tuple[dict, str]:


    board = {coord: "Blank" for coord in STANDARD_BOARD_INIT.keys()}

    with open(filename, "r") as f:
        lines = f.read().splitlines()
    turn_line = lines[0].strip()
    tokens_line = lines[1].strip()
    if turn_line == "b":
        turn = "Black"
    elif turn_line == "w":
        turn = "White"
    else:
        turn = "Black"

    tokens = tokens_line.split(",")
    for token in tokens:
        if len(token) < 2:
            continue
        coord = token[:-1]
        color_token = token[-1]
        if color_token == "b":
            board[coord] = "#8A8A8A"
        elif color_token == "w":
            board[coord] = "#D9D9D9"
    return board, turn

if __name__ == "__main__":
    #test
    test_01_board, test_01_turn = import_current_text_to_board("output/Test1.input")
    test_02_board, test_02_turn = import_current_text_to_board("output/Test2.input")
    print(test_01_board)
    print(test_01_turn)
