

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
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    turn_line = lines[0].strip()
    tokens_line = lines[1].strip()
    if turn_line == "b":
        turn = "Black"
    elif turn_line == "w":
        turn = "White"
    tokens = tokens_line.split(",")
    board = {}
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
    # test
    test_01_board, test_01_turn = import_current_text_to_board("./part2_test/Test1.input")
    test_02_board, test_02_turn = import_current_text_to_board("./part2_test/Test2.input")


