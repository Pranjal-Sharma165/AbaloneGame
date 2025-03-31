class BoardIO:
    WHITE_MARBLE = "#D9D9D9"
    BLACK_MARBLE = "#8A8A8A"
    NO_MARBLE = "Blank"

    BOARD_SAMPLE = {
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

    @staticmethod
    def export_current_board_to_text(board: dict, current_player: str, filename: str) -> None:
        """
        Exports the current board state and current player's turn to a text file.

        :param board: The board state as a dictionary with keys as coordinates and values as marble color or "Blank".
        :param current_player: The current player's color ("Black" or "White").
        :param filename: The filename where the board configuration will be saved.
        """
        turn_letter = "b" if current_player.lower() == "black" else "w"
        black_coords = [coord for coord, cell in board.items() if cell == BoardIO.BLACK_MARBLE]
        white_coords = [coord for coord, cell in board.items() if cell == BoardIO.WHITE_MARBLE]
        sort_key = lambda c: (c[0], int(c[1:]))
        black_coords.sort(key=sort_key)
        white_coords.sort(key=sort_key)
        token_line = ",".join([f"{c}b" for c in black_coords] + [f"{c}w" for c in white_coords])
        with open(filename, "w") as f:
            f.write(turn_letter + "\n")
            f.write(token_line)

    @staticmethod
    def import_current_text_to_board(filename: str) -> tuple:
        """
        Imports a board configuration and the current player's turn from a text file.

        :param filename: The filename from which to load the board configuration.
        :return: A tuple (board, turn) where board is a dictionary representing the board state, and turn is a string ("BLACK" or "WHITE").
        """
        board = {coord: "Blank" for coord in BoardIO.BOARD_SAMPLE.keys()}

        try:
            with open(filename, "r") as f:
                lines = f.read().splitlines()

            if len(lines) < 2:
                print(f"Warning: File {filename} has less than 2 lines")
                return board, "BLACK"

            turn_line = lines[0].strip().lower()
            turn = "BLACK" if turn_line == "b" else "WHITE" if turn_line == "w" else "BLACK"

            tokens_line = lines[1].strip()
            for token in tokens_line.split(","):
                if len(token) < 3:
                    continue

                coord, color_token = token[:-1], token[-1].lower()

                if color_token == "b":
                    board[coord] = "#8A8A8A"
                elif color_token == "w":
                    board[coord] = "#D9D9D9"

            return board, turn

        except Exception as e:
            print(f"Error loading board from {filename}: {e}")
            return board, "BLACK"







