import tkinter as tk
from tkinter import messagebox
import math
import time
import copy


BOARD_SIZE = 600
HEX_SIZE = 30
THEME_LIGHT = {"bg": "#F0F0F0", "hex_bg": "#FFFFFF", "hex_outline": "#000000", "text": "#000000", "btn_bg": "#D3D3D3", "btn_fg": "#000000"}
THEME_DARK = {"bg": "#2E2E2E", "hex_bg": "#444444", "hex_outline": "#FFFFFF", "text": "#FFFFFF", "btn_bg": "#555555", "btn_fg": "#FFFFFF"}
THEME_BLUE = {"bg": "#B3D9FF", "hex_bg": "#99CCFF", "hex_outline": "#1E90FF", "text": "#000000", "btn_bg": "#ADD8E6", "btn_fg": "#000000"}
THEME_GREEN = {"bg": "#E6F7E6", "hex_bg": "#99FF99", "hex_outline": "#32CD32", "text": "#000000", "btn_bg": "#90EE90", "btn_fg": "#000000"}
THEME_PURPLE = {"bg": "#F0E6F6", "hex_bg": "#D8B0D8", "hex_outline": "#800080", "text": "#000000", "btn_bg": "#DDA0DD", "btn_fg": "#000000"}
THEME_ORANGE = {"bg": "#FFE6CC", "hex_bg": "#FFD580", "hex_outline": "#FF6600", "text": "#000000", "btn_bg": "#FFD700", "btn_fg": "#000000"}
THEME = THEME_LIGHT

WHITE_MARBLE = "White"
BLACK_MARBLE = "Black"
NO_MARBLE = "Blank"

THEMES = {
    "Light": THEME_LIGHT,
    "Dark": THEME_DARK,
    "Blue": THEME_BLUE,
    "Green": THEME_GREEN,
    "Purple": THEME_PURPLE,
    "Orange": THEME_ORANGE
}

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

BELGIAN_BOARD_INIT = {
    "I5": WHITE_MARBLE, "I6": WHITE_MARBLE, "I7": NO_MARBLE, "I8": BLACK_MARBLE, "I9": BLACK_MARBLE,
    "H4": WHITE_MARBLE, "H5": WHITE_MARBLE, "H6": WHITE_MARBLE, "H7": BLACK_MARBLE, "H8": BLACK_MARBLE, "H9": BLACK_MARBLE,
    "G3": NO_MARBLE, "G4": WHITE_MARBLE, "G5": WHITE_MARBLE, "G6": NO_MARBLE, "G7": BLACK_MARBLE, "G8": BLACK_MARBLE, "G9": NO_MARBLE,
    "F2": NO_MARBLE, "F3": NO_MARBLE, "F4": NO_MARBLE, "F5": NO_MARBLE, "F6": NO_MARBLE, "F7": NO_MARBLE, "F8": NO_MARBLE, "F9": NO_MARBLE,
    "E1": NO_MARBLE, "E2": NO_MARBLE, "E3": NO_MARBLE, "E4": NO_MARBLE, "E5": NO_MARBLE, "E6": NO_MARBLE, "E7": NO_MARBLE, "E8": NO_MARBLE, "E9": NO_MARBLE,
    "D1": NO_MARBLE, "D2": NO_MARBLE, "D3": NO_MARBLE, "D4": NO_MARBLE, "D5": NO_MARBLE, "D6": NO_MARBLE, "D7": NO_MARBLE, "D8": NO_MARBLE,
    "C1": NO_MARBLE, "C2": BLACK_MARBLE, "C3": BLACK_MARBLE, "C4": NO_MARBLE, "C5": WHITE_MARBLE, "C6": WHITE_MARBLE, "C7": NO_MARBLE,
    "B1": BLACK_MARBLE, "B2": BLACK_MARBLE, "B3": BLACK_MARBLE, "B4": WHITE_MARBLE, "B5": WHITE_MARBLE, "B6": WHITE_MARBLE,
    "A1": BLACK_MARBLE, "A2": BLACK_MARBLE, "A3": NO_MARBLE, "A4": WHITE_MARBLE, "A5": WHITE_MARBLE
}

GERMAN_BOARD_INIT = {
    "I5": NO_MARBLE, "I6": NO_MARBLE, "I7": NO_MARBLE, "I8": NO_MARBLE, "I9": NO_MARBLE,
    "H4": WHITE_MARBLE, "H5": WHITE_MARBLE, "H6": NO_MARBLE, "H7": NO_MARBLE, "H8": BLACK_MARBLE, "H9": BLACK_MARBLE,
    "G3": WHITE_MARBLE, "G4": WHITE_MARBLE, "G5": WHITE_MARBLE, "G6": NO_MARBLE, "G7": BLACK_MARBLE, "G8": BLACK_MARBLE, "G9": BLACK_MARBLE,
    "F2": NO_MARBLE, "F3": WHITE_MARBLE, "F4": WHITE_MARBLE, "F5": NO_MARBLE, "F6": NO_MARBLE, "F7": BLACK_MARBLE, "F8": BLACK_MARBLE, "F9": NO_MARBLE,
    "E1": NO_MARBLE, "E2": NO_MARBLE, "E3": NO_MARBLE, "E4": NO_MARBLE, "E5": NO_MARBLE, "E6": NO_MARBLE, "E7": NO_MARBLE, "E8": NO_MARBLE, "E9": NO_MARBLE,
    "D1": NO_MARBLE, "D2": BLACK_MARBLE, "D3": BLACK_MARBLE, "D4": NO_MARBLE, "D5": NO_MARBLE, "D6": WHITE_MARBLE, "D7": WHITE_MARBLE, "D8": NO_MARBLE,
    "C1": BLACK_MARBLE, "C2": BLACK_MARBLE, "C3": BLACK_MARBLE, "C4": NO_MARBLE, "C5": WHITE_MARBLE, "C6": WHITE_MARBLE, "C7": WHITE_MARBLE,
    "B1": BLACK_MARBLE, "B2": BLACK_MARBLE, "B3": NO_MARBLE, "B4": NO_MARBLE, "B5": WHITE_MARBLE, "B6": WHITE_MARBLE,
    "A1": NO_MARBLE, "A2": NO_MARBLE, "A3": NO_MARBLE, "A4": NO_MARBLE, "A5": NO_MARBLE
}

used_board = STANDARD_BOARD_INIT

root = tk.Tk()
root.title("Abalone Game")
root.geometry("1000x800")

current_player = "Black"
move_count = 0
theme_mode = "Light"
is_paused = False
player_times = {"Black": [], "White": []}
start_time = None
pause_time = None


current_board = copy.deepcopy(used_board)

def draw_hexagon(x, y, size, fill_color, outline_color):
    angle = 60
    coords = []
    for i in range(6):
        x_i = x + size * math.cos(math.radians(angle * i))
        y_i = y + size * math.sin(math.radians(angle * i))
        coords.append((x_i, y_i))
    canvas.create_polygon(coords, fill=fill_color, outline=outline_color)

def draw_marble(x, y, size, fill_color, outline_color):
    coords = []
    angle = 60
    x_zero = x - size * 0.9
    y_zero = y + size * math.sin(math.radians(angle))
    x_1 = x + size * 0.9
    y_1 = y - size * math.sin(math.radians(angle))
    coords.append(x_zero)
    coords.append(y_zero)
    coords.append(x_1)
    coords.append(y_1)
    canvas.create_oval(coords, fill=fill_color, outline=outline_color)

def draw_board(board:dict) -> None:
    """
    Draws the game board with marbles

    :param board: a dictionary representing the game board
    """
    canvas.delete("all")
    canvas.config(bg=THEME["bg"])
    rows = {}
    for key in board.keys():
        row_letter = key[0].upper()
        rows.setdefault(row_letter, []).append(key)
    sorted_rows = sorted(rows.keys(), reverse=True)
    center_x, center_y = BOARD_SIZE // 2, BOARD_SIZE // 2
    hex_width = HEX_SIZE * 1.9
    for row_letter in sorted_rows:
        cell_keys = sorted(rows[row_letter], key=lambda k: int(k[1:]))
        num_cells = len(cell_keys)
        row_index = ord("I") - ord(row_letter)
        row_width = num_cells * hex_width
        start_x = center_x - row_width / 2 + hex_width / 2
        start_y = center_y + (row_index - 4) * HEX_SIZE * math.sqrt(3)
        for i, cell_key in enumerate(cell_keys):
            x = start_x + i * hex_width
            y = start_y
            draw_hexagon(x, y, HEX_SIZE, THEME["hex_bg"], THEME["hex_outline"])
            cell_value = board[cell_key]
            if cell_value == BLACK_MARBLE:
                draw_marble(x, y, HEX_SIZE, THEME["hex_outline"], "#000000")
                canvas.create_text(x, y, text=cell_key, fill="#FFFFFF", font=("Arial", 10, "bold"))
            elif cell_value == WHITE_MARBLE:
                draw_marble(x, y, HEX_SIZE, THEME["hex_bg"], "#000000")
                canvas.create_text(x, y, text=cell_key, fill=THEME["text"], font=("Arial", 10, "bold"))
            else:
                canvas.create_text(x, y, text=cell_key, fill=THEME["text"], font=("Arial", 10, "bold"))


def update_turn_display():
    if current_player == "Black":
        turn_label.config(
            text=f"Current Player: {current_player}",
            bg="black",
            fg="white",
            relief="solid",
            bd=2,
            padx=10,
            pady=5
        )
    else:
        turn_label.config(
            text=f"Current Player: {current_player}",
            bg="white",
            fg="black",
            relief="solid",
            bd=2,
            padx=10,
            pady=5
        )

def switch_theme(selected_theme=None):
    global THEME, theme_mode
    if selected_theme:
        THEME = THEMES[selected_theme]
        theme_mode = selected_theme
    else:
        theme_list = list(THEMES.keys())
        current_index = theme_list.index(theme_mode)
        next_index = (current_index + 1) % len(theme_list)
        theme_mode = theme_list[next_index]
        THEME = THEMES[theme_mode]
    draw_board(current_board)
    update_turn_display()
    pause_button.config(
        bg="#FF6347" if is_paused else THEME.get("btn_bg", "#D3D3D3"),
        fg="white" if is_paused else THEME.get("btn_fg", "#000000"),
        activebackground="#FF6347" if is_paused else THEME.get("btn_bg", "#D3D3D3"),
        activeforeground="white" if is_paused else THEME.get("btn_fg", "#000000")
    )

def change_theme():
    switch_theme()

def reset_game():
    global current_player, move_count, player_times, start_time, is_paused, pause_time, current_board
    current_player = "Black"
    move_count = 0
    player_times = {"Black": [], "White": []}
    start_time = None
    is_paused = False
    pause_time = None
    move_counter_label.config(text=f"Moves: {move_count}")
    timer_label.config(text="Time: 0s")
    update_turn_display()
    current_board = copy.deepcopy(STANDARD_BOARD_INIT)
    draw_board(current_board)
    start_timer()

def toggle_pause():
    global is_paused, pause_time, start_time
    if is_paused:
        is_paused = False
        pause_button.config(
            text="Pause Game",
            bg=THEME["btn_bg"],
            fg=THEME["btn_fg"],
            activebackground=THEME["btn_bg"],
            activeforeground=THEME["btn_fg"]
        )
        if pause_time is not None:
            start_time += time.time() - pause_time
            pause_time = None
    else:
        is_paused = True
        pause_button.config(
            text="Resume Game",
            bg="#FF6347",
            fg="white",
            activebackground="#FF6347",
            activeforeground="white"
        )
        pause_time = time.time()

def start_timer():
    global start_time
    if not is_paused:
        if start_time is None:
            start_time = time.time()
        elapsed_time = time.time() - start_time
        timer_label.config(text=f"Time: {int(elapsed_time)}s")
    root.after(1000, start_timer)

def end_turn():
    global current_player, move_count, start_time, is_paused, pause_time
    if start_time is not None:
        elapsed_time = time.time() - start_time
        player_times[current_player].append(elapsed_time)
        start_time = None
    move_count += 1
    move_counter_label.config(text=f"Moves: {move_count}")
    current_player = "White" if current_player == "Black" else "Black"
    is_paused = False
    pause_time = None
    update_turn_display()
    start_timer()

def start_game():
    start_frame.pack_forget()
    top_frame.pack(fill="x", pady=5)
    canvas.pack()
    bottom_frame.pack(fill="x", pady=5)
    draw_board(current_board)
    start_timer()
    command_frame.pack(side="bottom", fill="x", pady=10)

def exit_game():
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        root.destroy()

start_frame = tk.Frame(root, bg=THEME["bg"])
start_frame.pack(pady=100)

start_label = tk.Label(start_frame, text="Welcome to Abalone!", font=("Arial", 24, "bold"), bg=THEME["bg"], fg=THEME["text"])
start_label.pack(pady=20)

start_button = tk.Button(start_frame, text="Start Game", command=start_game, font=("Arial", 14), bg=THEME["btn_bg"], fg=THEME["btn_fg"], relief="raised", bd=2)
start_button.pack(pady=10)

exit_button = tk.Button(start_frame, text="Exit", command=exit_game, font=("Arial", 14), bg=THEME["btn_bg"], fg=THEME["btn_fg"], relief="raised", bd=2)
exit_button.pack(pady=10)

top_frame = tk.Frame(root, bg=THEME["bg"])
timer_label = tk.Label(top_frame, text="Time: 0s", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"])
reset_button = tk.Button(top_frame, text="Reset Game", command=reset_game, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
theme_button = tk.Button(top_frame, text="Switch Theme", command=switch_theme, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
pause_button = tk.Button(top_frame, text="Pause Game", command=toggle_pause, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
end_turn_button = tk.Button(top_frame, text="End Turn", command=end_turn, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)

timer_label.pack(side="left", padx=10)
reset_button.pack(side="left", padx=10)
theme_button.pack(side="left", padx=10)
pause_button.pack(side="left", padx=10)
end_turn_button.pack(side="left", padx=10)

theme_label = tk.Label(top_frame, text="Choose Theme", font=("Arial", 12))
theme_label.pack(side="left", padx=10)

theme_options = list(THEMES.keys())

theme_button.config(command=change_theme)
theme_button.pack_forget()
theme_dropdown = tk.OptionMenu(top_frame, tk.StringVar(value=theme_mode), *theme_options, command=switch_theme)
theme_dropdown.pack(side="left", padx=10)

start_button.config(bg="#4CAF50", fg="white", activebackground="#45a049", activeforeground="white")
exit_button.config(bg="#f44336", fg="white", activebackground="#e7352e", activeforeground="white")

reset_button.config(bg="#008CBA", fg="white", activebackground="#007bb5", activeforeground="white")
theme_button.config(bg="#FF9800", fg="white", activebackground="#f57c00", activeforeground="white")
pause_button.config(bg="#9C27B0", fg="white", activebackground="#8e24aa", activeforeground="white")
end_turn_button.config(bg="#2196F3", fg="white", activebackground="#1976d2", activeforeground="white")

bottom_frame = tk.Frame(root, bg=THEME["bg"])
turn_label = tk.Label(
    bottom_frame,
    text=f"Current Player: {current_player}",
    font=("Arial", 14),
    bg=THEME["bg"],
    fg=THEME["text"],
    relief="solid",
    bd=2,
    padx=10,
    pady=5
)
turn_label.pack(side="left", padx=10)
move_counter_label = tk.Label(bottom_frame, text=f"Moves: {move_count}", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"])

turn_label.pack(side="left", padx=10)
move_counter_label.pack(side="left", padx=10)

canvas = tk.Canvas(root, width=BOARD_SIZE, height=BOARD_SIZE, bg=THEME["bg"])


class SomeError(Exception):
    """
    temporary error handling for all
    """
    pass

def next_row(row: str) -> str:
    """
    Returns the next alphabet

    :param row: a string of alphabet
    :return: the next alphabet
    """
    return chr(ord(row) + 1)

def prev_row(row: str) -> str:
    """
    Returns the previous alphabet

    :param row: a string of alphabet
    :return: the previous alphabet
    """
    return chr(ord(row) - 1)

def parse_move_input(move_str: str) -> tuple[list, list]:
    """
    Parses the move command input into source and destination lists

    :param move_str: moving command following the rule
    :return: a tuple containing the source list and destination list
    """
    move_str = move_str.replace(" ", "")
    parts = move_str.split("],[")
    part1 = parts[0].lstrip("[")
    part2 = parts[1].rstrip("]")
    source_list = part1.split(",")
    dest_list = part2.split(",")
    return source_list, dest_list

def get_move_direction(source: str, dest: str) -> str:
    """
    Determines the move direction based on the source and destination coordinates

    :param source: the starting coordinate
    :param dest: the destination coordinate
    :return: a string of destination coordinate
    :raises Exception: will be updated soon
    """
    s_row, s_col = source[0], int(source[1:])
    d_row, d_col = dest[0], int(dest[1:])
    if next_row(s_row) == d_row and s_col == d_col:
        return "upper_left"
    elif next_row(s_row) == d_row and s_col + 1 == d_col:
        return "upper_right"
    elif s_row == d_row and s_col - 1 == d_col:
        return "left"
    elif s_row == d_row and s_col + 1 == d_col:
        return "right"
    elif prev_row(s_row) == d_row and s_col - 1 == d_col:
        return "down_left"
    elif prev_row(s_row) == d_row and s_col == d_col:
        return "down_right"
    else:
        raise SomeError("to be implemented soon")

def transform_coordinate(coord: str, direction: str) -> str:
    """
    Transforms a board coordinate in the given direction

    :param coord: starting coordinate
    :param direction: direction of destination
    :return: new coordinate after moving
    :raises Exception: will be implemented soon
    """
    row = coord[0]
    col = int(coord[1:])
    if direction == "upper_left":
        new_row = next_row(row)
        new_col = col
    elif direction == "upper_right":
        new_row = next_row(row)
        new_col = col + 1
    elif direction == "left":
        new_row = row
        new_col = col - 1
    elif direction == "right":
        new_row = row
        new_col = col + 1
    elif direction == "down_left":
        new_row = prev_row(row)
        new_col = col - 1
    elif direction == "down_right":
        new_row = prev_row(row)
        new_col = col
    else:
        raise SomeError("to be implemented soon")
    return f"{new_row}{new_col}"


def move_marbles_cmd(marble_coords: list, direction: str) -> bool:
    """
    Moves the marbles on the board to the specified direction

    :param marble_coords: list of coordinates of the marble(or marbles)
    :param direction: The direction to move
    :return: True if the move is successful
    """
    global current_board
    player = current_board[marble_coords[0]]
    new_coords = [transform_coordinate(coord, direction) for coord in marble_coords]
    for coord in marble_coords:
        current_board[coord] = NO_MARBLE
    for new_coord in new_coords:
        current_board[new_coord] = player
    return True

def process_move_command() -> None:
    """
    Processes the user's move command from the input field

    Parses the move, determines the direction, moves the marbles,
    and redraws the board
    """
    move_text = move_entry.get()
    source_list, dest_list = parse_move_input(move_text)
    direction = get_move_direction(source_list[0], dest_list[0])
    if move_marbles_cmd(source_list, direction):
        draw_board(current_board)
    else:
        raise SomeError("to be implemented soon")
    move_entry.delete(0, tk.END)

command_frame = tk.Frame(root, bg=THEME["bg"])
move_label = tk.Label(command_frame, text="Enter your move:", bg=THEME["bg"], fg=THEME["text"], font=("Arial", 12))
move_label.pack(pady=5)
entry_frame = tk.Frame(command_frame, bg=THEME["bg"])
entry_frame.pack(pady=3)
move_entry = tk.Entry(entry_frame, width=50, font=("Arial", 12))
move_entry.pack(side="left", padx=5)
move_button = tk.Button(entry_frame, text="Move", command=process_move_command, font=("Arial", 12), bg=THEME["btn_bg"], fg=THEME["btn_fg"])
move_button.pack(side="left", padx=5)

root.mainloop()
