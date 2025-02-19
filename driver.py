import tkinter as tk
from tkinter import messagebox, ttk
import math
import time
import copy
import re

# Constant values for board size and hexagon size
BOARD_SIZE = 500
HEX_SIZE = 30

# Define themes with background colors and element color schemes
THEME_LIGHT = {"bg": "#D3D3D3", "hex_bg": "#E0E0E0", "hex_outline": "#000000", "text": "#000000", "btn_bg": "#B0B0B0", "btn_fg": "#000000"}
THEME_DARK = {"bg": "#2E2E2E", "hex_bg": "#444444", "hex_outline": "#FFFFFF", "text": "#FFFFFF", "btn_bg": "#555555", "btn_fg": "#FFFFFF"}
THEME_BLUE = {"bg": "#A0C2FF", "hex_bg": "#80B3FF", "hex_outline": "#1E90FF", "text": "#000000", "btn_bg": "#A1C8E0", "btn_fg": "#000000"}
THEME_GREEN = {"bg": "#CCE6CC", "hex_bg": "#80E680", "hex_outline": "#32CD32", "text": "#000000", "btn_bg": "#80D68B", "btn_fg": "#000000"}
THEME_PURPLE = {"bg": "#D6B7D6", "hex_bg": "#C29AC7", "hex_outline": "#800080", "text": "#000000", "btn_bg": "#D29BE3", "btn_fg": "#000000"}
THEME_BROWN = {"bg": "#D2B48C", "hex_bg": "#C19A6B", "hex_outline": "#8B5A2B", "text": "#000000", "btn_bg": "#A67B5B", "btn_fg": "#000000"}

# Default theme selected (can be changed dynamically)
THEME = THEME_LIGHT

# Define marble colors for white, black, and empty spaces
WHITE_MARBLE = "#D9D9D9"
BLACK_MARBLE = "#8A8A8A"
NO_MARBLE = "Blank"

# Store all themes in a dictionary for easy switching
THEMES = {
    "Light": THEME_LIGHT,
    "Dark": THEME_DARK,
    "Blue": THEME_BLUE,
    "Green": THEME_GREEN,
    "Purple": THEME_PURPLE,
    "Brown": THEME_BROWN
}

# Initial board configurations for various game types - Standard
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
}  # this represents the board with the classic game starting positions

# Initial board configurations for various game types - Belgian
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
}  # this represents the board with the belgian daisy game starting positions

# Initial board configurations for various game types - German
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
}  #this represents the board with the german daisy game starting positions

used_board = STANDARD_BOARD_INIT  # used in specifying the game board to use for the game
current_mode = "Player VS Computer"

# Set up the Tkinter root window for the game
root = tk.Tk()
root.title("Abalone Game")
root.geometry("1000x800")

# Global variables for tracking the game state
current_player = "Black"
move_counts = {"Black": 0, "White": 0}
theme_mode = "Light"
is_paused = False
player_times = {"Black": [], "White": []}
start_time = None
pause_time = None

# Create a deep copy of the board to avoid modifying the original starting setup
current_board = copy.deepcopy(used_board)

def setup_board_layout(event):
    """
    Sets up the board layout based on the dropdown menu selection at the start page.
    """
    global current_board, used_board
    selected_board_layout=board_layout_box.get()
    if selected_board_layout == "Standard":
        used_board = STANDARD_BOARD_INIT
        current_board = copy.deepcopy(used_board)
    elif selected_board_layout == "German Daisy":
        used_board = GERMAN_BOARD_INIT
        current_board = copy.deepcopy(used_board)
    else:
        current_board = BELGIAN_BOARD_INIT
        used_board = BELGIAN_BOARD_INIT
        current_board = copy.deepcopy(used_board)

def setup_game_mode(event):
    """
     Sets up the game mode based on the dropdown menu selection at the start page.
    """
    global current_mode
    current_mode = game_mode_box.get()
    current_mode_label.config(text=current_mode)



def draw_hexagon(x, y, size, fill_color, outline_color):
    """
    Draws a hexagon (used for board cells)

    :param x: an int representing the x-coordinate of the center of the hexagon
    :param y: an int representing the y-coordinate of the center of the hexagon
    :param size: an int representing the diameter of the hexagon
    :param fill_color: a string of hexadecimal representing the hexagon's fill color
    :param outline_color: a string of hexadecimal representing the hexagon's outline color
    """
    angle = 60  # Angle between vertices of the hexagon
    coords = [] # List to hold hexagon vertices
    for i in range(6):
        x_i = x + size * math.cos(math.radians(angle * i))
        y_i = y + size * math.sin(math.radians(angle * i))
        coords.append((x_i, y_i))
    canvas.create_polygon(coords, fill=fill_color, outline=outline_color)


def draw_marble(x, y, size, fill_color, outline_color):
    """
    Draws a circle with a fill and outline to represent a marble.

    :param x: an int representing the x-coordinate of the center of the circle
    :param y: an int representing the y-coordinate of the center of the circle
    :param size: an int representing the diameter of the circle
    :param fill_color: a string of hexadecimal representing the circle's fill color
    :param outline_color: a string of hexadecimal representing the circle's outline color
    """
    radius = size * 0.9
    canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill_color, outline=outline_color)

def draw_board(board:dict) -> None:
    """
    Draws the game board based on a given board state as a dictionary.

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
                draw_marble(x, y, HEX_SIZE, THEME["hex_outline"], "#000000")  # Black marble
                text_color = "#e65252"  # White text for visibility
            elif cell_value == WHITE_MARBLE:
                draw_marble(x, y, HEX_SIZE, THEME["hex_bg"], "#000000")  # White marble
                text_color = "#1e1b26"  # Black text for visibility
            else:
                text_color = THEME["text"]  # Default theme text color for empty spaces

            canvas.create_text(x, y, text=cell_key, fill=text_color, font=("Arial", 10, "bold"))


def update_turn_display():
    """
        Updates the turn display label to indicate the current player and adjusts the label's background
        and foreground colors accordingly.
    """
    player_colors = {
        "Black": {"bg": "black", "fg": "white"},
        "White": {"bg": "white", "fg": "black"}
    }
    colors = player_colors.get(current_player, {"bg": "black", "fg": "white"})
    turn_label.config(
        text=f"Current Player: {current_player}",
        bg=colors["bg"],
        fg=colors["fg"],
        relief="solid",
        bd=2,
        padx=10,
        pady=5
    )


def switch_theme(selected_theme=None):
    """
       Switches the game theme to the selected theme or cycles through available themes if none is selected.
       Updates the board and buttons accordingly.
    """
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
    configure_button(pause_button, "#FF6347" if is_paused else THEME.get("btn_bg", "#D3D3D3"),
                     "white" if is_paused else THEME.get("btn_fg", "#000000"))


def change_theme():
    """
        Calls switch_theme() to cycle through available themes.
    """
    switch_theme()

def reset_game_state():
    """
       Resets the game state variables, including player turns, move count, timers, and board state.
       Refreshes the UI elements and redraws the board.
    """
    global current_player, move_count, player_times, start_time, is_paused, pause_time, current_board, used_board
    current_player = "Black"
    move_count = 0
    player_times = {"Black": [], "White": []}
    start_time = None
    is_paused = False
    pause_time = None
    move_counter_label.config(text=f"Moves: {move_count}")
    timer_label.config(text="Time: 0s")
    update_turn_display()
    canvas.delete("all")
    current_board = copy.deepcopy(used_board)
    draw_board(current_board)

def reset_game():
    """
        Resets the game state and restarts the timer.
    """
    reset_game_state()
    start_timer()

def stop_game():
    """
       Stops the game, resets the state, and hides game elements to return to the start screen.
    """
    reset_game_state()
    top_frame.pack_forget()
    mode_frame.pack_forget()
    score_frame.pack_forget()
    canvas.pack_forget()
    output_frame.pack_forget()
    log_frame.pack_forget()
    bottom_frame.pack_forget()
    command_frame.pack_forget()
    start_frame.pack(pady=100)

def toggle_pause():
    """
        Toggles the pause state of the game. Updates button appearance and game timer accordingly.
    """
    global is_paused, pause_time, start_time
    if is_paused:
        is_paused = False
        configure_button(pause_button, THEME["btn_bg"], THEME["btn_fg"])
        pause_button.config(text="Pause Game")
        if pause_time is not None:
            start_time += time.time() - pause_time
            pause_time = None
    else:
        is_paused = True
        configure_button(pause_button, "#FF6347", "white")
        pause_button.config(text="Resume Game")
        pause_time = time.time()


def start_timer():
    """
        Starts or updates the game timer if the game is not paused.
    """
    global start_time
    if not is_paused:
        if start_time is None:
            start_time = time.time()
        elapsed_time = time.time() - start_time
        timer_label.config(text=f"Time: {int(elapsed_time)}s")
    root.after(1000, start_timer)


def end_turn():
    """
        Ends the current player's turn, updates the move count, and switches the player.
        Also manages timing for each player's turns.
    """
    global current_player, move_count, start_time, is_paused, pause_time
    if start_time is not None:
        elapsed_time = time.time() - start_time
        player_times[current_player].append(elapsed_time)
        start_time = None
    move_counts[current_player] += 1
    move_counter_label.config(text=f"Moves: {move_counts}")
    current_player = "White" if current_player == "Black" else "Black"
    is_paused = False
    pause_time = None
    update_turn_display()
    start_timer()


def start_game():
    """
        Initializes and starts the game by displaying all UI elements and starting the timer.
    """
    start_frame.pack_forget()
    top_frame.pack(fill="x", pady=5)
    mode_frame.pack(fill="x", pady=5)
    score_frame.pack(side="top")
    canvas.pack()
    output_frame.pack(side="right", padx=20, pady=(0, 90))
    log_frame.pack(anchor="e")
    bottom_frame.pack(fill="x", pady=5)
    draw_board(current_board)
    start_timer()
    command_frame.pack(side="bottom", fill="x", pady=10)


def exit_game():
    """
        Prompts the user for confirmation before exiting the game.
    """
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        root.destroy()

def configure_button(button, bg_color, fg_color="white", active_bg=None, active_fg="white"):
    """
        Configures a button's appearance including background, foreground, and active colors.
    """
    active_bg = active_bg if active_bg else bg_color
    button.config(bg=bg_color, fg=fg_color, activebackground=active_bg, activeforeground=active_fg)


def undo_move():
    """
    Undo the last move when button is clicked.

    :return: None
    """
    global used_board, current_board
    current_board = copy.deepcopy(used_board)
    draw_board(current_board)


def display_ai_move_log():
    """
    Displays the complete log history for AI suggested moves.
    """
    move_window = tk.Toplevel()
    move_window.title("AI Move History Log")
    move_window.geometry("400x300")

    ai_move_label = (tk.Label(move_window, text="AI move log displayed here"))
    ai_move_label.pack()


def display_turn_duration_log():
    """
    Displays the complete log history for turn durations, including total aggregate time.
    """
    time_window = tk.Toplevel()
    time_window.title("Turn Duration History Log")
    time_window.geometry("400x300")

    time_duration_label = (tk.Label(time_window, text="Turn duration log displayed here"))
    time_duration_label.pack()


start_frame = tk.Frame(root, bg=THEME["bg"])
start_frame.pack(pady=100)

start_label = tk.Label(start_frame, text="Welcome to Abalone!", font=("Arial", 24, "bold"), bg=THEME["bg"], fg=THEME["text"])
start_label.pack(pady=20)

# Creating board layout label
board_layout_label = tk.Label(start_frame, text="Board Layout: ")
board_layout_label.pack(pady=10)

# Creating board layout box
board_layout_box = ttk.Combobox(start_frame, state = "readonly", values = ["Standard", "German Daisy", "Belgin Daisy"])
board_layout_box.pack(pady=5)
board_layout_box.set("Standard")

# Calling setup_board_layout function to change the board layout
board_layout_box.bind("<<ComboboxSelected>>", setup_board_layout)

# Creating game mode label
game_mode_label = tk.Label(start_frame, text="Game Mode: ")
game_mode_label.pack(pady=10)

#Creating game mode box
game_mode_box = ttk.Combobox(start_frame, state = "readonly", values = ["Computer VS Player", "Player VS Player", "Computer VS Computer"])
game_mode_box.pack(pady=5)
game_mode_box.set("Computer VS Player")

#Calling setup_game_mode function to change the game mode
game_mode_box.bind("<<ComboboxSelected>>", setup_game_mode)

start_button = tk.Button(start_frame, text="Start Game", command=start_game, font=("Arial", 14), bg=THEME["btn_bg"], fg=THEME["btn_fg"], relief="raised", bd=2)
start_button.pack(pady=10)

exit_button = tk.Button(start_frame, text="Exit", command=exit_game, font=("Arial", 14), bg=THEME["btn_bg"], fg=THEME["btn_fg"], relief="raised", bd=2)
exit_button.pack(pady=10)

top_frame = tk.Frame(root, bg=THEME["bg"])
timer_label = tk.Label(top_frame, text="Time: 0s", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"])
reset_button = tk.Button(top_frame, text="Reset Game", command=reset_game, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
theme_button = tk.Button(top_frame, text="Switch Theme", command=switch_theme, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
pause_button = tk.Button(top_frame, text="Pause Game", command=toggle_pause, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
undo_button = tk.Button(top_frame, text="Undo Move", command=undo_move, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
end_turn_button = tk.Button(top_frame, text="End Turn", command=end_turn, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
stop_button = tk.Button(top_frame, text="Stop Game", command=stop_game, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)


# Scoreboard UI that displays the remaining marbles for each player
score_frame = tk.Frame(root, bg=THEME["bg"])
white_score_label = tk.Label(score_frame, text="White marbles lost: 0", font=("Arial", 12, "bold"), bg=THEME["bg"], fg=THEME["text"])
white_score_label.pack(side="left", padx=(0,260))
black_score_label = tk.Label(score_frame, text="Black marbles lost: 0", font=("Arial", 12, "bold"), bg=THEME["bg"], fg=THEME["text"])
black_score_label.pack(side="right", padx=(260,0))


timer_label.pack(side="left", padx=10)
reset_button.pack(side="left", padx=10)
theme_button.pack(side="left", padx=10)
pause_button.pack(side="left", padx=10)
undo_button.pack(side="left", padx=10)
end_turn_button.pack(side="left", padx=10)
stop_button.pack(side="left", padx=10)

theme_label = tk.Label(top_frame, text="Choose Theme", font=("Arial", 12))
theme_label.pack(side="left", padx=10)

theme_options = list(THEMES.keys())

theme_button.config(command=change_theme)
theme_button.pack_forget()
theme_dropdown = tk.OptionMenu(top_frame, tk.StringVar(value=theme_mode), *theme_options, command=switch_theme)
theme_dropdown.pack(side="left", padx=10)

configure_button(start_button, "#4CAF50")
configure_button(exit_button, "#f44336")
configure_button(reset_button, "#008CBA")
configure_button(theme_button, "#FF9800")
configure_button(pause_button, "#9C27B0")
configure_button(undo_button, "#eb6e34")
configure_button(end_turn_button, "#2196F3")
configure_button(stop_button, "#FF0000")

# Output box UI that displays last turn duration, previous move, and suggested next move
output_frame = tk.Frame(root, bg=THEME["bg"], bd=5, relief="solid")

move_duration_label = tk.Label(output_frame, text="Duration of last turn: 00:00:36 seconds", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"], anchor="w", justify="left")
move_duration_label.pack(side="top", padx=10, fill="both")
prev_move_label = tk.Label(output_frame, text="Previous move: c3c4c5, d3d4d5", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"], anchor="w", justify="left")
prev_move_label.pack(side="top", padx=10, fill="both")
next_move_label = tk.Label(output_frame, text="Next move: b3a3, c3b3", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"], anchor="w", justify="left")
next_move_label.pack(side="top", padx=10, fill="both")

# Log button UI that includes two buttons to display turn duration and AI move history logs
log_frame = tk.Frame(root, bg=THEME["bg"])

move_history_button = tk.Button(log_frame, text="AI Move History", command=display_ai_move_log, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
move_history_button.pack(side="left", padx=(0,10))
time_history_button = tk.Button(log_frame, text="Turn Duration History", command=display_turn_duration_log, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
time_history_button.pack(side="left")

# Adding label to show game mode
mode_frame = tk.Frame(root, bg=THEME["bg"])
current_mode_label = tk.Label(mode_frame, text=current_mode, font=("Arial", 20), bg=THEME["bg"], fg=THEME["text"], relief="solid", bd=2, padx=10, pady=5)
current_mode_label.pack(padx=5)

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
move_counter_label = tk.Label(bottom_frame, text=f"Moves: {move_counts}", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"])

turn_label.pack(side="left", padx=10)
move_counter_label.pack(side="left", padx=10)

canvas = tk.Canvas(root, width=BOARD_SIZE, height=BOARD_SIZE, bg=THEME["bg"])


class MoveError(Exception):
    """Base exception for all move-related errors."""
    pass

class MoveFormatError(MoveError):
    """Raised when the move command format is incorrect."""
    pass

class InvalidMarbleError(MoveError):
    """Raised when trying to move a marble that does not belong to the current player."""
    pass

class InvalidDirectionError(MoveError):
    """Raised when the direction of the move is invalid or inconsistent."""
    pass

class BoardBoundaryError(MoveError):
    """Raised when the move goes out of board boundaries."""
    pass

class OverlapError(MoveError):
    """Raised when marbles would overlap after a move."""
    pass



def next_row(row: str) -> str:
    """
    Returns the next alphabetical letter for the next row of the game board.

    :param row: a string of the alphabet row
    :return: a character for the next alphabetical letter
    """
    return chr(ord(row) + 1)


def prev_row(row: str) -> str:
    """
    Returns the previous alphabetical letter for the previous row of the game board.

    :param row: a string of alphabet row
    :return: a character for the previous alphabetical letter
    """
    return chr(ord(row) - 1)


def parse_move_input(move_str: str) -> tuple[list, list]:
    """
    Parses the move command input into source and destination lists.

    Example: Moving E5 and E6 to F6 and F7 respectively is expressed as e5e6,f5f6

    :param move_str: a string for the moving command following the required format in the example
    :return: a tuple containing the source list and destination list
    """
    move_str = move_str.replace(" ", "").upper()
    parts = move_str.split(",")
    if len(parts) != 2:
        raise MoveFormatError("Invalid move format")
    source_str, dest_str = parts
    source_list = re.findall(r"[A-Z]\d+", source_str)
    dest_list = re.findall(r"[A-Z]\d+", dest_str)
    if not source_list or not dest_list:
        raise MoveFormatError("Coordinates are not correct formta")
    return source_list, dest_list


def get_move_direction(source: str, dest: str) -> str:
    """
    Determines the move direction based on the source and destination coordinates

    :param source: a string for the starting coordinate (i.e. E5)
    :param dest: a string for the destination coordinate (i.e. F6)
    :return: a string specifying the movement direction of the marble
    :raises Exception: catch-all exception to be updated later
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
        raise InvalidDirectionError("Cannot determine move direction.")


def transform_coordinate(coord: str, direction: str) -> str:
    """
    Transforms a board coordinate in the given direction

    :param coord: starting coordinate
    :param direction: direction of destination
    :return: new coordinate after moving
    :raises Exception: catch-all exception to be updated later
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
        raise InvalidDirectionError("Unknown direction")

    new_coord = f"{new_row}{new_col}"
    if new_coord not in current_board:
        raise BoardBoundaryError("Out of board boundaries.")
    return new_coord


def move_marbles_cmd(marble_coords: list, direction: str) -> bool:
    """
    Moves the marbles on the board in the specified direction.

    :param marble_coords: list of strings for the coordinates of the marble(or marbles)
    :param direction: a string representing the direction to move
    :return: True if the move is successful
    """
    global current_board
    player = current_board[marble_coords[0]]
    new_coords = [transform_coordinate(coord, direction) for coord in marble_coords]
    for orig, new_coord in zip(marble_coords, new_coords):
        if new_coord not in marble_coords and current_board.get(new_coord, NO_MARBLE) != NO_MARBLE:
            raise OverlapError("Marbles cannot overlap!")
    for coord in marble_coords:
        current_board[coord] = NO_MARBLE
    for new_coord in new_coords:
        current_board[new_coord] = player
    return True


def validate_move_directions(source_list: list, dest_list: list) -> str:
    """
    Checks that all coordinate pairs for moving multiple marbles have the same move direction.
    If they are consistent, the function returns the common move direction.
    Otherwise, it raises a SomeError.

    :param source_list: a list of source coordinates
    :param dest_list: A list of destination coordinates
    :return: The common move direction as a string
    :raises SomeError: to be implemented soon
    """
    if len(source_list) != len(dest_list):
        raise MoveFormatError("The number of source and destination coordinates do not match")
    direction = get_move_direction(source_list[0], dest_list[0])
    for s, d in zip(source_list[1:], dest_list[1:]):
        if get_move_direction(s, d) != direction:
            raise InvalidDirectionError("Inconsistent move directions for multiple marbles")
    return direction

def process_move_command() -> None:
    """
    Processes the user's move command from the input field.

    Parses the move, determines the direction, moves the marbles,
    and redraws the board, combining the previous separate functions
    """
    move_text = move_entry.get()
    try:
        source_list, dest_list = parse_move_input(move_text)
        expected_color = BLACK_MARBLE if current_player == "Black" else WHITE_MARBLE
        for coord in source_list:
            if current_board.get(coord) != expected_color:
                raise InvalidMarbleError("Move your own marbles!")
        direction = validate_move_directions(source_list, dest_list)
        if move_marbles_cmd(source_list, direction):
            draw_board(current_board)
            end_turn()
        else:
            raise MoveError("Marble move failed.")
    except MoveError as e:
        messagebox.showerror("Invalid Move", str(e))
    finally:
        move_entry.delete(0, tk.END)


command_frame = tk.Frame(root, bg=THEME["bg"])
move_label = tk.Label(command_frame, text="Enter your move:", bg=THEME["bg"], fg=THEME["text"], font=("Arial", 12))
move_label.pack(pady=5)
entry_frame = tk.Frame(command_frame, bg=THEME["bg"])
entry_frame.pack(pady=3)
move_entry = tk.Entry(entry_frame, width=50, font=("Arial", 12))
move_entry.pack(side="left", padx=5)

move_entry.bind("<Return>", lambda event: process_move_command())

move_button = tk.Button(entry_frame, text="Move", command=process_move_command, font=("Arial", 12), bg=THEME["btn_bg"], fg=THEME["btn_fg"])
move_button.pack(side="left", padx=5)

root.mainloop()
