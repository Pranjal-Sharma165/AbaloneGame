import tkinter as tk
from tkinter import messagebox, ttk
import math
import time
import copy
import os

from next_moves_generator import NextMove
from moves import Move, MoveError, PushNotAllowedError
from board_io import BoardIO


# Constant values for board size and hexagon size
BOARD_SIZE = 535
HEX_SIZE = 30

# Define themes with background colors and element color schemes
THEME_LIGHT = {
    "bg": "#4e5f7a",          # Background color
    "hex_bg": "#3b3c3c",      # Hexagon fill color
    "hex_outline": "#000000", # Hexagon outline color
    "text": "#000000",        # Text color
    "btn_bg": "#B0B0B0",      # Button background color
    "btn_fg": "#000000",      # Button text color
    "white_marble": "#FFFFFF" # Specific color for white marble
}
THEME_DARK = {"bg": "#4e5f7a","hex_bg": "#3b3c3c","hex_outline": "#FFFFFF","text": "#FFFFFF","btn_bg": "#555555","btn_fg": "#FFFFFF","white_marble": "#000000"}
THEME_BLUE = {"bg": "#4e5f7a","hex_bg": "#3b3c3c","hex_outline": "#0F52BA","text": "#0096FF","btn_bg": "#A1C8E0","btn_fg": "#000000","white_marble": "#89CFF0"}
THEME_GREEN = {"bg": "#4e5f7a", "hex_bg": "#3b3c3c", "hex_outline": "#50C878", "text": "#228B22", "btn_bg": "#80D68B", "btn_fg": "#000000", "white_marble": "#AFE1AF"}
THEME_PURPLE = {"bg": "#4e5f7a", "hex_bg": "#3b3c3c", "hex_outline": "#800080", "text": "#E6E6FA", "btn_bg": "#D29BE3", "btn_fg": "#000000", "white_marble": "#CBC3E3"}
THEME_BROWN = {"bg": "#4e5f7a", "hex_bg": "#3b3c3c", "hex_outline": "#5C4033", "text": "#C2B280", "btn_bg": "#A67B5B", "btn_fg": "#000000", "white_marble": "#e4d4c8"}

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
white_score = 0
black_score = 0
theme_mode = "Light"
is_paused = False
player_times = {"Black": [], "White": []}
start_time = None
pause_time = None
max_moves = 20  # Default value
move_time_limit = float("inf")  # Default: No time limit
total_pause_duration = 0  # Tracks the total time the game has been paused
total_game_time = None
game_start_time = None
game_time = 0
previous_board = None # Saved previous board state for undo

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
                text_color = "#D14B3A"  # White text for visibility
            elif cell_value == WHITE_MARBLE:
                draw_marble(x, y, HEX_SIZE, THEME["white_marble"], "#000000")  # White marble
                text_color = "#D14B3A"  # Black text for visibility
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
    global current_player, move_count, player_times, start_time, is_paused, pause_time, current_board, used_board, white_score, black_score, total_pause_duration, total_game_time, game_start_time

    current_player = "Black"
    move_count = 0
    move_counts["Black"] = 0
    move_counts["White"] = 0
    white_score = 0
    black_score = 0
    player_times = {"Player 1": [], "Player 2": []}
    start_time = None
    is_paused = False
    pause_time = None
    total_game_time = None
    game_start_time = None
    total_pause_duration = 0  # Reset the total pause duration
    move_counter_label.config(text=f"Moves: {move_count}")
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

    # Hide all game-related frames
    top_frame.pack_forget()
    status_frame.pack_forget()
    canvas.pack_forget()
    output_frame.pack_forget()
    bottom_frame.pack_forget()
    command_frame.pack_forget()
    move_history_frame.place_forget()  # Hide the move history frame
    time_history_frame.place_forget()  # Hide the time history frame

    # Show the landing page
    start_frame.pack(pady=100)


def toggle_pause():
    global is_paused, pause_time, start_time, total_pause_duration

    if is_paused:
        is_paused = False
        configure_button(pause_button, THEME["btn_bg"], THEME["btn_fg"])
        pause_button.config(text="Pause Game")

        if pause_time is not None:
            pause_duration = time.time() - pause_time
            total_pause_duration += pause_duration  # Accumulate pause duration

            # Adjust start_time to maintain correct move timing
            start_time += pause_duration

        pause_time = None

        # Re-enable UI elements
        move_entry.config(state=tk.NORMAL)
        undo_button.config(state=tk.NORMAL)
    else:
        is_paused = True
        configure_button(pause_button, "#FF6347", "white")
        pause_button.config(text="Resume Game")
        pause_time = time.time()  # Record pause time

        # Disable UI elements
        move_entry.config(state=tk.DISABLED)
        undo_button.config(state=tk.DISABLED)


def start_timer():
    global start_time, total_game_time, total_pause_duration, pause_time, game_start_time

    if is_paused:
        return  # Stop updating timer while paused

    if start_time is None:
        start_time = time.time()  # Start the timer only if it hasn't started

    if total_game_time is None:
        total_game_time = 0
        game_start_time = time.time()
        timer_label.config(text=f"Time: {int(total_game_time)}s")
    else:
        if pause_time is not None:
            # Correct game time calculation: Only subtract total pause duration once
            total_game_time = time.time() - game_start_time - total_pause_duration
        else:
            total_game_time = time.time() - game_start_time

    # Correct elapsed move time calculation
    elapsed_time = time.time() - start_time

    if move_time_limit != float("inf") and elapsed_time >= move_time_limit:
        messagebox.showwarning("Time Up!", f"{current_player}'s time is up! Turn is ending.")
        end_turn()
        return  # Stop updating timer after turn ends

    root.after(1000, start_timer)  # Call again after 1 second

def end_turn():
    global current_player, start_time, is_paused, pause_time, game_start_time, total_pause_duration, total_game_time

    # Calculate total moves played by both players
    total_moves_played = move_counts['Black'] + move_counts['White']

    # Check if the next move will exceed the move limit
    if total_moves_played >= 2 * max_moves - 1:
        messagebox.showinfo("Game Over", "Both players have reached their move limit! Game Over.")
        stop_game()
        return

    # Update move count for the current player
    move_counts[current_player] += 1

    # Update the move counter label
    move_counter_label.config(text=f"Moves: {move_counts}")

    # Calculate move duration and update logs
    if start_time is not None:
        move_duration = time.time() - start_time
        total_game_time = time.time() - game_start_time - total_pause_duration
        display_turn_duration_log(current_player, move_duration)

        # Reset start_time for the next move
        start_time = time.time()

    # Switch to the next player
    current_player = "White" if current_player == "Black" else "Black"
    is_paused = False
    pause_time = None
    update_turn_display()


def start_game():
    global game_start_time, total_pause_duration, is_paused, start_time, game_start_time, max_moves

    try:
        max_moves = int(max_moves_entry.get())
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number for max moves.")
        return

    game_start_time = time.time()
    total_pause_duration = 0
    is_paused = False
    start_time = time.time()
    game_start_time = time.time()

    pause_button.config(state=tk.NORMAL, text="Pause Game")
    update_total_game_time()

    start_frame.pack_forget()

    top_frame.pack(ipady=5, pady=3)
    status_frame.pack(pady=5, ipadx=254)
    canvas.pack()
    move_history_frame.place(relheight=0.5, relx=0.15, rely=0.12)
    time_history_frame.place(relheight=0.509, relx=0.85, rely=0.375, anchor="e")
    bottom_frame.pack(ipadx=46, ipady=5, pady=1)
    output_frame.pack(padx=20, pady=1)
    draw_board(current_board)
    start_timer()
    command_frame.pack(ipadx=28, ipady=10)


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
    global used_board, current_board, previous_board

    # Check if the game is paused
    if is_paused:
        messagebox.showinfo("Game Paused", "The game is paused. Resume the game to undo moves.")
        return

    # Check if there is a previous board state saved
    if previous_board:
        current_board = previous_board
        previous_board = None # Prevent player from undo twice
        draw_board(current_board)
    else:
        messagebox.showinfo("Undo", "You can only undo a move once.")


def display_ai_move_log(move):
    """
    Updates the move history display with the latest move.
    """
    move_history_text.config(state="normal")  # Enable editing
    move_history_text.insert(tk.END, f"{move}\n")  # Append the move
    move_history_text.see(tk.END)  # Scroll to the bottom
    move_history_text.config(state="disabled")  # Disable editing

def display_turn_duration_log(player, duration):
    """
    Updates the time history display with the time taken by the player for their move.
    """
    time_history_text.config(state="normal")  # Enable editing
    time_history_text.insert(tk.END, f"{player}: {duration:.2f} sec\n")  # Append the move duration
    time_history_text.see(tk.END)  # Scroll to the bottom
    time_history_text.config(state="disabled")  # Disable editing

def process_generate_all_next_moves():
    global current_board, current_player, move_counts

    current_color = BoardIO.BLACK_MARBLE if current_player == "Black" else BoardIO.WHITE_MARBLE
    output_dir = "./output"
    turn_total = move_counts['White'] + move_counts['Black']

    moves_filename = os.path.join(output_dir, f"moves_turn{turn_total}.txt")
    boards_filename = os.path.join(output_dir, f"boards_turn{turn_total}.txt")

    moves = NextMove.generate_and_save_all_next_moves(current_board, current_color, current_player,
                                                        moves_filename, boards_filename)
    messagebox.showinfo("Next Moves", f"Generated {len(moves)} moves.\n"
                                      f"Move notations saved.\n"
                                      f"Board configurations saved.")

def process_move_command():
    global white_score, black_score, previous_board

    # save a copy of the current board state before the next move is applied
    previous_board = copy.deepcopy(current_board)

    if is_paused:
        messagebox.showinfo("Game Paused", "The game is paused. Resume the game to command.")
        return

    move_text = move_entry.get().strip()

    if move_text.lower() == "save board":
        BoardIO.export_current_board_to_text(current_board, current_player,
                                     f"./output/turn_{move_counts['White'] + move_counts['Black']}.txt")
        messagebox.showinfo("Current board saved", f"Generated the current board.\n")
        move_entry.delete(0, tk.END)
        return

    elif move_text.lower() == "next move":
        process_generate_all_next_moves()
        move_entry.delete(0, tk.END)
        return

    try:
        source_list, dest_list = Move.parse_move_input(move_text)
        expected_color = BoardIO.BLACK_MARBLE if current_player == "Black" else BoardIO.WHITE_MARBLE
        opponent_color = BoardIO.WHITE_MARBLE if current_player == "Black" else BoardIO.BLACK_MARBLE

        for coord in source_list:
            if current_board.get(coord) != expected_color:
                raise MoveError("Move your own marbles!")

        direction = Move.validate_move_directions(source_list, dest_list)
        push_success = Move.move_marbles_cmd(current_board, source_list, direction, expected_color, opponent_color)
        if push_success:
            if expected_color == BoardIO.BLACK_MARBLE:
                white_score += 1
                white_score_label.config(text=f"White Marbles Lost: {white_score}")
            else:
                black_score += 1
                black_score_label.config(text=f"Black Marbles Lost: {black_score}")

        display_ai_move_log(move_text)
        draw_board(current_board)
        end_turn()
    except (MoveError, PushNotAllowedError) as e:
        messagebox.showerror("Invalid Move", str(e))
    finally:
        move_entry.delete(0, tk.END)

def update_total_game_time():
    if game_start_time and not is_paused:
        elapsed_time = time.time() - game_start_time - total_pause_duration
        timer_label.config(text=f"Time: {int(elapsed_time)}s")
    root.after(1000, update_total_game_time)  # Update every second


if __name__ == '__main__':
    command_entry = tk.Entry(root)


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

    # Creating label and entry for max moves allowed
    max_moves_label = tk.Label(start_frame, text="Max Moves Per Player:", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"])
    max_moves_label.pack(pady=10)

    max_moves_entry = tk.Entry(start_frame, font=("Arial", 12))
    max_moves_entry.pack(pady=5)
    max_moves_entry.insert(0, "20")  # Default value

    # Creating label and entry for time limit per move
    time_limit_label = tk.Label(start_frame, text="Time Limit Per Move (seconds)(i for ∞):", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"])
    time_limit_label.pack(pady=10)

    time_limit_entry = tk.Entry(start_frame, font=("Arial", 12))
    time_limit_entry.pack(pady=5)
    time_limit_entry.insert(0, "i")  # Default to infinity

    start_button = tk.Button(start_frame, text="Start Game", command=start_game, font=("Arial", 14), bg=THEME["btn_bg"], fg=THEME["btn_fg"], relief="raised", bd=2)
    start_button.pack(pady=10)

    exit_button = tk.Button(start_frame, text="Exit", command=exit_game, font=("Arial", 14), bg=THEME["btn_bg"], fg=THEME["btn_fg"], relief="raised", bd=2)
    exit_button.pack(pady=10)

    top_frame = tk.Frame(root, bg=THEME["bg"])
    reset_button = tk.Button(top_frame, text="Reset Game", command=reset_game, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
    theme_button = tk.Button(top_frame, text="Switch Theme", command=switch_theme, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
    pause_button = tk.Button(top_frame, text="Pause Game", command=toggle_pause, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
    undo_button = tk.Button(top_frame, text="Undo Move", command=undo_move, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
    # end_turn_button = tk.Button(top_frame, text="End Turn", command=end_turn, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
    stop_button = tk.Button(top_frame, text="Stop Game", command=stop_game, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)

    reset_button.pack(side="left", padx=10)
    theme_button.pack(side="left", padx=10)
    pause_button.pack(side="left", padx=10)
    undo_button.pack(side="left", padx=10)
    # end_turn_button.pack(side="left", padx=10)
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
    # configure_button(end_turn_button, "#2196F3")
    configure_button(stop_button, "#FF0000")

    # Output box UI that displays last turn duration, previous move, and suggested next move
    output_frame = tk.Frame(root, bg=THEME["bg"], bd=5, relief="solid")

    # move_duration_label = tk.Label(output_frame, text="Duration of last turn: 00:00:36 seconds", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"], anchor="w", justify="left")
    # move_duration_label.pack(side="top", padx=10, fill="both")
    # prev_move_label = tk.Label(output_frame, text="Previous move: c3c4c5, d3d4d5", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"], anchor="w", justify="left")
    # prev_move_label.pack(side="top", padx=10, fill="both")
    next_move_label = tk.Label(output_frame, text="Next move: {optimal next move goes here}", font=("Arial", 18, "bold"), bg=THEME["bg"], fg=THEME["text"])
    next_move_label.pack(side="top", padx=10)

    # Log button UI that includes two buttons to display turn duration and AI move history logs
    # log_frame = tk.Frame(root, bg=THEMEa["bg"])

    # Initialize the move history frame (but do not show it yet)
    move_history_frame = tk.Frame(root, bg=THEME["bg"], bd=5, relief="solid")
    move_history_label = tk.Label(move_history_frame, text="Move History", font=("Arial", 14, "bold"), bg=THEME["bg"], fg=THEME["text"])
    move_history_label.pack(pady=5)
    move_history_text = tk.Text(
        move_history_frame,
        wrap=tk.WORD,
        width=30,
        height=30,
        bg=THEME["bg"],
        fg=THEME["text"],
        state="disabled"  # Make it read-only
    )
    move_history_text.pack(fill="both", expand=True)

    # Frame for Time History (Right Side)
    time_history_frame = tk.Frame(root, bg=THEME["bg"], bd=5, relief="solid")
    # time_history_frame.place(relheight=0.509, relx=0.995, anchor="e", y=365)
    # time_history_frame.pack(side="right", fill="y", padx=10, pady=10)

    # Label for Time History
    time_history_label = tk.Label(time_history_frame, text="Time History", font=("Arial", 14, "bold"), bg=THEME["bg"], fg=THEME["text"])
    time_history_label.pack(pady=5)

    time_history_text = tk.Text(
        time_history_frame,
        wrap=tk.WORD,
        width=30,
        height=30,
        bg=THEME["bg"],
        fg=THEME["text"],
        state="disabled"  # Make it read-only
    )
    time_history_text.pack(fill="both", expand=True)





    # move_history_button = tk.Button(log_frame, text="AI Move History", command=display_ai_move_log, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
    # move_history_button.pack(side="left", padx=(0,10))
    # time_history_button = tk.Button(log_frame, text="Turn Duration History", command=display_turn_duration_log, bg=THEME["btn_bg"], fg=THEME["btn_fg"], font=("Arial", 12), relief="raised", bd=2)
    # time_history_button.pack(side="left")

    # Adding label to show game mode
    status_frame = tk.Frame(root, bg=THEME["bg"])
    timer_label = tk.Label(status_frame, text="Time: 0s", font=("Arial", 20), bg=THEME["bg"], fg=THEME["text"])
    white_score_label = tk.Label(status_frame, text=f"White Marbles Lost: {white_score}", font=("Arial", 15, "bold"), bg=THEME["bg"], fg=THEME["text"])
    white_score_label.pack(side="left")
    black_score_label = tk.Label(status_frame, text=f"Black Marbles Lost: {black_score}", font=("Arial", 15, "bold"), bg=THEME["bg"], fg=THEME["text"])
    black_score_label.pack(side="right")

    timer_label.pack(padx=10)

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


    command_frame = tk.Frame(root, bg=THEME["bg"])
    move_label = tk.Label(command_frame, text="Enter your Command:", bg=THEME["bg"], fg=THEME["text"], font=("Arial", 12))
    move_label.pack(pady=5)
    entry_frame = tk.Frame(command_frame, bg=THEME["bg"])
    entry_frame.pack(pady=3)
    move_entry = tk.Entry(entry_frame, width=50, font=("Arial", 12))
    move_entry.pack(side="left", padx=5)

    move_entry.bind("<Return>", lambda event: process_move_command())

    # move_button = tk.Button(entry_frame, text="Move", command=process_move_command, font=("Arial", 12), bg=THEME["btn_bg"], fg=THEME["btn_fg"])
    # move_button.pack(side="left", padx=5)

    root.mainloop()
