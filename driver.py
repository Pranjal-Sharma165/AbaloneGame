import tkinter as tk
from tkinter import messagebox
import math
import time

BOARD_SIZE = 600
HEX_SIZE = 30
THEME_LIGHT = {"bg": "#F0F0F0", "hex_bg": "#FFFFFF", "hex_outline": "#000000", "text": "#000000", "btn_bg": "#D3D3D3", "btn_fg": "#000000"}
THEME_DARK = {"bg": "#2E2E2E", "hex_bg": "#444444", "hex_outline": "#FFFFFF", "text": "#FFFFFF", "btn_bg": "#555555", "btn_fg": "#FFFFFF"}
THEME_BLUE = {"bg": "#B3D9FF", "hex_bg": "#99CCFF", "hex_outline": "#1E90FF", "text": "#000000", "btn_bg": "#ADD8E6", "btn_fg": "#000000"}
THEME_GREEN = {"bg": "#E6F7E6", "hex_bg": "#99FF99", "hex_outline": "#32CD32", "text": "#000000", "btn_bg": "#90EE90", "btn_fg": "#000000"}
THEME_PURPLE = {"bg": "#F0E6F6", "hex_bg": "#D8B0D8", "hex_outline": "#800080", "text": "#000000", "btn_bg": "#DDA0DD", "btn_fg": "#000000"}
THEME_ORANGE = {"bg": "#FFE6CC", "hex_bg": "#FFD580", "hex_outline": "#FF6600", "text": "#000000", "btn_bg": "#FFD700", "btn_fg": "#000000"}
THEME = THEME_LIGHT

THEMES = {
    "Light": THEME_LIGHT,
    "Dark": THEME_DARK,
    "Blue": THEME_BLUE,
    "Green": THEME_GREEN,
    "Purple": THEME_PURPLE,
    "Orange": THEME_ORANGE
}

root = tk.Tk()
root.title("Abalone Game")
root.geometry("800x700")

current_player = "Black"
move_count = 0
theme_mode = "Light"
is_paused = False
player_times = {"Black": [], "White": []}
start_time = None
pause_time = None

hex_coords = {
    "I": ["i5", "i6", "i7", "i8", "i9"],
    "H": ["h4", "h5", "h6", "h7", "h8", "h9"],
    "G": ["g3", "g4", "g5", "g6", "g7", "g8", "g9"],
    "F": ["f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"],
    "E": ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"],
    "D": ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"],
    "C": ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
    "B": ["b1", "b2", "b3", "b4", "b5", "b6"],
    "A": ["a1", "a2", "a3", "a4", "a5"]
}

def draw_hexagon(x, y, size, fill_color, outline_color):
    angle = 60
    coords = []
    for i in range(6):
        x_i = x + size * math.cos(math.radians(angle * i))
        y_i = y + size * math.sin(math.radians(angle * i))
        coords.append((x_i, y_i))
    canvas.create_polygon(coords, fill=fill_color, outline=outline_color)

def draw_board():
    canvas.delete("all")
    canvas.config(bg=THEME["bg"])

    center_x, center_y = BOARD_SIZE // 2, BOARD_SIZE // 2
    hex_width = HEX_SIZE * 1.9

    for row_label, cells in hex_coords.items():
        row_index = ord("I") - ord(row_label)
        num_cells = len(cells)

        row_width = num_cells * hex_width
        start_x = center_x - row_width / 2 + hex_width / 2
        start_y = center_y + (row_index - 4) * HEX_SIZE * math.sqrt(3)

        for i, cell in enumerate(cells):
            x = start_x + i * hex_width
            y = start_y
            draw_hexagon(x, y, HEX_SIZE, THEME["hex_bg"], THEME["hex_outline"])
            canvas.create_text(x, y, text=cell, fill=THEME["text"], font=("Arial", 10, "bold"))

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

    draw_board()
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
    global current_player, move_count, player_times, start_time, is_paused, pause_time
    current_player = "Black"
    move_count = 0
    player_times = {"Black": [], "White": []}
    start_time = None
    is_paused = False
    pause_time = None
    move_counter_label.config(text=f"Moves: {move_count}")
    timer_label.config(text="Time: 0s")
    update_turn_display()
    draw_board()
    start_timer()

def toggle_pause():
    global is_paused, pause_time, start_time
    if is_paused:
        # Resume the game
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
    draw_board()
    start_timer()

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
    relief="solid",  # Add a solid border
    bd=2,  # Border width
    padx=10,  # Horizontal padding
    pady=5  # Vertical padding
)
turn_label.pack(side="left", padx=10)
move_counter_label = tk.Label(bottom_frame, text=f"Moves: {move_count}", font=("Arial", 12), bg=THEME["bg"], fg=THEME["text"])

turn_label.pack(side="left", padx=10)
move_counter_label.pack(side="left", padx=10)

canvas = tk.Canvas(root, width=BOARD_SIZE, height=BOARD_SIZE, bg=THEME["bg"])

root.mainloop()
