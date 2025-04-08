import time
import json
import os
import copy
import random
from move_cy import convert_board_format
from driver import STANDARD_BOARD_INIT, BELGIAN_BOARD_INIT, GERMAN_BOARD_INIT
from functools import lru_cache

MOVE_GENERATOR = None

BOARD_TYPES = {
    "belgian": BELGIAN_BOARD_INIT,
    "german": GERMAN_BOARD_INIT,
    "standard": STANDARD_BOARD_INIT
}


class AIWithCustomWeights:
    ai_module = None

    def __init__(self, ai_module_path='AI', weights=None, name="AI"):
        global MOVE_GENERATOR
        import importlib

        if AIWithCustomWeights.ai_module is None:
            AIWithCustomWeights.ai_module = importlib.import_module(ai_module_path)

        self.ai = AIWithCustomWeights.ai_module
        self.original_weights = self.ai.WEIGHTS.copy()
        self.weights = weights if weights else self.original_weights
        self.name = name

        if weights:
            self._apply_weights()

        if MOVE_GENERATOR is None:
            try:
                from next_move_generator_cy import generate_all_next_moves
                MOVE_GENERATOR = generate_all_next_moves
            except ImportError:
                print("Error importing move generator")
                MOVE_GENERATOR = None

    def _apply_weights(self):
        if not self.weights:
            return
        for key, value in self.weights.items():
            if key in self.ai.WEIGHTS:
                self.ai.WEIGHTS[key] = value

    def _restore_weights(self):
        for key, value in self.original_weights.items():
            if key in self.ai.WEIGHTS:
                self.ai.WEIGHTS[key] = value

    def find_best_move(self, board, player, depth=5, time_limit=3.0):
        try:
            self._apply_weights()
            return self.ai.find_best_move(board, player, depth, time_limit, MOVE_GENERATOR)
        finally:
            self._restore_weights()


@lru_cache(maxsize=32)
def load_weights_from_file(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading weights file {filename}: {e}")
        return None


def save_weights_to_file(weights, filename):
    try:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(weights, f, indent=2)
        print(f"Weights saved to: {filename}")
    except Exception as e:
        print(f"Error saving weights to {filename}: {e}")


def modify_weights(weights, learning_rate):
    new_weights = {}
    for key, value in weights.items():
        adjustment = random.uniform(-learning_rate, learning_rate)
        new_weights[key] = value * (1 + adjustment)
    return new_weights


def conduct_battle(champion_ai, challenger_ai, num_games=1, time_limit=3.0, initial_board_type="standard",
                   max_depth=5, verbose=True, record_moves=False):
    initial_board_dict = BOARD_TYPES.get(initial_board_type.lower(), STANDARD_BOARD_INIT)

    stats = {
        "champion_wins": 0,
        "challenger_wins": 0,
        "ties": 0,
        "total_moves": 0,
    }

    if record_moves:
        stats["games"] = []

    if verbose:
        print(f"\n{'=' * 30}")
        print(f" {champion_ai.name} vs {challenger_ai.name} - {num_games} Games ")
        print(f" AI Battle Mode (Depth={max_depth})")
        print(f"{'=' * 30}")
        print(f"Board type: {initial_board_type}")
        print(f"Time limit: {time_limit} seconds")
        print(f"Maximum turns: 100")

    black_ai = challenger_ai
    white_ai = champion_ai
    max_moves = 100

    for game_num in range(1, num_games + 1):
        if verbose:
            print(f"\n--- Game {game_num}/{num_games} ---")

        board_dict = copy.deepcopy(initial_board_dict)
        board = convert_board_format(board_dict)

        current_player = "black"
        moves_made = 0
        black_lost = 0
        white_lost = 0

        game_moves = [] if record_moves else None

        if verbose:
            print(f"This game: Black({black_ai.name}) vs White({white_ai.name})")

        while moves_made < max_moves:

            ai = black_ai if current_player == "black" else white_ai
            if verbose:
                print(f"Turn {moves_made + 1}/{max_moves} - {current_player} ({ai.name})...")

            start_time = time.time()
            best_move, move_str, _ = ai.find_best_move(board, current_player, max_depth, time_limit)
            end_time = time.time()

            if best_move is None:
                if verbose:
                    print(f"No valid moves for {current_player}. Game over.")
                break

            if verbose:
                print(f"Selected move: {move_str} (Search time: {end_time - start_time:.2f}s)")

            if record_moves:
                game_moves.append({
                    "player": current_player,
                    "ai": ai.name,
                    "move": move_str,
                    "time": end_time - start_time
                })

            prev_black_count = len(board[0])
            prev_white_count = len(board[1])
            board = best_move
            next_black_count = len(board[0])
            next_white_count = len(board[1])

            if next_black_count < prev_black_count:
                lost = prev_black_count - next_black_count
                black_lost += lost
                if verbose:
                    print(f"Black lost {lost} marbles. Total lost: {black_lost}")

            if next_white_count < prev_white_count:
                lost = prev_white_count - next_white_count
                white_lost += lost
                if verbose:
                    print(f"White lost {lost} marbles. Total lost: {white_lost}")

            if black_lost >= 6:
                if verbose:
                    print(f"White wins! (Black lost {black_lost} marbles)")
                    print(f"Champion ({champion_ai.name}) wins this game!")
                stats["champion_wins"] += 1
                break

            if white_lost >= 6:
                if verbose:
                    print(f"Black wins! (White lost {white_lost} marbles)")
                    print(f"Challenger ({challenger_ai.name}) wins this game!")
                stats["challenger_wins"] += 1
                break

            current_player = "white" if current_player == "black" else "black"
            moves_made += 1

        if moves_made >= max_moves:
            if verbose:
                print(f"Maximum moves ({max_moves}) reached. Determining winner by marble count.")

            if black_lost < white_lost:
                if verbose:
                    print(f"Black wins! (Black lost {black_lost} vs White lost {white_lost})")
                    print(f"Challenger ({challenger_ai.name}) wins this game!")
                stats["challenger_wins"] += 1
            elif white_lost < black_lost:
                if verbose:
                    print(f"White wins! (Black lost {black_lost} vs White lost {white_lost})")
                    print(f"Champion ({champion_ai.name}) wins this game!")
                stats["champion_wins"] += 1
            else:  # Tie
                if verbose:
                    print(f"It's a tie! (Both lost {black_lost} marbles)")
                    print("Tie game - no points awarded to either AI")
                stats["ties"] += 1

        stats["total_moves"] += moves_made

        if record_moves and "games" in stats:
            stats["games"].append({
                "game_number": game_num,
                "winner": "champion" if white_lost < black_lost else "challenger" if black_lost < white_lost else "tie",
                "moves_made": moves_made,
                "black_lost": black_lost,
                "white_lost": white_lost,
                "moves": game_moves
            })

        if verbose:
            print(f"\n--- Current Match Score ---")
            print(f"Champion ({champion_ai.name}): {stats['champion_wins']} wins")
            print(f"Challenger ({challenger_ai.name}): {stats['challenger_wins']} wins")
            print(f"Ties: {stats['ties']}")

    if verbose:
        print(f"\n{'=' * 30}")
        print(f" Match Results: {num_games} games (Depth={max_depth})")
        print(f"{'=' * 30}")

        denom = max(1, num_games)
        win_percentage_champion = (stats['champion_wins'] / denom * 100)
        win_percentage_challenger = (stats['challenger_wins'] / denom * 100)

        print(f"Champion ({champion_ai.name}): {stats['champion_wins']} wins ({win_percentage_champion:.1f}%)")
        print(f"Challenger ({challenger_ai.name}): {stats['challenger_wins']} wins ({win_percentage_challenger:.1f}%)")
        print(f"Ties: {stats['ties']} ({stats['ties'] / denom * 100:.1f}%)")
        print(f"Average moves per game: {stats['total_moves'] / denom:.1f}")

    stats["champion_name"] = champion_ai.name
    stats["challenger_name"] = challenger_ai.name
    stats["champion_weights"] = champion_ai.weights
    stats["challenger_weights"] = challenger_ai.weights

    return stats


def save_match_results(stats, match_number, learning_rate):
    results_dir = "battle_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{results_dir}/match{match_number:04d}_{stats['champion_name']}_vs_{stats['challenger_name']}_{timestamp}.json"

    match_info = {
        "match_number": match_number,
        "champion": {
            "name": stats["champion_name"],
            "weights": stats["champion_weights"]
        },
        "challenger": {
            "name": stats["challenger_name"],
            "weights": stats["challenger_weights"]
        },
        "learning_rate": learning_rate,
        "stats": stats
    }

    match_json = json.dumps(match_info, indent=2)
    with open(filename, 'w') as f:
        f.write(match_json)

    print(f"\nMatch results saved to: {filename}")
    return filename


def run_evolutionary_battles(initial_weights=None, learning_rate=0.05,
                             learning_rate_decay=0.95, games_per_match=1, time_limit=3.0,
                             board_type="standard", max_depth=5, run_indefinitely=True,
                             verbose=True, record_moves=False):
    if initial_weights is None:
        import importlib
        ai_module = importlib.import_module('AI')
        initial_weights = ai_module.WEIGHTS.copy()

    if verbose:
        print(f"Starting evolutionary battles (press Ctrl+C to stop)")
        print(f"Initial weights: {initial_weights}")
        print(f"Initial learning rate: {learning_rate}")
        print(f"Learning rate decay: {learning_rate_decay}")

    results_dir = "evolution_results"
    os.makedirs(results_dir, exist_ok=True)

    champion_weights = initial_weights
    champion = AIWithCustomWeights(weights=champion_weights, name="Champion")

    match_history = []
    history_file = f"{results_dir}/evolution_history.json"

    current_learning_rate = learning_rate
    min_learning_rate = 0.001

    challenger_weights = modify_weights(champion_weights, current_learning_rate)
    challenger = AIWithCustomWeights(weights=challenger_weights, name="Challenger")

    match_num = 1

    try:
        while run_indefinitely:
            if verbose:
                print(f"\n{'*' * 50}")
                print(f"* MATCH {match_num}")
                print(f"* Learning rate: {current_learning_rate:.6f}")
                print(f"{'*' * 50}")
                print(f"Champion ({champion.name}) weights: {champion.weights}")
                print(f"Challenger ({challenger.name}) weights: {challenger.weights}")

            stats = conduct_battle(
                champion, challenger, games_per_match,
                time_limit, board_type, max_depth,
                verbose=verbose, record_moves=record_moves
            )

            result_file = save_match_results(stats, match_num, current_learning_rate)

            champion_wins = stats["champion_wins"]
            challenger_wins = stats["challenger_wins"]

            match_result = {
                "match_number": match_num,
                "champion_wins": champion_wins,
                "challenger_wins": challenger_wins,
                "ties": stats["ties"],
                "learning_rate": current_learning_rate,
            }

            match_history.append(match_result)

            if challenger_wins > champion_wins:
                if verbose:
                    print(f"\nChallenger ({challenger.name}) wins match {match_num}!")
                    print(f"Challenger becomes the new champion.")

                new_champion = AIWithCustomWeights(weights=challenger.weights, name="Champion")

                new_challenger_weights = modify_weights(challenger.weights, current_learning_rate)
                new_challenger = AIWithCustomWeights(weights=new_challenger_weights, name="Challenger")

                champion = new_champion
                challenger = new_challenger

                champion_file = f"{results_dir}/champion_weights_match{match_num:04d}.json"
                save_weights_to_file(champion.weights, champion_file)
            else:
                if verbose:
                    print(f"\nChampion ({champion.name}) defends in match {match_num}!")
                    print(f"Champion remains the same.")

                new_challenger_weights = modify_weights(champion.weights, current_learning_rate)
                new_challenger = AIWithCustomWeights(weights=new_challenger_weights, name="Challenger")
                challenger = new_challenger

            current_learning_rate = max(min_learning_rate, current_learning_rate * learning_rate_decay)

            match_num += 1

            if match_num % 5 == 0:
                boost_factor = 1.5
                current_learning_rate = min(learning_rate, current_learning_rate * boost_factor)
                if verbose:
                    print(f"\nApplying learning rate boost for match {match_num}: {current_learning_rate:.6f}")

            if match_num % 10 == 0:
                with open(history_file, 'w') as f:
                    json.dump(match_history, f, indent=2)
                print(f"Match history saved to: {history_file}")

    except KeyboardInterrupt:
        print("\n\nEvolution manually stopped by user.")
    finally:
        with open(history_file, 'w') as f:
            json.dump(match_history, f, indent=2)
        print(f"Final match history saved to: {history_file}")

    print(f"\nEvolutionary battles completed after {match_num - 1} matches.")
    print(f"Final champion weights: {champion.weights}")

    final_weights_file = f"{results_dir}/champion_weights_final.json"
    save_weights_to_file(champion.weights, final_weights_file)

    return champion.weights


def manual_weight_configuration():
    print("Configure AI weights manually")

    import importlib
    ai_module = importlib.import_module('AI')
    default_weights = ai_module.WEIGHTS.copy()

    print(f"Default weights: {default_weights}")

    try:
        weights = {}
        print("\nEnter weight values:")
        for key in default_weights:
            default = default_weights[key]
            value = input(f"{key} (default: {default}): ")
            weights[key] = float(value) if value.strip() else default

        print(f"Configured weights: {weights}")
        return weights
    except ValueError:
        print("Invalid input. Using default weights.")
        return default_weights


def main():
    print("Abalone AI Evolutionary Battle System")
    print("=" * 40)

    learning_rate = 0.08
    learning_rate_decay = 0.95
    games_per_match = 1
    time_limit = 7.0
    board_type = "belgian"
    max_depth = 3
    run_indefinitely = True
    verbose = True
    record_moves = False

    print("\nConfiguration options:")
    print("1. Use default settings")
    print("2. Configure settings")

    choice = input("Select option (1-2): ")

    if choice == "2":
        try:
            learning_rate = float(
                input(f"Initial learning rate (0.01-0.1, default: {learning_rate}): ") or learning_rate)
            learning_rate = max(0.01, min(0.1, learning_rate))

            learning_rate_decay = float(
                input(f"Learning rate decay (0.9-0.99, default: {learning_rate_decay}): ") or learning_rate_decay)
            learning_rate_decay = max(0.9, min(0.99, learning_rate_decay))

            games_per_match = int(input(f"Games per match (default: {games_per_match}): ") or games_per_match)
            games_per_match = max(1, min(10, games_per_match))

            time_limit = float(input(f"Time limit per move in seconds (default: {time_limit}): ") or time_limit)
            time_limit = max(1.0, min(10.0, time_limit))

            board_options = ["Standard", "Belgian", "German"]
            print("Board type:")
            for i, option in enumerate(board_options):
                print(f"{i + 1}. {option}")
            board_idx = int(input(f"Select (1-3, default: 2): ") or 2) - 1
            if 0 <= board_idx < len(board_options):
                board_type = board_options[board_idx]

            max_depth = int(input(f"Maximum search depth (1-5, default: {max_depth}): ") or max_depth)
            max_depth = max(1, min(5, max_depth))

            verbose_input = input(f"Verbose output (y/n, default: y): ").lower() or "y"
            verbose = verbose_input.startswith("y")

            if verbose:
                record_moves_input = input(f"Record detailed moves (y/n, default: n): ").lower() or "n"
                record_moves = record_moves_input.startswith("y")

        except ValueError:
            print("Invalid input detected. Using default values for any invalid entries.")

    print("\nInitial weights configuration:")
    print("1. Use default weights")
    print("2. Configure weights manually")
    print("3. Load weights from file")

    weights_choice = input("Select option (1-3): ")

    initial_weights = None

    if weights_choice == "2":
        initial_weights = manual_weight_configuration()
    elif weights_choice == "3":
        weights_dir = "weights"
        if not os.path.exists(weights_dir):
            print(f"{weights_dir} directory not found. Using default weights.")
        else:
            weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.json')]
            if not weight_files:
                print(f"No weight files found in {weights_dir}. Using default weights.")
            else:
                print("\nAvailable weight files:")
                for i, file in enumerate(weight_files):
                    print(f"{i + 1}. {file}")

                try:
                    file_idx = int(input("Select file (number): ")) - 1
                    if 0 <= file_idx < len(weight_files):
                        file_path = os.path.join(weights_dir, weight_files[file_idx])
                        initial_weights = load_weights_from_file(file_path)
                        print(f"Weights loaded from: {file_path}")
                except ValueError:
                    print("Invalid selection. Using default weights.")

    print("\nStarting evolutionary battles with the following settings:")
    print(f"Learning rate: {learning_rate} (decay: {learning_rate_decay})")
    print(f"Games per match: {games_per_match}")
    print(f"Time limit: {time_limit} seconds")
    print(f"Board type: {board_type}")
    print(f"Max depth: {max_depth}")
    print(f"Initial weights: {initial_weights or 'default'}")
    print(f"Verbose output: {verbose}")
    print(f"Record moves: {record_moves}")

    print("\nThe system will run until manually stopped (Ctrl+C).")
    print("Results are saved after each match, so you can stop anytime.")
    print("\nBattle Rules:")
    print(f"1. Each match consists of {games_per_match} game(s)")
    print("2. Champion plays as White, Challenger plays as Black")
    print("3. After a match, the winner becomes the Champion, the loser is replaced")
    print("4. This process continues indefinitely until manually stopped")

    input("\nPress Enter to start...")

    try:
        best_weights = run_evolutionary_battles(
            initial_weights=initial_weights,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            games_per_match=games_per_match,
            time_limit=time_limit,
            board_type=board_type,
            max_depth=max_depth,
            run_indefinitely=run_indefinitely,
            verbose=verbose,
            record_moves=record_moves
        )

        save_weights_to_file(best_weights, "best_weights_final.json")
        print("Evolution complete! Final best weights saved to best_weights_final.json")

    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Results up to the last completed match were saved.")
        print("You can run the analyzer module to examine the results so far.")


if __name__ == "__main__":
    main()