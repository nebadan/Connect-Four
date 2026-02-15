"""
main.py — Connect Four: Human vs. AI (Minimax Agent)

This module provides the interactive command-line interface for playing
Connect Four against an AI agent powered by the Minimax algorithm with
alpha-beta pruning and depth-limited heuristic evaluation.

Features:
    - Colored terminal display with Unicode piece symbols
    - Player order selection (choose to go first or second)
    - Input validation with helpful error messages
    - AI performance metrics displayed after each move
    - Game outcome announcement with option to replay

Usage:
    python main.py
"""

import sys
from game import (
    ROWS, COLS, PLAYER_1, PLAYER_2,
    create_board, get_legal_moves, make_move,
    is_terminal, get_winner, print_board, get_opponent,
)
from minimax import get_best_move


# ANSI Formatting
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Default search depth
DEFAULT_DEPTH = 6


def clear_screen():
    """Print newlines to visually separate game states."""
    print("\n" * 2)


def print_banner():
    """Display the welcome banner and game rules."""
    print(f"""
{BLUE}{BOLD}╔══════════════════════════════════════════════════════╗
║                                                      ║
║           🎮  CONNECT FOUR — AI CHALLENGE  🎮         ║
║                                                      ║
║         Minimax Agent with Alpha-Beta Pruning         ║
║                                                      ║
╚══════════════════════════════════════════════════════╝{RESET}

{CYAN}Rules:{RESET}
  • Drop pieces into columns (1-7) to build a line of four.
  • Connect {BOLD}four in a row{RESET} — horizontally, vertically,
    or diagonally — to win!
  • If the board fills up with no winner, it's a draw.

{DIM}Pieces:  🔴 = Player 1   🟡 = Player 2{RESET}
""")


def choose_player_order():
    """
    Let the user choose whether to play first or second.

    Returns:
        tuple: (human_player, ai_player) — the player numbers.
    """
    while True:
        print(f"{CYAN}Do you want to go first? (y/n):{RESET} ", end="")
        choice = input().strip().lower()
        if choice in ("y", "yes"):
            print(f"\n{GREEN}You are 🔴 Player 1. You go first!{RESET}")
            return PLAYER_1, PLAYER_2
        elif choice in ("n", "no"):
            print(f"\n{GREEN}You are 🟡 Player 2. AI goes first!{RESET}")
            return PLAYER_2, PLAYER_1
        else:
            print(f"{RED}Please enter 'y' or 'n'.{RESET}")


def choose_difficulty():
    """
    Let the user choose the AI difficulty (search depth).

    Returns:
        int: The search depth for the minimax algorithm.
    """
    print(f"\n{CYAN}Select difficulty:{RESET}")
    print(f"  {DIM}1{RESET} — Easy   (depth 3)")
    print(f"  {DIM}2{RESET} — Medium (depth 5)")
    print(f"  {DIM}3{RESET} — Hard   (depth 7)")
    print()

    while True:
        print(f"{CYAN}Your choice (1/2/3):{RESET} ", end="")
        choice = input().strip()
        if choice == "1":
            return 3
        elif choice == "2":
            return 5
        elif choice == "3":
            return 7
        else:
            print(f"{RED}Please enter 1, 2, or 3.{RESET}")


def get_human_move(board):
    """
    Prompt the human player for a valid column selection.

    Validates that:
        - The input is a number
        - The column is within bounds (1-7)
        - The selected column is not full

    Args:
        board: The current board state.

    Returns:
        int: A valid column index (0-indexed).
    """
    legal = get_legal_moves(board)

    while True:
        print(f"{CYAN}Your move (column 1-{COLS}):{RESET} ", end="")
        try:
            raw = input().strip()
            col = int(raw) - 1  # Convert to 0-indexed

            if col < 0 or col >= COLS:
                print(f"{RED}Invalid column. Please enter a number "
                      f"between 1 and {COLS}.{RESET}")
                continue

            if col not in legal:
                print(f"{RED}Column {col + 1} is full. "
                      f"Choose another column.{RESET}")
                continue

            return col

        except ValueError:
            print(f"{RED}Invalid input. Please enter a number "
                  f"between 1 and {COLS}.{RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Game aborted. Goodbye!{RESET}")
            sys.exit(0)


def display_ai_stats(stats):
    """
    Display the AI's search performance metrics.

    Args:
        stats (dict): Contains 'nodes_expanded', 'time_seconds', 'depth'.
    """
    print(f"  {DIM}├─ Nodes expanded : {stats['nodes_expanded']:,}{RESET}")
    print(f"  {DIM}├─ Search depth   : {stats['depth']}{RESET}")
    print(f"  {DIM}└─ Time taken     : {stats['time_seconds']:.3f}s{RESET}")


def announce_result(winner, human_player, ai_player):
    """
    Display the final game result.

    Args:
        winner: The winning player number, or None for a draw.
        human_player (int): The human's player number.
        ai_player (int): The AI's player number.
    """
    print()
    if winner == human_player:
        print(f"{GREEN}{BOLD}🎉 Congratulations! You won! 🎉{RESET}")
    elif winner == ai_player:
        print(f"{RED}{BOLD}🤖 The AI wins! Better luck next time. 🤖{RESET}")
    else:
        print(f"{YELLOW}{BOLD}🤝 It's a draw! Well played. 🤝{RESET}")
    print()


def play_game():
    """
    Main game loop: manages turns, input, AI moves, and game termination.

    The game proceeds as follows:
        1. Initialize an empty board
        2. Let the user choose to go first or second
        3. Alternate turns between human and AI
        4. On each turn:
           - Display the current board
           - Get the move from human input or AI computation
           - Apply the move and check for game-over
        5. Announce the result when the game ends
    """
    print_banner()

    human_player, ai_player = choose_player_order()
    depth = choose_difficulty()

    board = create_board()
    current_player = PLAYER_1  # Player 1 always starts

    piece_name = {
        PLAYER_1: f"{RED}🔴 Player 1{RESET}",
        PLAYER_2: f"{YELLOW}🟡 Player 2{RESET}",
    }

    # Total stats tracking
    total_nodes = 0
    total_time = 0.0
    move_count = 0

    print_board(board)

    while True:
        if current_player == human_player:
            # ── Human's Turn ──
            col = get_human_move(board)
            board = make_move(board, col, human_player)
            print(f"\n  {piece_name[human_player]} plays column "
                  f"{BOLD}{col + 1}{RESET}")
        else:
            # ── AI's Turn ──
            print(f"\n  {DIM}🤖 AI is thinking...{RESET}")
            col, stats = get_best_move(board, ai_player, max_depth=depth)
            board = make_move(board, col, ai_player)

            total_nodes += stats["nodes_expanded"]
            total_time += stats["time_seconds"]
            move_count += 1

            print(f"\n  {piece_name[ai_player]} plays column "
                  f"{BOLD}{col + 1}{RESET}")
            display_ai_stats(stats)

        print_board(board)

        # ── Check for game over ──
        if is_terminal(board):
            winner = get_winner(board)
            announce_result(winner, human_player, ai_player)

            # Display cumulative AI performance summary
            if move_count > 0:
                print(f"{DIM}{'─' * 40}{RESET}")
                print(f"{CYAN}{BOLD}AI Performance Summary:{RESET}")
                print(f"  {DIM}Total nodes expanded : "
                      f"{total_nodes:,}{RESET}")
                print(f"  {DIM}Total AI moves       : {move_count}{RESET}")
                print(f"  {DIM}Avg nodes/move       : "
                      f"{total_nodes // max(move_count, 1):,}{RESET}")
                print(f"  {DIM}Total thinking time  : "
                      f"{total_time:.3f}s{RESET}")
                print(f"{DIM}{'─' * 40}{RESET}")
            return

        # Switch player
        current_player = get_opponent(current_player)


def main():
    """Entry point: run games in a loop until the user quits."""
    while True:
        play_game()
        print()
        print(f"{CYAN}Play again? (y/n):{RESET} ", end="")
        try:
            again = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if again not in ("y", "yes"):
            break
        clear_screen()

    print(f"\n{GREEN}Thanks for playing! Goodbye. 👋{RESET}\n")


if __name__ == "__main__":
    main()
