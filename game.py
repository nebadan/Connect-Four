"""
game.py — Connect Four Game Engine

This module provides the core game logic for a Connect Four game,
including board representation, move generation, terminal state detection,
and utility evaluation.

Game Properties:
    - Deterministic: No randomness involved
    - Turn-based: Players alternate moves
    - Zero-sum: One player's gain is the other's loss
    - Perfect information: The entire board state is visible to both players

Board Representation:
    A 6x7 NumPy array where:
        0 = Empty cell
        1 = Player 1 (Human by default, uses 🔴)
        2 = Player 2 (AI by default, uses 🟡)

    Row 0 is the TOP of the board, Row 5 is the BOTTOM.
    Pieces "drop" to the lowest available row in a column.
"""

import numpy as np

# Board dimensions
ROWS = 6
COLS = 7

# Cell states
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = 2

# Number of pieces in a row needed to win
WIN_LENGTH = 4


def create_board():
    """
    Create and return an empty Connect Four board.

    Returns:
        np.ndarray: A 6x7 array of zeros representing an empty board.
    """
    return np.zeros((ROWS, COLS), dtype=int)


def get_legal_moves(board):
    """
    Determine which columns are available for a move.

    A column is legal if its top row (row 0) is empty,
    meaning there is at least one open cell.

    Args:
        board (np.ndarray): The current board state.

    Returns:
        list[int]: A list of column indices where a piece can be dropped.
    """
    return [col for col in range(COLS) if board[0][col] == EMPTY]


def make_move(board, col, player):
    """
    Drop a piece into the specified column for the given player.

    The piece falls to the lowest available row in the column.
    Returns a new board (does not modify the original).

    Args:
        board (np.ndarray): The current board state.
        col (int): The column index to drop the piece into (0-6).
        player (int): The player making the move (PLAYER_1 or PLAYER_2).

    Returns:
        np.ndarray: A new board state with the piece placed.

    Raises:
        ValueError: If the column is full or out of bounds.
    """
    if col < 0 or col >= COLS:
        raise ValueError(f"Column {col} is out of bounds (0-{COLS - 1}).")
    if board[0][col] != EMPTY:
        raise ValueError(f"Column {col} is full.")

    new_board = board.copy()
    # Find the lowest empty row in this column
    for row in range(ROWS - 1, -1, -1):
        if new_board[row][col] == EMPTY:
            new_board[row][col] = player
            break
    return new_board


def _check_window(window, player):
    """
    Check if a window of 4 cells constitutes a win for the given player.

    Args:
        window (np.ndarray): A 1D array of 4 cell values.
        player (int): The player to check for.

    Returns:
        bool: True if all 4 cells belong to the player.
    """
    return np.all(window == player)


def get_winner(board):
    """
    Determine if there is a winner on the current board.

    Checks all horizontal, vertical, and diagonal lines of 4.

    Args:
        board (np.ndarray): The current board state.

    Returns:
        int or None: The winning player (PLAYER_1 or PLAYER_2), or None.
    """
    # Check horizontal windows
    for row in range(ROWS):
        for col in range(COLS - WIN_LENGTH + 1):
            window = board[row, col:col + WIN_LENGTH]
            for player in [PLAYER_1, PLAYER_2]:
                if _check_window(window, player):
                    return player

    # Check vertical windows
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(COLS):
            window = board[row:row + WIN_LENGTH, col]
            for player in [PLAYER_1, PLAYER_2]:
                if _check_window(window, player):
                    return player

    # Check positively sloped diagonals (\)
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(COLS - WIN_LENGTH + 1):
            window = np.array([board[row + i][col + i] for i in range(WIN_LENGTH)])
            for player in [PLAYER_1, PLAYER_2]:
                if _check_window(window, player):
                    return player

    # Check negatively sloped diagonals (/)
    for row in range(WIN_LENGTH - 1, ROWS):
        for col in range(COLS - WIN_LENGTH + 1):
            window = np.array([board[row - i][col + i] for i in range(WIN_LENGTH)])
            for player in [PLAYER_1, PLAYER_2]:
                if _check_window(window, player):
                    return player

    return None


def is_terminal(board):
    """
    Determine if the game has reached a terminal state.

    A state is terminal if:
        - A player has won (4 in a row), OR
        - The board is full (draw)

    Args:
        board (np.ndarray): The current board state.

    Returns:
        bool: True if the game is over.
    """
    return get_winner(board) is not None or len(get_legal_moves(board)) == 0


def utility(board, ai_player):
    """
    Compute the utility value of a terminal board state.

    This function assigns numerical outcomes based on the Minimax framework:
        +1 if the AI player wins
        -1 if the opponent wins
         0 if the game is a draw

    Args:
        board (np.ndarray): The current (terminal) board state.
        ai_player (int): The player number of the AI (PLAYER_1 or PLAYER_2).

    Returns:
        int: The utility value (+1, -1, or 0).
    """
    winner = get_winner(board)
    if winner == ai_player:
        return 1
    elif winner is not None:
        return -1
    else:
        return 0


def get_opponent(player):
    """
    Return the opponent of the given player.

    Args:
        player (int): PLAYER_1 or PLAYER_2.

    Returns:
        int: The opponent player.
    """
    return PLAYER_1 if player == PLAYER_2 else PLAYER_2


def print_board(board):
    """
    Display the board in the terminal with colored pieces.

    Uses ANSI color codes for a visually appealing display:
        🔴 (Red) for Player 1
        🟡 (Yellow) for Player 2
        ⚫ (Dark circle) for empty cells

    The column numbers are displayed below the board for easy reference.

    Args:
        board (np.ndarray): The current board state.
    """
    # ANSI color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    symbols = {
        EMPTY: "⚫",
        PLAYER_1: f"{RED}🔴{RESET}",
        PLAYER_2: f"{YELLOW}🟡{RESET}",
    }

    print()
    print(f"  {BLUE}{BOLD}╔{'═══╦' * (COLS - 1)}═══╗{RESET}")
    for row_idx, row in enumerate(board):
        row_str = f"  {BLUE}{BOLD}║{RESET}"
        for cell in row:
            row_str += f" {symbols[cell]} {BLUE}{BOLD}║{RESET}"
        print(row_str)
        if row_idx < ROWS - 1:
            print(f"  {BLUE}{BOLD}╠{'═══╬' * (COLS - 1)}═══╣{RESET}")
    print(f"  {BLUE}{BOLD}╚{'═══╩' * (COLS - 1)}═══╝{RESET}")

    # Column numbers
    col_nums = "   "
    for col in range(COLS):
        col_nums += f" {BOLD}{col + 1}{RESET}   "
    print(col_nums)
    print()
