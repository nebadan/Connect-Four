"""
minimax.py — Minimax Algorithm with Alpha-Beta Pruning

This module implements the Minimax decision rule for a Connect Four AI agent.
The agent assumes the opponent plays optimally and selects actions to maximize
its minimum guaranteed outcome.

Key Features:
    1. Recursive Minimax with alpha-beta pruning for efficient search
    2. Depth-limited search to handle the large game tree of Connect Four
    3. Heuristic evaluation function for non-terminal board states
    4. Move ordering (center-column preference) to improve pruning efficiency
    5. Performance metrics tracking (nodes expanded, search time)

Algorithm Overview:
    - The Maximizing player (AI) aims to maximize the board evaluation.
    - The Minimizing player (Human) aims to minimize the board evaluation.
    - Alpha-beta pruning eliminates branches that cannot influence the final
      decision, significantly reducing the number of nodes explored.
    - When the depth limit is reached, a heuristic evaluation function
      estimates the board's favorability instead of searching further.
"""

import time
import numpy as np
from game import (
    ROWS, COLS, EMPTY, WIN_LENGTH,
    get_legal_moves, make_move, is_terminal, get_winner,
    utility, get_opponent,
)


# ──────────────────────────────────────────────
#  Heuristic Evaluation Function
# ──────────────────────────────────────────────

def _score_window(window, ai_player, opponent):
    """
    Evaluate a window of 4 cells and return a heuristic score.

    Scoring criteria:
        +100  : 4 AI pieces (win)
        +10   : 3 AI pieces + 1 empty (strong threat)
        +5    : 2 AI pieces + 2 empty (potential buildup)
        -8    : 3 opponent pieces + 1 empty (must block)
        -4    : 2 opponent pieces + 2 empty (opponent buildup)

    The asymmetric scoring (blocking threats scored slightly lower than
    own threats) encourages offensive play while still prioritizing defense.

    Args:
        window (np.ndarray): Array of 4 cell values.
        ai_player (int): The AI's player number.
        opponent (int): The opponent's player number.

    Returns:
        int: The heuristic score for this window.
    """
    score = 0
    ai_count = np.count_nonzero(window == ai_player)
    opp_count = np.count_nonzero(window == opponent)
    empty_count = np.count_nonzero(window == EMPTY)

    if ai_count == 4:
        score += 100
    elif ai_count == 3 and empty_count == 1:
        score += 10
    elif ai_count == 2 and empty_count == 2:
        score += 5

    if opp_count == 3 and empty_count == 1:
        score -= 8
    elif opp_count == 2 and empty_count == 2:
        score -= 4

    return score


def heuristic_evaluate(board, ai_player):
    """
    Evaluate a non-terminal board state using a heuristic function.

    This function is used when the search depth limit is reached and the
    board is not in a terminal state. It estimates the board's favorability
    by examining every possible window of 4 cells across all directions.

    The evaluation considers:
        1. Center column preference — pieces in the center column have
           more potential connections, so they receive a bonus.
        2. Horizontal windows — all consecutive 4-cell groups in each row.
        3. Vertical windows — all consecutive 4-cell groups in each column.
        4. Diagonal windows — both positive and negative slope diagonals.

    Args:
        board (np.ndarray): The current board state.
        ai_player (int): The AI's player number.

    Returns:
        float: A heuristic score (positive = favorable for AI,
               negative = favorable for opponent).
    """
    opponent = get_opponent(ai_player)
    score = 0

    # Center column preference (center pieces have more connections)
    center_col = COLS // 2
    center_array = board[:, center_col]
    center_count = np.count_nonzero(center_array == ai_player)
    score += center_count * 6

    # Score horizontal windows
    for row in range(ROWS):
        for col in range(COLS - WIN_LENGTH + 1):
            window = board[row, col:col + WIN_LENGTH]
            score += _score_window(window, ai_player, opponent)

    # Score vertical windows
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(COLS):
            window = board[row:row + WIN_LENGTH, col]
            score += _score_window(window, ai_player, opponent)

    # Score positively sloped diagonals (\)
    for row in range(ROWS - WIN_LENGTH + 1):
        for col in range(COLS - WIN_LENGTH + 1):
            window = np.array([board[row + i][col + i] for i in range(WIN_LENGTH)])
            score += _score_window(window, ai_player, opponent)

    # Score negatively sloped diagonals (/)
    for row in range(WIN_LENGTH - 1, ROWS):
        for col in range(COLS - WIN_LENGTH + 1):
            window = np.array([board[row - i][col + i] for i in range(WIN_LENGTH)])
            score += _score_window(window, ai_player, opponent)

    return score


# ──────────────────────────────────────────────
#  Minimax Algorithm with Alpha-Beta Pruning
# ──────────────────────────────────────────────

def minimax(board, depth, maximizing, ai_player, alpha, beta, stats):
    """
    Recursive Minimax algorithm with alpha-beta pruning.

    This function implements the core adversarial search logic:
        - At MAX nodes (AI's turn): choose the action that maximizes value.
        - At MIN nodes (opponent's turn): choose the action that minimizes value.
        - Alpha-beta pruning: skip branches that cannot affect the outcome.

    The search terminates when:
        1. A terminal state is reached (win/loss/draw) → return utility
        2. The depth limit is reached → return heuristic evaluation

    Parameters:
        board (np.ndarray): The current board state.
        depth (int): Remaining search depth (decrements each level).
        maximizing (bool): True if current player is the maximizer (AI).
        ai_player (int): The AI's player number.
        alpha (float): Best value the maximizer can guarantee (lower bound).
        beta (float): Best value the minimizer can guarantee (upper bound).
        stats (dict): Dictionary tracking 'nodes_expanded' count.

    Returns:
        float: The minimax value of this board state.

    Algorithm:
        ```
        function MINIMAX(state, depth, isMax, α, β):
            if TERMINAL(state) or depth == 0:
                return EVALUATE(state)
            if isMax:
                value = -∞
                for each action in ACTIONS(state):
                    value = max(value, MINIMAX(result, depth-1, false, α, β))
                    α = max(α, value)
                    if α ≥ β: break   // β cutoff
                return value
            else:
                value = +∞
                for each action in ACTIONS(state):
                    value = min(value, MINIMAX(result, depth-1, true, α, β))
                    β = min(β, value)
                    if α ≥ β: break   // α cutoff
                return value
        ```
    """
    stats["nodes_expanded"] += 1

    # Base case: terminal state — return exact utility
    if is_terminal(board):
        winner = get_winner(board)
        if winner == ai_player:
            # Prefer faster wins (add depth bonus)
            return 1000 + depth
        elif winner is not None:
            # Prefer slower losses (subtract depth penalty)
            return -1000 - depth
        else:
            return 0  # Draw

    # Base case: depth limit reached — return heuristic estimate
    if depth == 0:
        return heuristic_evaluate(board, ai_player)

    # Get legal moves, ordered by center preference for better pruning
    legal_moves = get_legal_moves(board)
    legal_moves = _order_moves(legal_moves)

    opponent = get_opponent(ai_player)

    if maximizing:
        # AI's turn: maximize the evaluation
        max_eval = float("-inf")
        for col in legal_moves:
            child_board = make_move(board, col, ai_player)
            eval_score = minimax(child_board, depth - 1, False,
                                 ai_player, alpha, beta, stats)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if alpha >= beta:
                break  # Beta cutoff — minimizer would never allow this
        return max_eval
    else:
        # Opponent's turn: minimize the evaluation
        min_eval = float("inf")
        for col in legal_moves:
            child_board = make_move(board, col, opponent)
            eval_score = minimax(child_board, depth - 1, True,
                                 ai_player, alpha, beta, stats)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if alpha >= beta:
                break  # Alpha cutoff — maximizer would never allow this
        return min_eval


def _order_moves(moves):
    """
    Order moves by proximity to the center column.

    Center columns are explored first because they typically lead to
    stronger positions, which improves alpha-beta pruning efficiency.

    Args:
        moves (list[int]): List of legal column indices.

    Returns:
        list[int]: Sorted list with center columns first.
    """
    center = COLS // 2
    return sorted(moves, key=lambda col: abs(col - center))


# ──────────────────────────────────────────────
#  Public API: Get Best Move
# ──────────────────────────────────────────────

def get_best_move(board, ai_player, max_depth=6):
    """
    Determine the best move for the AI using depth-limited Minimax
    with alpha-beta pruning.

    This is the entry point for the AI agent. It evaluates all legal moves
    and returns the column that leads to the highest minimax value.

    Args:
        board (np.ndarray): The current board state.
        ai_player (int): The AI's player number (PLAYER_1 or PLAYER_2).
        max_depth (int): Maximum search depth (default: 6).
                         Higher values = stronger play but slower.

    Returns:
        tuple: (best_column, stats_dict)
            - best_column (int): The column index of the best move.
            - stats_dict (dict): Performance metrics:
                - 'nodes_expanded' (int): Total nodes explored in the search.
                - 'time_seconds' (float): Wall-clock time taken for the search.
                - 'depth' (int): The search depth used.
    """
    stats = {"nodes_expanded": 0}
    start_time = time.time()

    legal_moves = get_legal_moves(board)
    legal_moves = _order_moves(legal_moves)

    best_score = float("-inf")
    best_col = legal_moves[0]  # Default to first legal move

    for col in legal_moves:
        child_board = make_move(board, col, ai_player)
        score = minimax(child_board, max_depth - 1, False,
                        ai_player, float("-inf"), float("inf"), stats)
        if score > best_score:
            best_score = score
            best_col = col

    elapsed = time.time() - start_time
    stats["time_seconds"] = round(elapsed, 3)
    stats["depth"] = max_depth

    return best_col, stats
