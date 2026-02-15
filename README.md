# Connect Four — Minimax Game-Playing Agent

An intelligent game-playing agent for **Connect Four** that uses the **Minimax algorithm** with **alpha-beta pruning** and **depth-limited heuristic evaluation** to compete against a human player.

Built as part of the Artificial Intelligence Principles course assignment.

## Features

- **Minimax with Alpha-Beta Pruning** — optimal adversarial search with efficient branch elimination
- **Depth-Limited Search** — configurable search depth (Easy / Medium / Hard)
- **Heuristic Evaluation** — intelligent board scoring when the depth limit is reached
- **Interactive CLI** — colored terminal display with Unicode piece symbols (🔴🟡)
- **Player Order Selection** — choose to go first or second
- **Performance Metrics** — nodes expanded, search depth, and computation time displayed after each AI move

## Requirements

- **Python 3.7+**
- **NumPy**

Install the required dependency:

```bash
pip install numpy
```

## How to Run

```bash
python main.py
```

You will be prompted to:
1. Choose whether to go first or second
2. Select a difficulty level (Easy / Medium / Hard)
3. Enter column numbers (1–7) to drop your pieces

## Project Structure

```
Assignment_2/
├── game.py       # Game engine: board, moves, win detection, utility
├── minimax.py    # AI agent: minimax, alpha-beta, heuristic evaluation
├── main.py       # Interactive command-line game interface
├── report.tex    # LaTeX report (2-3 pages)
└── README.md     # This file
```

### Module Descriptions

| Module | Description |
|--------|-------------|
| `game.py` | Core game logic — board representation (6×7 NumPy array), legal move generation, terminal state detection (win/draw), utility function, and colored board display |
| `minimax.py` | AI decision engine — recursive Minimax with alpha-beta pruning, depth-limited search, heuristic evaluation function, move ordering, and performance tracking |
| `main.py` | User interface — game loop, input validation, player order selection, difficulty settings, and result announcements |

## Game Rules

- The board is a 6-row × 7-column vertical grid
- Players take turns dropping pieces into columns
- Pieces fall to the lowest available position in the chosen column
- First to connect **4 in a row** (horizontally, vertically, or diagonally) wins
- If the board fills up with no winner, the game is a draw

## Building the Report

To compile the LaTeX report into a PDF:

```bash
pdflatex report.tex
```

This produces `report.pdf` in the same directory.
