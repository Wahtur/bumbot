"""
search_engine.py
----------------
Owner: Partner A (Search & Features)

Implements the adversarial search engine for the chess bot using Minimax
with Alpha-Beta pruning.

Key Features:
  - Alpha-Beta Pruning: Prunes provably sub-optimal game tree branches.
  - Principal Variation Leaf Tracking: Crucial for Baxter et al.'s TD-Leaf(lambda);
    tracks and returns the exact leaf board state chosen by the minimax search.
  - Move Ordering (MVV-LVA): Sorts captures first (Most Valuable Victim - Least
    Valuable Attacker) to dramatically improve alpha-beta cutoff efficiency.
  - Terminal State Resolution: Accurately evaluates checkmate and draws (stalemate,
    repetition, insufficient material, 50-move rule).
"""

from typing import Optional, Tuple, List
import random
import chess
import torch
import numpy as np

import feature_extractor

# High terminal values for checkmate (adjusted for depth to prefer faster mates)
MATE_SCORE = 10000.0
DRAW_SCORE = 0.0

# Piece values for MVV-LVA move ordering
VICTIM_SCORES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000,
}


def score_move(board: chess.Board, move: chess.Move) -> int:
    """
    Heuristic move prioritization for Alpha-Beta move ordering:
      - MVV-LVA for captures
      - Promotion bonus
      - Checks bonus
    """
    score = 0
    if board.is_capture(move):
        # Determine victim
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is not None:
            victim_val = VICTIM_SCORES.get(captured_piece.piece_type, 0)
        else:
            # En passant capture
            victim_val = VICTIM_SCORES[chess.PAWN]
            
        # Attacker
        attacker_piece = board.piece_at(move.from_square)
        attacker_val = VICTIM_SCORES.get(attacker_piece.piece_type, 0) if attacker_piece else 0
        
        # MVV-LVA score
        score = 10000 + victim_val * 10 - attacker_val
    elif move.promotion:
        score = 9000
    elif board.gives_check(move):
        score = 1000
        
    return score


def order_moves(board: chess.Board) -> List[chess.Move]:
    """Sorts legal moves descending by heuristic priority to maximize cutoffs."""
    moves = list(board.legal_moves)
    # Fast path if 1 or 0 moves
    if len(moves) <= 1:
        return moves
    moves.sort(key=lambda m: score_move(board, m), reverse=True)
    return moves


def evaluate_board(board: chess.Board, weights) -> float:
    """
    Evaluates the static board state using the dot product of features and weights:
        V(s) = w . f(s)
    Score is oriented from White's perspective (+White, -Black).
    Supports both NumPy arrays (faster in multiprocessing) and PyTorch tensors.
    """
    if isinstance(weights, np.ndarray):
        features = feature_extractor.extract_features_numpy(board)
        return float(np.dot(weights, features))
    else:
        features = feature_extractor.extract_features(board)
        return float(torch.dot(weights, features).item())


def alpha_beta(
    board: chess.Board,
    weights: torch.Tensor,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    node_counter: Optional[List[int]] = None,
) -> Tuple[float, Optional[chess.Move], chess.Board]:
    """
    Recursive Alpha-Beta minimax search.

    Returns:
        (best_eval, best_move, leaf_board_state)
    """
    if node_counter is not None:
        node_counter[0] += 1

    # Check terminal game states
    if board.is_checkmate():
        # If maximizing (White's turn) and in checkmate, White lost
        # Depth penalty encourages finding the quickest checkmate
        val = -(MATE_SCORE + depth) if is_maximizing else (MATE_SCORE + depth)
        return val, None, board.copy()

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition(3)
    ):
        return DRAW_SCORE, None, board.copy()

    # Leaf depth reached: evaluate static board state
    if depth <= 0:
        val = evaluate_board(board, weights)
        return val, None, board.copy()

    ordered_moves = order_moves(board)
    if not ordered_moves:
        # No legal moves available
        if board.is_check():
            val = -(MATE_SCORE + depth) if is_maximizing else (MATE_SCORE + depth)
        else:
            val = DRAW_SCORE
        return val, None, board.copy()

    best_move: Optional[chess.Move] = None
    best_leaf: chess.Board = board

    if is_maximizing:
        max_eval = -float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score, _, leaf = alpha_beta(
                board, weights, depth - 1, alpha, beta, False, node_counter
            )
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                best_leaf = leaf

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff

        return max_eval, best_move, best_leaf

    else:
        min_eval = float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score, _, leaf = alpha_beta(
                board, weights, depth - 1, alpha, beta, True, node_counter
            )
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                best_leaf = leaf

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff

        return min_eval, best_move, best_leaf


def get_best_move(
    board: chess.Board,
    weights: torch.Tensor,
    depth: int = 3,
) -> Tuple[Optional[chess.Move], float, chess.Board, int]:
    """
    Root search call for a given board position.
    
    Returns:
        (best_move, evaluation, leaf_board, nodes_searched)
    """
    is_maximizing = (board.turn == chess.WHITE)
    counter = [0]
    eval_score, best_move, leaf_board = alpha_beta(
        board=board,
        weights=weights,
        depth=depth,
        alpha=-float("inf"),
        beta=float("inf"),
        is_maximizing=is_maximizing,
        node_counter=counter,
    )
    return best_move, eval_score, leaf_board, counter[0]
