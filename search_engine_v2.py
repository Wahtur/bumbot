"""
search_engine_v2.py
-------------------
Upgraded Adversarial Search Engine for Bumbot (Phase 3):
  - Zobrist Hashing Transposition Table with replacement policy (size 2^19)
  - Iterative Deepening with Principal Variation (PV) leaf state tracking
  - Killer Move Heuristics (2 quiet beta-cutoff moves per search ply)
  - Move Ordering: TT Hash Move -> MVV-LVA Captures -> Killer Moves -> Quiet Moves
  - Extended Quiescence Search (safety cap 16) with Delta Pruning
  - MATE_SCORE handling adjusted for search ply
"""

from typing import Optional, Tuple, List, Dict
import random
import chess
import chess.polyglot
import numpy as np
import torch

import feature_extractor_v2 as feature_extractor

MATE_SCORE = 10000.0
DRAW_SCORE = 0.0

# TT Bound Flags
FLAG_EXACT = 0
FLAG_LOWERBOUND = 1  # Fail-high (beta cutoff)
FLAG_UPPERBOUND = 2  # Fail-low (alpha cutoff)

VICTIM_SCORES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000,
}

MAX_SEARCH_PLY = 64


class TranspositionTable:
    """
    Fixed-capacity Zobrist Transposition Table with depth-preferred replacement policy.
    Maps 64-bit Zobrist key to (key, depth, score, flag, best_move).
    """
    def __init__(self, size_power: int = 19):  # 2^19 = 524,288 entries (~25MB RAM)
        self.size = 1 << size_power
        self.mask = self.size - 1
        # Arrays for contiguous memory or simple dict/slot list
        # We store (key, depth, score, flag, best_move)
        self.table: List[Optional[Tuple[int, int, float, int, Optional[chess.Move]]]] = [None] * self.size
        self.hits = 0
        self.collisions = 0

    def get(self, key: int, depth: int, alpha: float, beta: float) -> Tuple[Optional[float], Optional[chess.Move]]:
        """Look up transposition entry; returns (usable_score, hash_move)."""
        idx = key & self.mask
        entry = self.table[idx]
        if entry is None or entry[0] != key:
            return None, None
            
        _, entry_depth, entry_score, flag, hash_move = entry
        
        # Only return cutoff score if searched depth is >= current requested depth
        if entry_depth >= depth:
            if flag == FLAG_EXACT:
                return entry_score, hash_move
            elif flag == FLAG_LOWERBOUND and entry_score >= beta:
                return entry_score, hash_move
            elif flag == FLAG_UPPERBOUND and entry_score <= alpha:
                return entry_score, hash_move
                
        return None, hash_move

    def store(self, key: int, depth: int, score: float, flag: int, best_move: Optional[chess.Move]):
        """Stores or updates transposition entry using depth-preferred replacement."""
        idx = key & self.mask
        entry = self.table[idx]
        if entry is None or depth >= entry[1]:
            self.table[idx] = (key, depth, score, flag, best_move)

    def clear(self):
        self.table = [None] * self.size
        self.hits = 0
        self.collisions = 0


# Global Transposition Table instance
TT = TranspositionTable(size_power=19)


def evaluate_board(board: chess.Board, weights) -> float:
    """
    Computes static evaluation V(s) = w^T x(s) oriented from White's perspective (+White, -Black).
    Supports NumPy arrays and PyTorch tensors.
    """
    if isinstance(weights, np.ndarray):
        features = feature_extractor.extract_features_numpy(board)
        return float(np.dot(weights, features))
    else:
        features = feature_extractor.extract_features(board)
        return float(torch.dot(weights, features).item())


def score_move(
    board: chess.Board,
    move: chess.Move,
    hash_move: Optional[chess.Move] = None,
    killers: Optional[List[chess.Move]] = None,
) -> int:
    """
    Fast move ordering score:
      1. TT Hash Move: 300,000
      2. Captures (MVV-LVA): 10,000 + 10 * victim - attacker
      3. Pawn Promotions: 9,000
      4. Killer Move 1: 8,000
      5. Killer Move 2: 7,000
      6. Quiet moves: 0
    """
    if hash_move is not None and move == hash_move:
        return 300000

    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        victim_val = VICTIM_SCORES.get(captured_piece.piece_type, 100) if captured_piece else 100
        attacker_piece = board.piece_at(move.from_square)
        attacker_val = VICTIM_SCORES.get(attacker_piece.piece_type, 100) if attacker_piece else 100
        return 10000 + victim_val * 10 - attacker_val

    if move.promotion:
        return 9000

    if killers is not None:
        if len(killers) > 0 and move == killers[0]:
            return 8000
        if len(killers) > 1 and move == killers[1]:
            return 7000

    return 0


def order_moves(
    board: chess.Board,
    hash_move: Optional[chess.Move] = None,
    killers: Optional[List[chess.Move]] = None,
) -> List[chess.Move]:
    """Sorts legal moves descending by heuristic score for optimal cutoffs."""
    moves = list(board.legal_moves)
    if len(moves) <= 1:
        return moves
    moves.sort(key=lambda m: score_move(board, m, hash_move, killers), reverse=True)
    return moves


def quiescence(
    board: chess.Board,
    weights,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    max_qdepth: int = 16,
    node_counter: Optional[List[int]] = None,
) -> float:
    """
    Quiescence search with Delta Pruning and safety cap (max_qdepth=16).
    Delta Pruning skips futile captures where stand_pat + captured_val + 200 < alpha.
    """
    if node_counter is not None:
        node_counter[0] += 1

    stand_pat = evaluate_board(board, weights)

    if is_maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        if max_qdepth <= 0:
            return alpha

        # Generate only captures and promotions
        capture_moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not capture_moves:
            return alpha

        capture_moves.sort(key=lambda m: score_move(board, m), reverse=True)

        for move in capture_moves:
            # Delta Pruning (unless in check or pawn promotion)
            if not board.is_check() and not move.promotion:
                captured = board.piece_at(move.to_square)
                cap_val = VICTIM_SCORES.get(captured.piece_type, 100) if captured else 100
                # Scale cap_val to match feature units (pawns ~ 1.0)
                cap_val_scaled = cap_val / 100.0
                if stand_pat + cap_val_scaled + 2.0 < alpha:
                    continue  # Prune futile capture

            board.push(move)
            score = quiescence(board, weights, alpha, beta, False, max_qdepth - 1, node_counter)
            board.pop()

            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)

        if max_qdepth <= 0:
            return beta

        capture_moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not capture_moves:
            return beta

        capture_moves.sort(key=lambda m: score_move(board, m), reverse=True)

        for move in capture_moves:
            if not board.is_check() and not move.promotion:
                captured = board.piece_at(move.to_square)
                cap_val = VICTIM_SCORES.get(captured.piece_type, 100) if captured else 100
                cap_val_scaled = cap_val / 100.0
                if stand_pat - cap_val_scaled - 2.0 > beta:
                    continue  # Prune futile capture

            board.push(move)
            score = quiescence(board, weights, alpha, beta, True, max_qdepth - 1, node_counter)
            board.pop()

            if score <= alpha:
                return alpha
            beta = min(beta, score)

        return beta


def alpha_beta(
    board: chess.Board,
    weights,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    ply: int = 0,
    killer_table: Optional[List[List[chess.Move]]] = None,
    node_counter: Optional[List[int]] = None,
) -> Tuple[float, Optional[chess.Move], chess.Board]:
    """
    Alpha-Beta search with Transposition Table, Killer Moves, and PV Leaf Tracking.
    Returns: (eval_score, best_move, pv_leaf_board)
    """
    if node_counter is not None:
        node_counter[0] += 1

    orig_alpha = alpha

    # 1. Terminal Check
    if board.is_checkmate():
        val = -(MATE_SCORE - ply) if is_maximizing else (MATE_SCORE - ply)
        return val, None, board.copy()

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition(3)
    ):
        return DRAW_SCORE, None, board.copy()

    # 2. Leaf depth reached: run quiescence search
    if depth <= 0:
        val = quiescence(board, weights, alpha, beta, is_maximizing, max_qdepth=16, node_counter=node_counter)
        return val, None, board.copy()

    # 3. Transposition Table Probe
    key = chess.polyglot.zobrist_hash(board)
    tt_score, hash_move = TT.get(key, depth, alpha, beta)
    if tt_score is not None:
        return tt_score, hash_move, board.copy()

    # 4. Move Generation & Ordering
    killers = killer_table[ply] if (killer_table is not None and ply < len(killer_table)) else None
    ordered_moves = order_moves(board, hash_move=hash_move, killers=killers)

    if not ordered_moves:
        if board.is_check():
            val = -(MATE_SCORE - ply) if is_maximizing else (MATE_SCORE - ply)
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
                board, weights, depth - 1, alpha, beta, False, ply + 1, killer_table, node_counter
            )
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                best_leaf = leaf

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                # Killer Move Store (if quiet move caused beta cutoff)
                if killer_table is not None and ply < len(killer_table) and not board.is_capture(move):
                    k_list = killer_table[ply]
                    if move not in k_list:
                        k_list.insert(0, move)
                        if len(k_list) > 2:
                            k_list.pop()
                break  # Beta cutoff

        # Store in TT
        if max_eval <= orig_alpha:
            flag = FLAG_UPPERBOUND
        elif max_eval >= beta:
            flag = FLAG_LOWERBOUND
        else:
            flag = FLAG_EXACT
        TT.store(key, depth, max_eval, flag, best_move)

        return max_eval, best_move, best_leaf

    else:
        min_eval = float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score, _, leaf = alpha_beta(
                board, weights, depth - 1, alpha, beta, True, ply + 1, killer_table, node_counter
            )
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                best_leaf = leaf

            beta = min(beta, eval_score)
            if beta <= alpha:
                # Killer Move Store
                if killer_table is not None and ply < len(killer_table) and not board.is_capture(move):
                    k_list = killer_table[ply]
                    if move not in k_list:
                        k_list.insert(0, move)
                        if len(k_list) > 2:
                            k_list.pop()
                break  # Alpha cutoff

        # Store in TT
        if min_eval <= orig_alpha:
            flag = FLAG_UPPERBOUND
        elif min_eval >= beta:
            flag = FLAG_LOWERBOUND
        else:
            flag = FLAG_EXACT
        TT.store(key, depth, min_eval, flag, best_move)

        return min_eval, best_move, best_leaf


def get_best_move(
    board: chess.Board,
    weights,
    depth: int = 3,
) -> Tuple[Optional[chess.Move], float, chess.Board, int]:
    """
    Iterative Deepening root search from depth 1 to requested depth.
    Leverages Transposition Table move ordering across successive iterations.
    
    Returns:
        (best_move, evaluation, principal_variation_leaf_board, nodes_searched)
    """
    is_maximizing = (board.turn == chess.WHITE)
    counter = [0]
    killer_table: List[List[chess.Move]] = [[] for _ in range(MAX_SEARCH_PLY)]

    best_move: Optional[chess.Move] = None
    best_eval: float = 0.0
    best_leaf: chess.Board = board

    for d in range(1, depth + 1):
        eval_score, move, leaf = alpha_beta(
            board=board,
            weights=weights,
            depth=d,
            alpha=-float("inf"),
            beta=float("inf"),
            is_maximizing=is_maximizing,
            ply=0,
            killer_table=killer_table,
            node_counter=counter,
        )
        if move is not None:
            best_move = move
            best_eval = eval_score
            best_leaf = leaf

    return best_move, best_eval, best_leaf, counter[0]


def get_best_move_sampled(
    board: chess.Board,
    weights,
    depth: int = 2,
    temperature: float = 0.0,
) -> Tuple[Optional[chess.Move], float, chess.Board, int]:
    """
    Root Move Selection with Softmax Temperature Exploration on PRE-SQUASHED linear scores.
    If temperature <= 0.01: executes deterministic iterative deepening minimax search.
    If temperature > 0.01: samples from Boltzmann distribution over candidate legal moves.
    """
    if temperature <= 0.01:
        return get_best_move(board, weights, depth=depth)

    ordered_moves = order_moves(board)
    if not ordered_moves:
        return None, 0.0, board, 1

    if len(ordered_moves) == 1:
        move = ordered_moves[0]
        board.push(move)
        leaf = board.copy()
        board.pop()
        val = evaluate_board(board, weights)
        return move, val, leaf, 1

    is_maximizing = (board.turn == chess.WHITE)
    counter = [0]
    move_scores = []
    leaves = []

    sub_depth = max(0, depth - 1)
    killer_table: List[List[chess.Move]] = [[] for _ in range(MAX_SEARCH_PLY)]

    for move in ordered_moves:
        board.push(move)
        eval_score, _, leaf = alpha_beta(
            board=board,
            weights=weights,
            depth=sub_depth,
            alpha=-float("inf"),
            beta=float("inf"),
            is_maximizing=not is_maximizing,
            ply=1,
            killer_table=killer_table,
            node_counter=counter,
        )
        board.pop()

        q_score = eval_score if is_maximizing else -eval_score
        move_scores.append(q_score)
        leaves.append(leaf)

    scores_arr = np.array(move_scores, dtype=np.float64)
    max_q = float(np.max(scores_arr))
    scaled = (scores_arr - max_q) / max(0.01, float(temperature))
    scaled = np.clip(scaled, -50.0, 50.0)
    exp_scores = np.exp(scaled)
    sum_exp = float(np.sum(exp_scores))

    if sum_exp <= 1e-12 or np.isnan(sum_exp):
        probs = np.ones(len(ordered_moves), dtype=np.float64) / len(ordered_moves)
    else:
        probs = exp_scores / sum_exp

    chosen_idx = int(np.random.choice(len(ordered_moves), p=probs))
    return ordered_moves[chosen_idx], move_scores[chosen_idx], leaves[chosen_idx], counter[0]
