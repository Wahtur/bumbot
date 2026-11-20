"""
search_engine.py
----------------
Upgraded Adversarial Search Engine for Bumbot V2 (Phase 3):
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

import feature_extractor

MATE_SCORE = 10000.0
DRAW_SCORE = 0.0

FLAG_EXACT = 0
FLAG_LOWERBOUND = 1
FLAG_UPPERBOUND = 2

VICTIM_SCORES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 10000,
}

MAX_SEARCH_PLY = 64


def score_to_tt(score: float, ply: int) -> float:
    """Encodes mate score relative to current search node before TT storage."""
    if score >= MATE_SCORE - MAX_SEARCH_PLY:
        return score + ply
    elif score <= -MATE_SCORE + MAX_SEARCH_PLY:
        return score - ply
    return score


def score_from_tt(score: float, ply: int) -> float:
    """Decodes mate score from TT relative to root ply on retrieval."""
    if score >= MATE_SCORE - MAX_SEARCH_PLY:
        return score - ply
    elif score <= -MATE_SCORE + MAX_SEARCH_PLY:
        return score + ply
    return score


class TranspositionTable:
    def __init__(self, size_power: int = 19):
        self.size = 1 << size_power
        self.mask = self.size - 1
        self.table: List[Optional[Tuple[int, int, float, int, Optional[chess.Move]]]] = [None] * self.size

    def get(self, key: int, depth: int, alpha: float, beta: float, ply: int = 0) -> Tuple[Optional[float], Optional[chess.Move]]:
        idx = key & self.mask
        entry = self.table[idx]
        if entry is None or entry[0] != key:
            return None, None
            
        _, entry_depth, raw_score, flag, hash_move = entry
        score = score_from_tt(raw_score, ply)
        
        if entry_depth >= depth:
            if flag == FLAG_EXACT:
                return score, hash_move
            elif flag == FLAG_LOWERBOUND and score >= beta:
                return score, hash_move
            elif flag == FLAG_UPPERBOUND and score <= alpha:
                return score, hash_move
                
        return None, hash_move

    def store(self, key: int, depth: int, score: float, flag: int, best_move: Optional[chess.Move], ply: int = 0):
        idx = key & self.mask
        entry = self.table[idx]
        tt_score = score_to_tt(score, ply)
        if entry is None or depth >= entry[1]:
            self.table[idx] = (key, depth, tt_score, flag, best_move)

    def clear(self):
        self.table = [None] * self.size


TT = TranspositionTable(size_power=19)


def evaluate_board(board: chess.Board, weights) -> float:
    dim = len(weights)
    if dim == 800:
        if isinstance(weights, np.ndarray):
            features = feature_extractor.extract_features_numpy(board)
            return float(np.dot(weights, features))
        else:
            features = feature_extractor.extract_features(board)
            return float(torch.dot(weights, features).item())
    elif dim == 808:
        try:
            import feature_extractor_v2
            if isinstance(weights, np.ndarray):
                features = feature_extractor_v2.extract_features_numpy(board)
                return float(np.dot(weights, features))
            else:
                features = feature_extractor_v2.extract_features(board)
                return float(torch.dot(weights, features).item())
        except Exception:
            return 0.0
    else:
        try:
            import feature_extractor_v1
            if isinstance(weights, np.ndarray):
                features = feature_extractor_v1.extract_features_numpy(board)
                return float(np.dot(weights, features))
            else:
                features = feature_extractor_v1.extract_features(board)
                return float(torch.dot(weights, features).item())
        except Exception:
            return 0.0


def score_move(
    board: chess.Board,
    move: chess.Move,
    hash_move: Optional[chess.Move] = None,
    killers: Optional[List[chess.Move]] = None,
    history_table: Optional[List[List[List[int]]]] = None,
) -> int:
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

    if history_table is not None:
        turn_idx = 0 if board.turn == chess.WHITE else 1
        h_val = history_table[turn_idx][move.from_square][move.to_square]
        if h_val > 0:
            return min(6000, h_val)

    return 0


def order_moves(
    board: chess.Board,
    hash_move: Optional[chess.Move] = None,
    killers: Optional[List[chess.Move]] = None,
    history_table: Optional[List[List[List[int]]]] = None,
) -> List[chess.Move]:
    moves = list(board.legal_moves)
    if len(moves) <= 1:
        return moves
    moves.sort(key=lambda m: score_move(board, m, hash_move, killers, history_table), reverse=True)
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
    if node_counter is not None:
        node_counter[0] += 1

    stand_pat = evaluate_board(board, weights)

    if is_maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        if max_qdepth <= 0:
            return alpha

        capture_moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not capture_moves:
            return alpha

        capture_moves.sort(key=lambda m: score_move(board, m), reverse=True)
        has_promo = bool(board.pieces(chess.PAWN, board.turn) & (chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2))

        for move in capture_moves:
            if not board.is_check() and not move.promotion and not has_promo:
                captured = board.piece_at(move.to_square)
                cap_val = VICTIM_SCORES.get(captured.piece_type, 100) if captured else 100
                cap_val_scaled = cap_val / 100.0
                if stand_pat + cap_val_scaled + 2.0 < alpha:
                    continue

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
        has_promo = bool(board.pieces(chess.PAWN, board.turn) & (chess.BB_RANK_7 if board.turn == chess.WHITE else chess.BB_RANK_2))

        for move in capture_moves:
            if not board.is_check() and not move.promotion and not has_promo:
                captured = board.piece_at(move.to_square)
                cap_val = VICTIM_SCORES.get(captured.piece_type, 100) if captured else 100
                cap_val_scaled = cap_val / 100.0
                if stand_pat - cap_val_scaled - 2.0 > beta:
                    continue

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
    null_move_allowed: bool = True,
    history_table: Optional[List[List[List[int]]]] = None,
) -> Tuple[float, Optional[chess.Move], chess.Board]:
    if node_counter is not None:
        node_counter[0] += 1

    orig_alpha = alpha

    # 1. Checkmate check (exact mate distance relative to root)
    if board.is_checkmate():
        val = -(MATE_SCORE - ply) if is_maximizing else (MATE_SCORE - ply)
        return val, None, board.copy()

    # 2. Path-aware repetition detection (prevents TT repetition blindness & shuffling)
    # Check 2-fold repetition along current game/search path
    if ply > 0 and board.is_repetition(2):
        # Contempt logic:
        # If ahead in material (> 0.3), penalize repetition heavily (-0.30) to force conversion
        # If roughly equal (|eval| <= 0.3), penalize repetition slightly (-0.08) to avoid opening loops
        # If behind, repetition is a welcome saving resource (0.0)
        mat_val = evaluate_board(board, weights)
        if is_maximizing:
            if mat_val > 0.3:
                rep_score = -0.30
            elif mat_val >= -0.3:
                rep_score = -0.08
            else:
                rep_score = 0.0
        else:
            if mat_val < -0.3:
                rep_score = 0.30
            elif mat_val <= 0.3:
                rep_score = 0.08
            else:
                rep_score = 0.0
        return rep_score, None, board.copy()

    # 3. Standard draw conditions
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
        or board.is_repetition(3)
    ):
        return DRAW_SCORE, None, board.copy()

    # Check extension (+1 ply): extend search depth when in check to resolve tactical threats
    in_check = board.is_check()
    if in_check and ply < MAX_SEARCH_PLY - 4:
        depth += 1

    if depth <= 0:
        val = quiescence(board, weights, alpha, beta, is_maximizing, max_qdepth=16, node_counter=node_counter)
        return val, None, board.copy()

    # 4. Transposition Table lookup with relative mate decoding
    key = chess.polyglot.zobrist_hash(board)
    tt_score, hash_move = TT.get(key, depth, alpha, beta, ply=ply)
    if tt_score is not None:
        return tt_score, hash_move, board.copy()

    # 5. Null-Move Pruning (NMP)
    # At depth >= 2, non-root, not in check, if static eval beats beta/alpha, test null move
    if depth >= 2 and ply > 0 and not in_check and null_move_allowed:
        side = board.turn
        has_pieces = bool(
            board.pieces(chess.KNIGHT, side)
            | board.pieces(chess.BISHOP, side)
            | board.pieces(chess.ROOK, side)
            | board.pieces(chess.QUEEN, side)
        )
        if has_pieces:
            stand_pat = evaluate_board(board, weights)
            r = 2 if depth >= 3 else 1
            if is_maximizing and stand_pat >= beta:
                board.push(chess.Move.null())
                null_val, _, _ = alpha_beta(
                    board, weights, depth - 1 - r, beta - 0.01, beta, False,
                    ply + 1, killer_table, node_counter, null_move_allowed=False, history_table=history_table
                )
                board.pop()
                if null_val >= beta:
                    return beta, None, board.copy()
            elif (not is_maximizing) and stand_pat <= alpha:
                board.push(chess.Move.null())
                null_val, _, _ = alpha_beta(
                    board, weights, depth - 1 - r, alpha, alpha + 0.01, True,
                    ply + 1, killer_table, node_counter, null_move_allowed=False, history_table=history_table
                )
                board.pop()
                if null_val <= alpha:
                    return alpha, None, board.copy()

    killers = killer_table[ply] if (killer_table is not None and ply < len(killer_table)) else None
    ordered_moves = order_moves(board, hash_move=hash_move, killers=killers, history_table=history_table)

    if not ordered_moves:
        if in_check:
            val = -(MATE_SCORE - ply) if is_maximizing else (MATE_SCORE - ply)
        else:
            val = DRAW_SCORE
        return val, None, board.copy()

    best_move: Optional[chess.Move] = None
    best_leaf: chess.Board = board

    if is_maximizing:
        max_eval = -float("inf")
        for move_idx, move in enumerate(ordered_moves):
            is_cap = board.is_capture(move)
            is_promo = bool(move.promotion)
            is_killer = (killers is not None and move in killers)
            gives_chk = board.gives_check(move)

            # Late Move Reductions (LMR): reduce depth for quiet moves ordered late
            can_reduce = (
                depth >= 3
                and move_idx >= 3
                and not is_cap
                and not is_promo
                and not is_killer
                and not in_check
                and not gives_chk
            )
            reduction = 1 if can_reduce else 0
            if can_reduce and depth >= 5 and move_idx >= 6:
                reduction = 2

            board.push(move)

            if reduction > 0:
                eval_score, _, leaf = alpha_beta(
                    board, weights, depth - 1 - reduction, alpha, beta, False,
                    ply + 1, killer_table, node_counter, null_move_allowed=True, history_table=history_table
                )
                if eval_score > alpha:
                    eval_score, _, leaf = alpha_beta(
                        board, weights, depth - 1, alpha, beta, False,
                        ply + 1, killer_table, node_counter, null_move_allowed=True, history_table=history_table
                    )
            else:
                eval_score, _, leaf = alpha_beta(
                    board, weights, depth - 1, alpha, beta, False,
                    ply + 1, killer_table, node_counter, null_move_allowed=True, history_table=history_table
                )
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                best_leaf = leaf

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                if not is_cap and not is_promo:
                    if killer_table is not None and ply < len(killer_table):
                        k_list = killer_table[ply]
                        if move not in k_list:
                            k_list.insert(0, move)
                            if len(k_list) > 2:
                                k_list.pop()
                    if history_table is not None:
                        history_table[0][move.from_square][move.to_square] += depth * depth
                break

        if max_eval <= orig_alpha:
            flag = FLAG_UPPERBOUND
        elif max_eval >= beta:
            flag = FLAG_LOWERBOUND
        else:
            flag = FLAG_EXACT
        TT.store(key, depth, max_eval, flag, best_move, ply=ply)
        return max_eval, best_move, best_leaf

    else:
        min_eval = float("inf")
        for move_idx, move in enumerate(ordered_moves):
            is_cap = board.is_capture(move)
            is_promo = bool(move.promotion)
            is_killer = (killers is not None and move in killers)
            gives_chk = board.gives_check(move)

            can_reduce = (
                depth >= 3
                and move_idx >= 3
                and not is_cap
                and not is_promo
                and not is_killer
                and not in_check
                and not gives_chk
            )
            reduction = 1 if can_reduce else 0
            if can_reduce and depth >= 5 and move_idx >= 6:
                reduction = 2

            board.push(move)

            if reduction > 0:
                eval_score, _, leaf = alpha_beta(
                    board, weights, depth - 1 - reduction, alpha, beta, True,
                    ply + 1, killer_table, node_counter, null_move_allowed=True, history_table=history_table
                )
                if eval_score < beta:
                    eval_score, _, leaf = alpha_beta(
                        board, weights, depth - 1, alpha, beta, True,
                        ply + 1, killer_table, node_counter, null_move_allowed=True, history_table=history_table
                    )
            else:
                eval_score, _, leaf = alpha_beta(
                    board, weights, depth - 1, alpha, beta, True,
                    ply + 1, killer_table, node_counter, null_move_allowed=True, history_table=history_table
                )
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                best_leaf = leaf

            beta = min(beta, eval_score)
            if beta <= alpha:
                if not is_cap and not is_promo:
                    if killer_table is not None and ply < len(killer_table):
                        k_list = killer_table[ply]
                        if move not in k_list:
                            k_list.insert(0, move)
                            if len(k_list) > 2:
                                k_list.pop()
                    if history_table is not None:
                        history_table[1][move.from_square][move.to_square] += depth * depth
                break

        if min_eval <= orig_alpha:
            flag = FLAG_UPPERBOUND
        elif min_eval >= beta:
            flag = FLAG_LOWERBOUND
        else:
            flag = FLAG_EXACT
        TT.store(key, depth, min_eval, flag, best_move, ply=ply)
        return min_eval, best_move, best_leaf


def get_best_move(
    board: chess.Board,
    weights,
    depth: int = 3,
) -> Tuple[Optional[chess.Move], float, chess.Board, int]:
    is_maximizing = (board.turn == chess.WHITE)
    counter = [0]
    killer_table: List[List[chess.Move]] = [[] for _ in range(MAX_SEARCH_PLY)]
    history_table: List[List[List[int]]] = [[[0] * 64 for _ in range(64)] for _ in range(2)]

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
            null_move_allowed=True,
            history_table=history_table,
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
    history_table: List[List[List[int]]] = [[[0] * 64 for _ in range(64)] for _ in range(2)]

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
            null_move_allowed=True,
            history_table=history_table,
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
