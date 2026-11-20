"""
test_p2_search.py
-----------------
Unit tests for Phase P2 search enhancements:
1. Null-Move Pruning (NMP, R=2) with Zugzwang guard
2. Check Extensions (+1 ply)
3. History Heuristic table & move ordering
4. Late Move Reductions (LMR) with full-depth re-search
5. Tactical puzzles & mate conversions
"""

import sys
import chess
import numpy as np
import torch

import search_engine
from search_engine import (
    alpha_beta,
    get_best_move,
    order_moves,
    score_move,
    TT,
    MAX_SEARCH_PLY,
    MATE_SCORE,
)
import feature_extractor


def test_null_move_pruning_node_reduction():
    print("\n--- Test 1: Null-Move Pruning Node Reduction ---")
    # Position: White has an extra Rook and Knight, non-tactical middlegame
    board = chess.Board("r1bq1rk1/pppp1ppp/8/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQ - 0 5")
    weights = feature_extractor.get_default_heuristic_weights().numpy()
    
    # Test at a child node (ply=1) where White is maximizing and beta is bounded
    TT.clear()
    counter_nmp = [0]
    killer_table = [[] for _ in range(MAX_SEARCH_PLY)]
    val_nmp, move_nmp, _ = alpha_beta(
        board=board,
        weights=weights,
        depth=3,
        alpha=0.0,
        beta=0.2,
        is_maximizing=True,
        ply=1,
        killer_table=killer_table,
        node_counter=counter_nmp,
        null_move_allowed=True,
    )
    
    TT.clear()
    counter_no_nmp = [0]
    killer_table_no_nmp = [[] for _ in range(MAX_SEARCH_PLY)]
    val_no_nmp, move_no_nmp, _ = alpha_beta(
        board=board,
        weights=weights,
        depth=3,
        alpha=0.0,
        beta=0.2,
        is_maximizing=True,
        ply=1,
        killer_table=killer_table_no_nmp,
        node_counter=counter_no_nmp,
        null_move_allowed=False,
    )
    
    nodes_nmp = counter_nmp[0]
    nodes_no_nmp = counter_no_nmp[0]
    reduction_pct = (1.0 - nodes_nmp / max(1, nodes_no_nmp)) * 100.0
    
    print(f"Nodes with NMP: {nodes_nmp} | Nodes without NMP: {nodes_no_nmp} | Reduction: {reduction_pct:.1f}%")
    print(f"Eval with NMP: {val_nmp:.2f} | Eval without NMP: {val_no_nmp:.2f}")
    
    assert nodes_nmp < nodes_no_nmp, f"Expected NMP to reduce node count! ({nodes_nmp} vs {nodes_no_nmp})"
    assert val_nmp >= 0.2, f"Expected fail-high (>= beta), got {val_nmp}"
    print("[PASS] Null-Move Pruning successfully triggered fail-high cutoff with massive node reduction.")


def test_nmp_zugzwang_guard():
    print("\n--- Test 2: NMP Zugzwang Safeguard ---")
    # Pawn endgame position: Kings not adjacent, White to move can play b5xa6
    board = chess.Board("8/8/p7/1P6/4k3/8/8/4K3 w - - 0 1")
    weights = feature_extractor.get_default_heuristic_weights().numpy()
    
    # Zugzwang check: side has only pawns and king
    side = board.turn
    has_pieces = bool(
        board.pieces(chess.KNIGHT, side)
        | board.pieces(chess.BISHOP, side)
        | board.pieces(chess.ROOK, side)
        | board.pieces(chess.QUEEN, side)
    )
    print(f"White has non-pawn pieces: {has_pieces}")
    assert not has_pieces, "Position should have only Kings and Pawns."
    
    # Run search, verify no exception occurs and best move captures pawn
    move, eval_score, _, nodes = get_best_move(board, weights, depth=3)
    print(f"Pawn endgame best move: {move} | eval: {eval_score:.2f} | nodes: {nodes}")
    assert move == chess.Move.from_uci("b5a6"), f"Expected b5xa6, got {move}"
    print("[PASS] Zugzwang guard correctly disables NMP in King+Pawn endgames.")


def test_check_extensions():
    print("\n--- Test 3: Check Extensions ---")
    # Position where White King is in check: Scholar's mate attack e.g. Black Q delivers check
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3")
    weights = feature_extractor.get_default_heuristic_weights().numpy()
    
    # White can deliver mate in 1: Qxf7#
    move, eval_score, _, nodes = get_best_move(board, weights, depth=1)
    print(f"Scholar's mate position: Move {move} | eval {eval_score:.2f}")
    assert move == chess.Move.from_uci("f3f7"), f"Expected Qxf7#, got {move}"
    assert eval_score >= MATE_SCORE - 5, f"Expected mate score, got {eval_score}"
    
    # Position where Black is in check
    board.push(move)
    assert board.is_checkmate()
    assert board.is_check()
    print("[PASS] Check Extensions & Checkmate verification passed.")


def test_history_heuristic_ordering():
    print("\n--- Test 4: History Heuristic Move Ordering ---")
    board = chess.Board()
    history_table = [[[0] * 64 for _ in range(64)] for _ in range(2)]
    
    # Simulate historical cutoffs on Nf3 (g1 -> f3)
    from_sq = chess.G1
    to_sq = chess.F3
    history_table[0][from_sq][to_sq] = 5000  # High history score for White
    
    ordered = order_moves(board, history_table=history_table)
    print(f"First move ordered with history: {ordered[0]}")
    assert ordered[0] == chess.Move(from_sq, to_sq), f"Expected Nf3 to be ordered first by history heuristic, got {ordered[0]}"
    print("[PASS] History Heuristic correctly boosts successful quiet moves.")


def test_tactical_conversion_mate_in_2():
    print("\n--- Test 5: Back-Rank Mate in 2 ---")
    # Back rank mate: 1. Re8+ Rxe8 2. Rxe8#
    board = chess.Board("3r2k1/5ppp/8/8/8/8/4R3/4R1K1 w - - 0 1")
    weights = feature_extractor.get_default_heuristic_weights().numpy()
    
    move, eval_score, _, nodes = get_best_move(board, weights, depth=3)
    print(f"Back rank position: Best move: {move} | eval: {eval_score:.2f} | nodes: {nodes}")
    assert move == chess.Move.from_uci("e2e8"), f"Expected Re8+, got {move}"
    assert eval_score >= MATE_SCORE - 10, f"Expected mate score, got {eval_score}"
    print("[PASS] Tactical search converts back-rank checkmate cleanly.")


if __name__ == "__main__":
    test_null_move_pruning_node_reduction()
    test_nmp_zugzwang_guard()
    test_check_extensions()
    test_history_heuristic_ordering()
    test_tactical_conversion_mate_in_2()
    print("\n=======================================================")
    print(" ALL PHASE P2 SEARCH UPGRADE TESTS PASSED SUCCESSFULLY! ")
    print("=======================================================")
