"""
test_p1_p3_features.py
----------------------
Unit tests for Phases P1 & P3:
1. Exact Feature Dimensions (400 base, 800 tapered)
2. Exact Mathematical Antisymmetry on diverse positions
3. Dead Pawn PST Pruning verification
4. Selective L2 Regularization Mask
5. Orthogonal P3 Features verification (Safe mobility, Endgame mop-up, King zone, Outposts)
"""

import sys
import numpy as np
import torch
import chess

import feature_extractor


def test_dimensions_and_names():
    print("\n--- Test 1: Dimensions and Names ---")
    base_names = feature_extractor.get_base_feature_names()
    tapered_names = feature_extractor.get_feature_names()
    
    print(f"Base Feature Count: {len(base_names)}")
    print(f"Tapered Feature Count: {len(tapered_names)}")
    
    assert len(base_names) == 400, f"Expected 400 base features, got {len(base_names)}"
    assert len(tapered_names) == 800, f"Expected 800 tapered features, got {len(tapered_names)}"
    assert feature_extractor.BASE_FEATURE_DIM == 400
    assert feature_extractor.TOTAL_FEATURE_DIM == 800
    print("[PASS] Dimensions and naming schemas are 100% consistent.")


def test_exact_antisymmetry():
    print("\n--- Test 2: Exact Mathematical Antisymmetry ---")
    test_fens = [
        # Standard initial position
        chess.STARTING_FEN,
        # Italian game
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5",
        # Asymmetric pawn structure (French Defense)
        "rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq c6 0 4",
        # Technical endgame: King + Rook + Pawn vs King + Rook
        "8/5k2/8/8/4P3/8/5K2/4R1r1 w - - 0 1",
        # Endgame: Opposite colored bishops with passed pawns
        "8/2b2k2/8/4P3/8/5K2/2B5/8 b - - 0 1",
        # Mop-up endgame: White Queen vs lone Black King
        "8/8/8/4k3/8/8/8/4K1Q1 w - - 0 1",
        # Black ahead in material
        "r1bqkbnr/pppp1ppp/8/8/4n3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    ]
    
    for i, fen in enumerate(test_fens):
        board = chess.Board(fen)
        mirrored = board.mirror()
        
        feat = feature_extractor.extract_features_numpy(board)
        feat_mirrored = feature_extractor.extract_features_numpy(mirrored)
        
        diff = np.max(np.abs(feat + feat_mirrored))
        print(f"FEN {i+1} Max Antisymmetry Violation: {diff:.8f}")
        assert np.allclose(feat, -feat_mirrored, atol=1e-5), f"Antisymmetry violation on FEN {i+1}: max diff = {diff}"
        
    print("[PASS] 100% exact mathematical antisymmetry holds on all test positions.")


def test_dead_pawn_pst_pruned():
    print("\n--- Test 3: Dead Pawn PST Pruning ---")
    base_names = feature_extractor.get_base_feature_names()
    pawn_pst_names = [n for n in base_names if n.startswith("pst_pawn_")]
    
    print(f"Total Pawn PST Features: {len(pawn_pst_names)} (expected 48)")
    assert len(pawn_pst_names) == 48, f"Expected 48 pawn PST squares, got {len(pawn_pst_names)}"
    
    # Verify no rank 1 or rank 8 pawn features exist
    for r in [1, 8]:
        for f in "abcdefgh":
            illegal_sq = f"{f}{r}"
            assert f"pst_pawn_{illegal_sq}" not in pawn_pst_names, f"Dead feature pst_pawn_{illegal_sq} found!"
            
    print("[PASS] Dead Pawn PST features (Ranks 1 and 8) completely pruned.")


def test_selective_l2_regularization_mask():
    print("\n--- Test 4: Selective L2 Regularization Mask ---")
    mask = feature_extractor.get_regularization_mask()
    
    assert mask.shape == (800,), f"Expected shape (800,), got {mask.shape}"
    # Material features unpenalized: 0..4 in MG, 400..404 in EG
    assert np.all(mask[0:5] == 0.0), "MG material weights must be unpenalized (mask=0.0)"
    assert np.all(mask[400:405] == 0.0), "EG material weights must be unpenalized (mask=0.0)"
    
    # Positional/PST features regularized
    assert np.all(mask[5:400] == 1.0), "MG positional weights must be regularized (mask=1.0)"
    assert np.all(mask[405:800] == 1.0), "EG positional weights must be regularized (mask=1.0)"
    
    unpenalized_cnt = int(np.sum(mask == 0.0))
    regularized_cnt = int(np.sum(mask == 1.0))
    print(f"Unpenalized Material Weights: {unpenalized_cnt} | Regularized Positional Weights: {regularized_cnt}")
    assert unpenalized_cnt == 10
    assert regularized_cnt == 790
    print("[PASS] Selective L2 Regularization (Hierarchical Shrinkage) mask verified.")


def test_orthogonal_p3_features():
    print("\n--- Test 5: Orthogonal P3 Features ---")
    # Position with White Knight outpost on d5 defended by e4 pawn,
    # where Black has no pawns on adjacent c and e files (true immune outpost!)
    board = chess.Board("r1bqkb1r/pp3ppp/2n5/3N4/4P3/8/PPPP1PPP/R1BQK2R w KQkq - 0 6")
    feat = feature_extractor.extract_base_features_numpy(board)
    
    # 395: knight_outpost_diff
    outpost_val = feat[395]
    print(f"Knight Outpost Feature (idx 395): {outpost_val}")
    assert outpost_val >= 1.0, f"Expected outpost >= 1.0, got {outpost_val}"
    
    # Position with winning endgame: lone King on a8 vs White Queen
    board_mop = chess.Board("k7/8/1K6/8/8/8/8/7Q w - - 0 1")
    feat_mop = feature_extractor.extract_base_features_numpy(board_mop)
    # 391: endgame_losing_king_to_edge
    edge_val = feat_mop[391]
    print(f"Losing King to Edge Feature (idx 391): {edge_val:.3f}")
    assert edge_val > 0.8, f"Expected high edge penalty for King on a8, got {edge_val}"
    
    # 392: endgame_king_proximity
    prox_val = feat_mop[392]
    print(f"King Proximity Feature (idx 392): {prox_val:.3f}")
    assert prox_val > 0.5, f"Expected high proximity for Kings close to each other, got {prox_val}"
    
    print("[PASS] P3 Orthogonal features accurately detect tactical and endgame structures.")


if __name__ == "__main__":
    test_dimensions_and_names()
    test_exact_antisymmetry()
    test_dead_pawn_pst_pruned()
    test_selective_l2_regularization_mask()
    test_orthogonal_p3_features()
    print("\n===========================================================")
    print(" ALL PHASE P1 & P3 FEATURE EXTRACTOR TESTS PASSED 100%! ")
    print("===========================================================")
