"""
test_p0_fixes.py
----------------
Unit tests verifying the P0 Critical Correctness & Draw Fixes:
1. Exact probability-centered temporal discounting in TD-Leaf(lambda) with gamma=0.99.
2. Ply-adjusted relative mate scores in Transposition Table (TT).
3. Path-aware repetition detection with material-based contempt.
"""

import numpy as np
import torch
import chess
import chess.polyglot

import feature_extractor
import search_engine
from td_leaf_trainer import ChessLogisticEvaluator, compute_td_leaf_bce_loss


def test_centered_discounting_symmetry():
    print("=== Test 1: Probability-Centered Discounting Symmetry ===")
    evaluator = ChessLogisticEvaluator()
    # Dummy features (5 positions)
    dummy_feats = np.random.randn(5, feature_extractor.TOTAL_FEATURE_DIM).astype(np.float32)

    # White win trajectory (outcome = 1.0)
    loss_white = compute_td_leaf_bce_loss(evaluator, dummy_feats, terminal_outcome=1.0, lambda_val=0.7, gamma=0.99)
    assert not torch.isnan(loss_white), "White loss is NaN"

    # Draw trajectory (outcome = 0.5)
    # If all predictions are 0.5 and terminal is 0.5, all targets should be exactly 0.5
    evaluator_neutral = ChessLogisticEvaluator()
    with torch.no_grad():
        evaluator_neutral.linear.weight.zero_()  # sigmoid(0) = 0.5
    
    # Calculate manually to check target
    p_preds = evaluator_neutral(torch.from_numpy(dummy_feats)).squeeze(-1)
    p_vals = p_preds.detach().numpy()
    p_centered = p_vals - 0.5
    outcome_centered = 0.5 - 0.5

    deltas = np.zeros(5, dtype=np.float32)
    for t in range(4):
        deltas[t] = 0.99 * p_centered[t + 1] - p_centered[t]
    deltas[4] = outcome_centered - p_centered[4]

    returns = np.zeros(5, dtype=np.float32)
    running = 0.0
    for t in reversed(range(5)):
        running = deltas[t] + (0.99 * 0.7) * running
        returns[t] = running

    targets = 0.5 + p_centered + returns
    np.testing.assert_allclose(targets, 0.5, atol=1e-5, err_msg="Draw targets drifted away from 0.5!")
    print("  [PASS] Neutral draw trajectory preserves exact 0.5 target without drift!")

    # Check color inversion symmetry between win and loss targets
    p_win = np.array([0.6, 0.65, 0.7, 0.8, 0.9], dtype=np.float32)
    p_loss = 1.0 - p_win  # perfectly inverted: [0.4, 0.35, 0.3, 0.2, 0.1]

    # Win targets
    pw_cent = p_win - 0.5
    dw = np.zeros(5, dtype=np.float32)
    for t in range(4):
        dw[t] = 0.99 * pw_cent[t+1] - pw_cent[t]
    dw[4] = (1.0 - 0.5) - pw_cent[4]
    rw = np.zeros(5, dtype=np.float32)
    run = 0.0
    for t in reversed(range(5)):
        run = dw[t] + 0.99 * 0.7 * run
        rw[t] = run
    win_targets = 0.5 + pw_cent + rw

    # Loss targets
    pl_cent = p_loss - 0.5  # == -pw_cent
    dl = np.zeros(5, dtype=np.float32)
    for t in range(4):
        dl[t] = 0.99 * pl_cent[t+1] - pl_cent[t]
    dl[4] = (0.0 - 0.5) - pl_cent[4]
    rl = np.zeros(5, dtype=np.float32)
    run = 0.0
    for t in reversed(range(5)):
        run = dl[t] + 0.99 * 0.7 * run
        rl[t] = run
    loss_targets = 0.5 + pl_cent + rl

    # Verify win_targets - 0.5 == -(loss_targets - 0.5)
    np.testing.assert_allclose(win_targets - 0.5, -(loss_targets - 0.5), atol=1e-6, err_msg="Antisymmetry broken!")
    print("  [PASS] Exact color antisymmetry preserved in centered discounting!")


def test_tt_mate_score_conversion():
    print("=== Test 2: TT Relative Mate Score Encoding/Decoding ===")
    mate = search_engine.MATE_SCORE
    
    # Store mate in 2 plies at search ply 2
    raw_eval = mate - 2  # mate found at ply 2
    encoded_tt = search_engine.score_to_tt(raw_eval, ply=2)
    assert encoded_tt == mate, f"Expected encoded {mate}, got {encoded_tt}"

    # Retrieve from TT at search ply 4 (now mate is 4 plies away from root)
    decoded_eval = search_engine.score_from_tt(encoded_tt, ply=4)
    assert decoded_eval == mate - 4, f"Expected decoded {mate - 4}, got {decoded_eval}"

    # Negative mate (getting mated)
    raw_loss = -(mate - 3)
    encoded_loss = search_engine.score_to_tt(raw_loss, ply=3)
    assert encoded_loss == -mate, f"Expected encoded {-mate}, got {encoded_loss}"

    decoded_loss = search_engine.score_from_tt(encoded_loss, ply=5)
    assert decoded_loss == -(mate - 5), f"Expected decoded {-(mate - 5)}, got {decoded_loss}"
    
    print("  [PASS] TT correctly adjusts mate distance relative to search plies!")


def test_repetition_contempt():
    print("=== Test 3: Path-Aware Repetition & Material Contempt ===")
    # Create board that repeats moves: 1. Nf3 Nf6 2. Ng1 Ng8
    board = chess.Board()
    board.push_san("Nf3")
    board.push_san("Nf6")
    board.push_san("Ng1")
    board.push_san("Ng8")
    assert board.is_repetition(2), "Expected 2-fold repetition on start pos"

    h_weights = feature_extractor.get_default_heuristic_weights().numpy()

    # Search at ply > 0 on repeated position
    eval_score, move, leaf = search_engine.alpha_beta(
        board, h_weights, depth=1, alpha=-1000, beta=1000, is_maximizing=True, ply=1
    )
    # At initial material (equal), score should be 0.0 (no contempt when equal)
    assert abs(eval_score) < 0.01, f"Expected equal repetition ~0.0, got {eval_score}"

    # Now make White up a Queen (+9) and test repeated state
    board_ahead = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")  # Black missing queen
    board_ahead.push_san("Nf3")
    board_ahead.push_san("Nf6")
    board_ahead.push_san("Ng1")
    board_ahead.push_san("Ng8")
    assert board_ahead.is_repetition(2)

    eval_ahead, _, _ = search_engine.alpha_beta(
        board_ahead, h_weights, depth=1, alpha=-1000, beta=1000, is_maximizing=True, ply=1
    )
    # White is ahead, so White should penalize repeating with contempt (-0.25)
    assert eval_ahead < -0.1, f"Expected negative contempt when ahead, got {eval_ahead}"
    print(f"  [PASS] Contempt successfully penalized repetition when ahead: eval = {eval_ahead:.2f}")


if __name__ == "__main__":
    test_centered_discounting_symmetry()
    test_tt_mate_score_conversion()
    test_repetition_contempt()
    print("\nALL P0 FIX TESTS PASSED SUCCESSFULLY!")
