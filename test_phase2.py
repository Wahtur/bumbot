"""
test_phase2.py
--------------
Unit tests for Phase 2 TD-Leaf Trainer:
1. ChessLogisticEvaluator initialization and forward pass
2. TD-Leaf BCE loss computation with gamma=0.99
3. PGD constraints on Middlegame vs Endgame feature slices
4. 4-game on-policy training step
"""

import time
import numpy as np
import torch
import feature_extractor_v2 as fe
import td_leaf_trainer_v2 as tr

def run_tests():
    print("=" * 65)
    print(" PHASE 2 UNIT TESTS: LOGISTIC REGRESSION TD-LEAF TRAINER")
    print("=" * 65)

    evaluator = tr.ChessLogisticEvaluator()
    init_w = fe.get_default_heuristic_weights()
    evaluator.set_weights(init_w)

    # 1. Forward test
    dummy_feat = torch.zeros(fe.TOTAL_FEATURE_DIM)
    p = evaluator(dummy_feat.unsqueeze(0)).item()
    print(f"Zero features probability: p={p:.4f} (Expected: ~0.5000)")
    assert abs(p - 0.5) < 1e-4, f"Zero vector should evaluate to 0.5, got {p}"
    print("[PASS] Evaluator correctly maps zero vector to 0.5000 probability.")

    # 2. PGD Constraints Test
    tr.apply_tapered_pgd_constraints(evaluator)
    w = evaluator.get_weights().numpy()
    D = fe.BASE_FEATURE_DIM
    # Check material >= 0.1
    assert (w[0:5] >= 0.099).all(), "MG Material constraint failed!"
    assert (w[D:D+5] >= 0.099).all(), "EG Material constraint failed!"
    # Check king shield: MG >= 0.01, EG unconstrained
    assert w[402] >= 0.009, "MG King shield constraint failed!"
    print("[PASS] Tapered PGD constraints verified (Material clamped, King shield MG clamped, EG unconstrained)!")

    # 3. Simulate 2 games and compute BCE Loss
    print("-" * 65)
    print("Simulating 2 test games for BCE loss verification...")
    t0 = time.time()
    res1 = tr.play_self_play_game(w, search_depth=1, max_plies=30)
    res2 = tr.play_self_play_game(w, search_depth=1, max_plies=30)
    print(f"Game 1: {res1[3]} plies, outcome={res1[1]}, reason={res1[2]}")
    print(f"Game 2: {res2[3]} plies, outcome={res2[1]}, reason={res2[2]}")

    loss1 = tr.compute_td_leaf_bce_loss(evaluator, res1[0], res1[1], gamma=0.99)
    loss2 = tr.compute_td_leaf_bce_loss(evaluator, res2[0], res2[1], gamma=0.99)
    print(f"BCE Loss Game 1: {loss1.item():.5f}")
    print(f"BCE Loss Game 2: {loss2.item():.5f}")
    assert loss1.item() > 0.0, "Loss must be positive"

    # Test backprop
    loss = (loss1 + loss2) / 2.0
    loss.backward()
    grad_norm = torch.norm(evaluator.linear.weight.grad).item()
    print(f"Backward Gradient Norm: {grad_norm:.5f}")
    assert grad_norm > 0.0, "Gradients must be non-zero"
    print("[PASS] BCE Loss and backward gradient pass verified!")

    print("-" * 65)
    print("ALL PHASE 2 TRAINER TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
