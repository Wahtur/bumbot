"""
test_phase3.py
--------------
Unit tests and benchmark for Phase 3 Search Engine:
1. Valid moves generated across distinct test positions.
2. Checkmate tactical detection and mate score handling.
3. Speedup & throughput comparison with Transposition Table + Iterative Deepening.
"""

import time
import chess
import numpy as np
import feature_extractor_v2 as fe
import search_engine_v2 as se

def run_tests():
    print("=" * 65)
    print(" PHASE 3 UNIT TESTS: SEARCH ENGINE V2 (TT + ITERATIVE DEEPENING)")
    print("=" * 65)

    weights = fe.get_default_heuristic_weights().numpy()

    # 1. Test Starting Position
    b = chess.Board()
    t0 = time.time()
    move, score, leaf, nodes = se.get_best_move(b, weights, depth=3)
    elapsed = time.time() - t0
    print(f"Starting Pos (D3): Move={move}, Score={score:.3f}, Nodes={nodes}, Time={elapsed:.3f}s ({nodes/elapsed:,.0f} nodes/s)")
    assert move is not None and move in b.legal_moves, f"Illegal move returned: {move}"
    print("[PASS] Starting position move is legal and evaluated.")

    # 2. Tactical Mate-in-1 Detection Test (Scholar's Mate threat)
    # 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
    b_mate = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    move, score, leaf, nodes = se.get_best_move(b_mate, weights, depth=2)
    print(f"Mate-in-1 Test   : Move={move}, Score={score}, Nodes={nodes}")
    assert move == chess.Move.from_uci("h5f7"), f"Failed to find Qxf7# mate! Found: {move}"
    assert score >= 9000.0, f"Expected checkmate score >= 9000, got {score}"
    print("[PASS] Mate-in-1 (Qxf7#) detected correctly with high mate score!")

    # 3. Transposition Table Hit Rate Verification
    b_mid = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
    se.TT.clear()
    t0 = time.time()
    m1, s1, l1, n1 = se.get_best_move(b_mid, weights, depth=3)
    e1 = time.time() - t0

    # Search again (TT should provide immediate hits)
    t0 = time.time()
    m2, s2, l2, n2 = se.get_best_move(b_mid, weights, depth=3)
    e2 = time.time() - t0

    print(f"Middlegame Search 1 (Cold TT): {n1} nodes, {e1:.3f}s ({n1/e1:,.0f} n/s)")
    print(f"Middlegame Search 2 (Warm TT): {n2} nodes, {e2:.3f}s ({n2/e2:,.0f} n/s)")
    assert m1 == m2, f"Move mismatch between TT searches: {m1} vs {m2}"
    print(f"[PASS] TT search consistency verified (Move {m1})!")

    # 4. PV Leaf State Verification
    assert isinstance(leaf, chess.Board), "Leaf is not a chess.Board!"
    feat = fe.extract_features_numpy(leaf)
    assert len(feat) == 808, f"Expected 808 leaf features, got {len(feat)}"
    print("[PASS] Principal Variation leaf board returns valid 808-feature tensor!")

    print("-" * 65)
    print("ALL PHASE 3 SEARCH TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
