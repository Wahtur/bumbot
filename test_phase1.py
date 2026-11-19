"""
test_phase1.py
--------------
Unit tests for Phase 1 Feature Extractor:
1. Exact color antisymmetry across 50 random positions:
   extract(board) == -extract(board.mirror())
2. Throughput benchmark (evaluations / second).
"""

import time
import random
import numpy as np
import chess

# Let's import the new feature extractor
import feature_extractor_v2 as fe

def run_tests():
    print("=" * 60)
    print(" PHASE 1 UNIT TESTS: FEATURE EXTRACTOR V2")
    print("=" * 60)

    print(f"Base Feature Dim    : {fe.BASE_FEATURE_DIM}")
    print(f"Tapered Feature Dim : {fe.TOTAL_FEATURE_DIM}")
    assert fe.TOTAL_FEATURE_DIM == 808, f"Expected 808, got {fe.TOTAL_FEATURE_DIM}"

    # 1. Antisymmetry Test on Starting Position
    b = chess.Board()
    v_start = fe.extract_features_numpy(b)
    v_mirror = fe.extract_features_numpy(b.mirror())
    diff = np.abs(v_start + v_mirror).max()
    print(f"Starting position max antisymmetry error: {diff:.8f}")
    assert diff < 1e-6, f"Starting position failed antisymmetry: diff={diff}"
    print("[PASS] Starting position is perfectly antisymmetric!")

    # 2. Antisymmetry Test on 50 Random Valid Game Positions
    random.seed(42)
    max_error = 0.0
    for i in range(50):
        board = chess.Board()
        # Play 10 to 60 random moves
        num_moves = random.randint(10, 60)
        for _ in range(num_moves):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))

        f_normal = fe.extract_features_numpy(board)
        f_mirror = fe.extract_features_numpy(board.mirror())

        err = np.abs(f_normal + f_mirror).max()
        if err > max_error:
            max_error = err
        assert err < 1e-5, f"Position {i} failed antisymmetry: max error = {err}"

    print(f"[PASS] 50/50 Random positions passed antisymmetry! Max diff across all: {max_error:.8e}")

    # 3. Throughput Benchmark
    print("-" * 60)
    print("Benchmarking throughput...")
    test_boards = []
    board = chess.Board()
    for _ in range(100):
        if board.is_game_over():
            board = chess.Board()
        moves = list(board.legal_moves)
        if moves:
            board.push(random.choice(moves))
            test_boards.append(board.copy())

    N = len(test_boards)
    # Warmup
    for tb in test_boards[:10]:
        fe.extract_features_numpy(tb)

    t0 = time.time()
    iterations = 200
    for _ in range(iterations):
        for tb in test_boards:
            fe.extract_features_numpy(tb)
    elapsed = time.time() - t0
    total_evals = N * iterations
    throughput = total_evals / elapsed

    print(f"Evaluated {total_evals:,} positions in {elapsed:.2f}s")
    print(f"Throughput: {throughput:,.0f} leaf evaluations/second")
    print("-" * 60)
    print("ALL PHASE 1 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
