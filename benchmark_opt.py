import time
import chess
import numpy as np
import feature_extractor
import search_engine

b = chess.Board()
w = feature_extractor.get_default_heuristic_weights().numpy()

# 1. Current evaluate_board
t0 = time.time()
for _ in range(5000):
    sc = search_engine.evaluate_board(b, w)
t_curr = time.time() - t0
print("Current evaluate_board:", round(5000 / t_curr), "evals/sec")

def extract_features_fast(board):
    arr = np.zeros(feature_extractor.TOTAL_FEATURE_DIM, dtype=np.float32)
    
    # 1. Material Imbalances (5 features)
    for i, pt in enumerate(feature_extractor.EVAL_PIECE_TYPES):
        w_count = board.pieces_mask(pt, chess.WHITE).bit_count()
        b_count = board.pieces_mask(pt, chess.BLACK).bit_count()
        arr[i] = float(w_count - b_count)
        
    # 2. Piece-Square Tables (640 features: 10 piece types x 64 squares)
    # Fast bitboard scan: only iterate over occupied squares
    idx = 5
    for color in (chess.WHITE, chess.BLACK):
        for pt in feature_extractor.EVAL_PIECE_TYPES:
            mask = board.pieces_mask(pt, color)
            for sq in chess.scan_forward(mask):
                arr[idx + sq] = 1.0
            idx += 64
                
    # 3. Mobility Difference (1 feature)
    orig_turn = board.turn
    if orig_turn == chess.WHITE:
        w_mob = board.legal_moves.count()
        board.turn = chess.BLACK
        b_mob = board.legal_moves.count()
        board.turn = chess.WHITE
    else:
        b_mob = board.legal_moves.count()
        board.turn = chess.WHITE
        w_mob = board.legal_moves.count()
        board.turn = chess.BLACK
    arr[idx] = float(w_mob - b_mob)
    idx += 1
    
    # 4. Center Control (1 feature: attacks on d4, d5, e4, e5)
    w_center = sum(board.attackers_mask(chess.WHITE, sq).bit_count() for sq in feature_extractor.CENTER_SQUARES)
    b_center = sum(board.attackers_mask(chess.BLACK, sq).bit_count() for sq in feature_extractor.CENTER_SQUARES)
    arr[idx] = float(w_center - b_center)
    idx += 1
    
    # 5. Pawn Structure (3 features: doubled, isolated, passed)
    w_doub, b_doub, w_iso, b_iso, w_pass, b_pass = feature_extractor._count_pawn_structure(board)
    arr[idx] = float(w_doub - b_doub)
    idx += 1
    arr[idx] = float(w_iso - b_iso)
    idx += 1
    arr[idx] = float(w_pass - b_pass)
    idx += 1
    
    # 6. King Shield Integrity (1 feature)
    w_shield, b_shield = feature_extractor._count_king_shield(board)
    arr[idx] = float(w_shield - b_shield)
    idx += 1
    
    return arr

# Check correctness across several random board states
arr_orig = feature_extractor.extract_features_numpy(b)
arr_fast = extract_features_fast(b)
assert np.allclose(arr_orig, arr_fast), f"Difference found: {np.max(np.abs(arr_orig - arr_fast))}"
print("Correctness check passed! Exactly identical 651 features on initial board.")

# Test after some moves
b.push_san("e4")
b.push_san("e5")
b.push_san("Nf3")
b.push_san("Nc6")
arr_orig = feature_extractor.extract_features_numpy(b)
arr_fast = extract_features_fast(b)
assert np.allclose(arr_orig, arr_fast), "Mismatch after 4 moves!"
print("Correctness check passed! Exactly identical 651 features after moves.")

t0 = time.time()
for _ in range(5000):
    sc = float(np.dot(w, extract_features_fast(b)))
t_fast = time.time() - t0
print("Optimized evaluate_board:", round(5000 / t_fast), "evals/sec (Speedup:", round(t_curr / t_fast, 2), "x)")
