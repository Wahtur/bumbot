"""
feature_extractor.py
--------------------
Owner: Partner A (Search & Features)

Extracts handcrafted classical machine learning features from a chess.Board state
into a 1D PyTorch tensor for linear value function approximation in TD-Leaf(lambda).

Features (Total dimension: 652):
  - Material Imbalance (5): Net difference (White - Black) for Pawns, Knights, Bishops, Rooks, Queens.
  - Piece-Square Tables (640): Binary indicators (10 piece types x 64 squares) for board presence.
  - Mobility (1): Difference in legal moves (White - Black).
  - Center Control (1): Difference in attacks on d4, d5, e4, e5 (White - Black).
  - Pawn Structure (3):
      * Doubled pawns difference (White - Black)
      * Isolated pawns difference (White - Black)
      * Passed pawns difference (White - Black)
  - King Shield Integrity (1): Difference in pawn shield count around the king (White - Black).
"""

from typing import List, Tuple
import numpy as np
import torch
import chess

# Constants for piece types (excluding king from PST as per 10-piece 640-feature spec)
EVAL_PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
]

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
}

CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
TOTAL_FEATURE_DIM = 651


def get_feature_names() -> List[str]:
    """Return descriptive names for all features in the vector for interpretability."""
    names: List[str] = []
    
    # 1. Material
    for pt in EVAL_PIECE_TYPES:
        names.append(f"mat_{PIECE_NAMES[pt]}_diff")
        
    # 2. Piece-Square Tables (White then Black)
    for color, c_name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        for pt in EVAL_PIECE_TYPES:
            pt_name = PIECE_NAMES[pt]
            for sq in range(64):
                sq_name = chess.square_name(sq)
                names.append(f"pst_{c_name}_{pt_name}_{sq_name}")
                
    # 3. Mobility
    names.append("mobility_diff")
    
    # 4. Center Control
    names.append("center_control_diff")
    
    # 5. Pawn Structure
    names.append("doubled_pawns_diff")
    names.append("isolated_pawns_diff")
    names.append("passed_pawns_diff")
    
    # 6. King Shield
    names.append("king_shield_diff")
    
    assert len(names) == TOTAL_FEATURE_DIM, f"Expected {TOTAL_FEATURE_DIM} names, got {len(names)}"
    return names


def _count_pawn_structure(board: chess.Board) -> Tuple[int, int, int, int, int, int]:
    """
    Returns (w_doubled, b_doubled, w_isolated, b_isolated, w_passed, b_passed).
    Utilizes bitwise operations for high speed.
    """
    w_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
    
    # Pre-count pawns per file
    w_file_counts = [
        (w_pawns & chess.BB_FILES[f]).bit_count() for f in range(8)
    ]
    b_file_counts = [
        (b_pawns & chess.BB_FILES[f]).bit_count() for f in range(8)
    ]
    
    w_doubled = sum(max(0, c - 1) for c in w_file_counts)
    b_doubled = sum(max(0, c - 1) for c in b_file_counts)
    
    # Isolated pawns: no friendly pawns on neighboring files
    w_isolated = 0
    b_isolated = 0
    for f in range(8):
        left_has_w = (f > 0 and w_file_counts[f - 1] > 0)
        right_has_w = (f < 7 and w_file_counts[f + 1] > 0)
        if w_file_counts[f] > 0 and not left_has_w and not right_has_w:
            w_isolated += w_file_counts[f]
            
        left_has_b = (f > 0 and b_file_counts[f - 1] > 0)
        right_has_b = (f < 7 and b_file_counts[f + 1] > 0)
        if b_file_counts[f] > 0 and not left_has_b and not right_has_b:
            b_isolated += b_file_counts[f]
            
    # Passed pawns: no enemy pawns ahead on same or adjacent files
    w_passed = 0
    b_passed = 0
    
    for sq in chess.scan_forward(w_pawns):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        # Front files mask (files max(0, f-1) to min(7, f+1)) and ranks > r
        mask = 0
        for adj_f in range(max(0, f - 1), min(8, f + 2)):
            mask |= chess.BB_FILES[adj_f]
        ahead_ranks = 0
        for rank_idx in range(r + 1, 8):
            ahead_ranks |= chess.BB_RANKS[rank_idx]
        if not (b_pawns & mask & ahead_ranks):
            w_passed += 1
            
    for sq in chess.scan_forward(b_pawns):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        mask = 0
        for adj_f in range(max(0, f - 1), min(8, f + 2)):
            mask |= chess.BB_FILES[adj_f]
        ahead_ranks = 0
        for rank_idx in range(0, r):
            ahead_ranks |= chess.BB_RANKS[rank_idx]
        if not (w_pawns & mask & ahead_ranks):
            b_passed += 1
            
    return w_doubled, b_doubled, w_isolated, b_isolated, w_passed, b_passed


def _count_king_shield(board: chess.Board) -> Tuple[int, int]:
    """
    Returns (w_shield, b_shield) counting friendly pawns shielding the king.
    """
    w_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
    
    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)
    
    w_shield = 0
    if w_king_sq is not None:
        kf = chess.square_file(w_king_sq)
        kr = chess.square_rank(w_king_sq)
        # Only evaluate shield if king is in ranks 0-2 (uncastled or castled early/midgame)
        if kr <= 2:
            shield_files = range(max(0, kf - 1), min(8, kf + 2))
            shield_ranks = range(kr + 1, min(8, kr + 3))
            for f in shield_files:
                for r in shield_ranks:
                    if w_pawns & (1 << chess.square(f, r)):
                        w_shield += 1
                        
    b_shield = 0
    if b_king_sq is not None:
        kf = chess.square_file(b_king_sq)
        kr = chess.square_rank(b_king_sq)
        if kr >= 5:
            shield_files = range(max(0, kf - 1), min(8, kf + 2))
            shield_ranks = range(max(0, kr - 2), kr)
            for f in shield_files:
                for r in shield_ranks:
                    if b_pawns & (1 << chess.square(f, r)):
                        b_shield += 1
                        
    return w_shield, b_shield


def extract_features_numpy(board: chess.Board) -> np.ndarray:
    """
    Extracts all 651 engineered features from a chess.Board into a 1D NumPy float32 array.
    Evaluation is oriented from White's perspective (+White, -Black).
    """
    arr = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    
    # 1. Material Imbalances (5 features)
    for i, pt in enumerate(EVAL_PIECE_TYPES):
        w_count = board.pieces_mask(pt, chess.WHITE).bit_count()
        b_count = board.pieces_mask(pt, chess.BLACK).bit_count()
        arr[i] = float(w_count - b_count)
        
    # 2. Piece-Square Tables (640 features: 10 piece types x 64 squares)
    idx = 5
    for color in (chess.WHITE, chess.BLACK):
        for pt in EVAL_PIECE_TYPES:
            mask = board.pieces_mask(pt, color)
            for sq in range(64):
                if mask & (1 << sq):
                    arr[idx] = 1.0
                idx += 1
                
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
    w_center = sum(board.attackers_mask(chess.WHITE, sq).bit_count() for sq in CENTER_SQUARES)
    b_center = sum(board.attackers_mask(chess.BLACK, sq).bit_count() for sq in CENTER_SQUARES)
    arr[idx] = float(w_center - b_center)
    idx += 1
    
    # 5. Pawn Structure (3 features: doubled, isolated, passed)
    w_doub, b_doub, w_iso, b_iso, w_pass, b_pass = _count_pawn_structure(board)
    arr[idx] = float(w_doub - b_doub)
    idx += 1
    arr[idx] = float(w_iso - b_iso)
    idx += 1
    arr[idx] = float(w_pass - b_pass)
    idx += 1
    
    # 6. King Shield Integrity (1 feature)
    w_shield, b_shield = _count_king_shield(board)
    arr[idx] = float(w_shield - b_shield)
    idx += 1
    
    assert idx == TOTAL_FEATURE_DIM, f"Feature index mismatch: {idx} != {TOTAL_FEATURE_DIM}"
    return arr


def extract_features(board: chess.Board) -> torch.Tensor:
    """
    Extracts all 651 engineered features into a 1D PyTorch float32 tensor.
    """
    return torch.from_numpy(extract_features_numpy(board))


def get_default_heuristic_weights() -> torch.Tensor:
    """
    Constructs a baseline heuristic weight vector matching standard chess principles:
      - Pawns: 100, Knights: 300, Bishops: 320, Rooks: 500, Queens: 900
      - Center control: +10
      - Mobility: +2
      - Doubled pawns penalty: -15
      - Isolated pawns penalty: -20
      - Passed pawns bonus: +30
      - King shield bonus: +15
    Used as an initial baseline bot and for comparison in learning curves.
    """
    weights = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    
    # Material weights
    weights[0] = 1.00   # Pawn
    weights[1] = 3.00   # Knight
    weights[2] = 3.25   # Bishop
    weights[3] = 5.00   # Rook
    weights[4] = 9.00   # Queen
    
    # Heuristic PST weights (scaled in centipawns / 100)
    # Simple central bias for knights, bishops, pawns
    pst_offset_white = 5
    pst_offset_black = 5 + 320
    
    # Knights on White (index 1 of 5)
    for sq in range(64):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        dist_from_center = abs(3.5 - f) + abs(3.5 - r)
        center_bias = max(0.0, (4.0 - dist_from_center) * 0.05)
        # White knights (idx 5 + 64)
        weights[pst_offset_white + 64 + sq] = center_bias
        # Black knights (negative because it favors Black)
        weights[pst_offset_black + 64 + sq] = -center_bias
        
    # High-level strategic features
    idx = 5 + 640
    weights[idx] = 0.02      # Mobility diff
    idx += 1
    weights[idx] = 0.08      # Center control diff
    idx += 1
    weights[idx] = -0.15     # Doubled pawns (penalty for White, so negative)
    idx += 1
    weights[idx] = -0.20     # Isolated pawns (penalty for White)
    idx += 1
    weights[idx] = 0.30      # Passed pawns (bonus for White)
    idx += 1
    weights[idx] = 0.15      # King shield (bonus for White)
    
    return torch.tensor(weights, dtype=torch.float32)
