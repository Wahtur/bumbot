"""
feature_extractor_v2.py
-----------------------
Upgraded Classical Feature Extractor for Bumbot (Phase 1):
  - Antisymmetric Piece-Square Tables (PST) including Kings (6 piece types x 64 squares = 384 base features)
  - Material Imbalance (5 base features)
  - Passed Pawn Rank Buckets (6 base features: ranks 2 through 7)
  - Bishop Pair (1 base feature)
  - Rook on Open / Semi-Open File (2 base features)
  - Tempo (1 base feature)
  - Doubled Pawns (1 base feature)
  - Isolated Pawns (1 base feature)
  - Center Control (1 base feature)
  - King Shield Integrity (1 base feature)
  - Fast Bitboard Pseudo-Mobility (1 base feature)
  
Base Feature Dimension: 404
Tapered Game Phase Feature Dimension: 2 x 404 = 808
  concat(rho * x_base, (1.0 - rho) * x_base)

Guarantees 100% exact mathematical antisymmetry:
  extract(board) == -extract(board.mirror())
"""

from typing import List, Tuple
import numpy as np
import torch
import chess

# 6 piece types for PST: Pawn, Knight, Bishop, Rook, Queen, King
ALL_PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

# 5 material piece types (excluding King)
MATERIAL_PIECE_TYPES = [
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
    chess.KING: "king",
}

# Precompute bitboard spans for fast passed pawn detection
WHITE_PASSED_SPAN = [0] * 64
BLACK_PASSED_SPAN = [0] * 64

for sq in range(64):
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    
    w_mask = 0
    b_mask = 0
    for fi in range(max(0, f - 1), min(8, f + 2)):
        for ri in range(r + 1, 8):
            w_mask |= (1 << chess.square(fi, ri))
        for ri in range(0, r):
            b_mask |= (1 << chess.square(fi, ri))
            
    WHITE_PASSED_SPAN[sq] = w_mask
    BLACK_PASSED_SPAN[sq] = b_mask

# Center squares: d4 (27), e4 (28), d5 (35), e5 (36)
CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)

BASE_FEATURE_DIM = 404
TOTAL_FEATURE_DIM = 808  # 404 middlegame + 404 endgame


def get_base_feature_names() -> List[str]:
    """Returns human-readable names for the 404 base features."""
    names: List[str] = []
    
    # 1. Material Imbalances (0..4)
    for pt in MATERIAL_PIECE_TYPES:
        names.append(f"mat_{PIECE_NAMES[pt]}_diff")
        
    # 2. Antisymmetric Piece-Square Tables (5..388: 6 pieces x 64 squares)
    for pt in ALL_PIECE_TYPES:
        pt_name = PIECE_NAMES[pt]
        for sq in range(64):
            names.append(f"pst_{pt_name}_{chess.square_name(sq)}")
            
    # 3. Passed Pawn Rank Buckets (389..394: ranks 2..7)
    for r in range(2, 8):
        names.append(f"passed_pawn_rank_{r}")
        
    # 4. Strategic features (395..403)
    names.append("bishop_pair_diff")
    names.append("rook_open_file_diff")
    names.append("rook_semi_open_file_diff")
    names.append("tempo")
    names.append("doubled_pawns_diff")
    names.append("isolated_pawns_diff")
    names.append("center_control_diff")
    names.append("king_shield_diff")
    names.append("pseudo_mobility_diff")
    
    return names


def get_feature_names() -> List[str]:
    """Returns human-readable names for all 808 tapered features."""
    base_names = get_base_feature_names()
    mg_names = [f"mg_{name}" for name in base_names]
    eg_names = [f"eg_{name}" for name in base_names]
    return mg_names + eg_names


def extract_base_features_numpy(board: chess.Board) -> np.ndarray:
    """
    Extracts the 404 base features from chess.Board.
    Guarantees exact antisymmetry: extract(board) == -extract(board.mirror()).
    """
    arr = np.zeros(BASE_FEATURE_DIM, dtype=np.float32)
    
    white_occupied = board.occupied_co[chess.WHITE]
    black_occupied = board.occupied_co[chess.BLACK]
    all_occupied = board.occupied
    
    # 1. Material Imbalances (5 features, idx: 0..4)
    for i, pt in enumerate(MATERIAL_PIECE_TYPES):
        w_cnt = board.pieces_mask(pt, chess.WHITE).bit_count()
        b_cnt = board.pieces_mask(pt, chess.BLACK).bit_count()
        arr[i] = float(w_cnt - b_cnt)
        
    # 2. Antisymmetric PSTs (384 features, idx: 5..388)
    # 6 pieces x 64 squares. White: +1 at sq, Black: -1 at mirror(sq)
    pst_offset = 5
    for pt_idx, pt in enumerate(ALL_PIECE_TYPES):
        base_pt = pst_offset + pt_idx * 64
        # White pieces
        w_mask = board.pieces_mask(pt, chess.WHITE)
        for sq in chess.scan_forward(w_mask):
            arr[base_pt + sq] += 1.0
            
        # Black pieces (mirrored square)
        b_mask = board.pieces_mask(pt, chess.BLACK)
        for sq in chess.scan_forward(b_mask):
            mirror_sq = sq ^ 56  # chess.square_mirror(sq)
            arr[base_pt + mirror_sq] -= 1.0
            
    # 3. Passed Pawn Rank Buckets (6 features, idx: 389..394: ranks 2..7 -> indices 0..5)
    w_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
    
    passed_offset = 389
    for sq in chess.scan_forward(w_pawns):
        if (WHITE_PASSED_SPAN[sq] & b_pawns) == 0:
            rank = sq // 8  # 0..7 (pawn ranks 1..6)
            bucket = rank - 1
            if 0 <= bucket <= 5:
                arr[passed_offset + bucket] += 1.0
                
    for sq in chess.scan_forward(b_pawns):
        if (BLACK_PASSED_SPAN[sq] & w_pawns) == 0:
            rank = sq // 8
            adv_rank = 7 - rank
            bucket = adv_rank - 1
            if 0 <= bucket <= 5:
                arr[passed_offset + bucket] -= 1.0
                
    # 4. Bishop Pair (1 feature, idx: 395)
    w_bishops = board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count()
    b_bishops = board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count()
    arr[395] = float((1 if w_bishops >= 2 else 0) - (1 if b_bishops >= 2 else 0))
    
    # 5. Rook on Open & Semi-Open Files (2 features, idx: 396, 397)
    w_rooks = board.pieces_mask(chess.ROOK, chess.WHITE)
    b_rooks = board.pieces_mask(chess.ROOK, chess.BLACK)
    w_open_rooks, b_open_rooks = 0, 0
    w_semi_rooks, b_semi_rooks = 0, 0
    
    for f in range(8):
        f_mask = chess.BB_FILES[f]
        w_p_file = (w_pawns & f_mask) != 0
        b_p_file = (b_pawns & f_mask) != 0
        
        is_open = (not w_p_file) and (not b_p_file)
        w_semi = (not w_p_file) and b_p_file
        b_semi = (not b_p_file) and w_p_file
        
        w_r_on_f = (w_rooks & f_mask).bit_count()
        b_r_on_f = (b_rooks & f_mask).bit_count()
        
        if is_open:
            w_open_rooks += w_r_on_f
            b_open_rooks += b_r_on_f
        else:
            if w_semi:
                w_semi_rooks += w_r_on_f
            if b_semi:
                b_semi_rooks += b_r_on_f
                
    arr[396] = float(w_open_rooks - b_open_rooks)
    arr[397] = float(w_semi_rooks - b_semi_rooks)
    
    # 6. Tempo (1 feature, idx: 398)
    arr[398] = 1.0 if board.turn == chess.WHITE else -1.0
    
    # 7. Doubled Pawns & Isolated Pawns (2 features, idx: 399, 400)
    w_doubled, b_doubled = 0, 0
    w_isolated, b_isolated = 0, 0
    
    for f in range(8):
        f_mask = chess.BB_FILES[f]
        w_cnt = (w_pawns & f_mask).bit_count()
        b_cnt = (b_pawns & f_mask).bit_count()
        
        if w_cnt > 1:
            w_doubled += (w_cnt - 1)
        if b_cnt > 1:
            b_doubled += (b_cnt - 1)
            
        # Adjacent files mask
        adj_mask = 0
        if f > 0:
            adj_mask |= chess.BB_FILES[f - 1]
        if f < 7:
            adj_mask |= chess.BB_FILES[f + 1]
            
        if w_cnt > 0 and (w_pawns & adj_mask) == 0:
            w_isolated += w_cnt
        if b_cnt > 0 and (b_pawns & adj_mask) == 0:
            b_isolated += b_cnt
            
    arr[399] = float(w_doubled - b_doubled)
    arr[400] = float(w_isolated - b_isolated)
    
    # 8. Center Control (1 feature, idx: 401)
    w_center_att = 0
    b_center_att = 0
    for sq in CENTER_SQUARES:
        w_center_att += board.attackers_mask(chess.WHITE, sq).bit_count()
        b_center_att += board.attackers_mask(chess.BLACK, sq).bit_count()
    arr[401] = float(w_center_att - b_center_att)
    
    # 9. King Shield Integrity (1 feature, idx: 402)
    w_shield, b_shield = 0, 0
    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)
    
    if w_king_sq is not None:
        kr = w_king_sq // 8
        kf = w_king_sq % 8
        if kr <= 3:  # Only count shield when king is near home
            for fi in range(max(0, kf - 1), min(8, kf + 2)):
                for ri in range(kr + 1, min(8, kr + 3)):
                    if (w_pawns & (1 << chess.square(fi, ri))) != 0:
                        w_shield += 1
                        
    if b_king_sq is not None:
        kr = b_king_sq // 8
        kf = b_king_sq % 8
        if kr >= 4:  # Only count shield when king is near home
            for fi in range(max(0, kf - 1), min(8, kf + 2)):
                for ri in range(max(0, kr - 2), kr):
                    if (b_pawns & (1 << chess.square(fi, ri))) != 0:
                        b_shield += 1
                        
    arr[402] = float(w_shield - b_shield)
    
    # 10. Fast Bitboard Pseudo-Mobility (1 feature, idx: 403)
    # Sum pseudo-legal target squares for N, B, R, Q, K
    w_mob, b_mob = 0, 0
    
    # Knights
    for sq in chess.scan_forward(board.pieces_mask(chess.KNIGHT, chess.WHITE)):
        w_mob += (chess.BB_KNIGHT_ATTACKS[sq] & ~white_occupied).bit_count()
    for sq in chess.scan_forward(board.pieces_mask(chess.KNIGHT, chess.BLACK)):
        b_mob += (chess.BB_KNIGHT_ATTACKS[sq] & ~black_occupied).bit_count()
        
    # Bishops
    for sq in chess.scan_forward(board.pieces_mask(chess.BISHOP, chess.WHITE)):
        w_mob += (board.attacks_mask(sq) & ~white_occupied).bit_count()
    for sq in chess.scan_forward(board.pieces_mask(chess.BISHOP, chess.BLACK)):
        b_mob += (board.attacks_mask(sq) & ~black_occupied).bit_count()
        
    # Rooks
    for sq in chess.scan_forward(board.pieces_mask(chess.ROOK, chess.WHITE)):
        w_mob += (board.attacks_mask(sq) & ~white_occupied).bit_count()
    for sq in chess.scan_forward(board.pieces_mask(chess.ROOK, chess.BLACK)):
        b_mob += (board.attacks_mask(sq) & ~black_occupied).bit_count()
        
    # Queens
    for sq in chess.scan_forward(board.pieces_mask(chess.QUEEN, chess.WHITE)):
        w_mob += (board.attacks_mask(sq) & ~white_occupied).bit_count()
    for sq in chess.scan_forward(board.pieces_mask(chess.QUEEN, chess.BLACK)):
        b_mob += (board.attacks_mask(sq) & ~black_occupied).bit_count()
        
    # Kings
    if w_king_sq is not None:
        w_mob += (chess.BB_KING_ATTACKS[w_king_sq] & ~white_occupied).bit_count()
    if b_king_sq is not None:
        b_mob += (chess.BB_KING_ATTACKS[b_king_sq] & ~black_occupied).bit_count()
        
    arr[403] = float(w_mob - b_mob)
    
    return arr


def compute_game_phase(board: chess.Board) -> float:
    """
    Computes game phase factor rho in [0.0, 1.0] from remaining non-pawn material:
      Knight = 1, Bishop = 1, Rook = 2, Queen = 4
      Max non-pawn phase = 24.
      rho = 1.0 in opening/middlegame; rho = 0.0 in pawn endgames.
      Guaranteed symmetric: rho(board) == rho(board.mirror()).
    """
    knights = board.pieces_mask(chess.KNIGHT, chess.WHITE).bit_count() + board.pieces_mask(chess.KNIGHT, chess.BLACK).bit_count()
    bishops = board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count() + board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count()
    rooks   = board.pieces_mask(chess.ROOK, chess.WHITE).bit_count() + board.pieces_mask(chess.ROOK, chess.BLACK).bit_count()
    queens  = board.pieces_mask(chess.QUEEN, chess.WHITE).bit_count() + board.pieces_mask(chess.QUEEN, chess.BLACK).bit_count()
    
    phase = 1 * knights + 1 * bishops + 2 * rooks + 4 * queens
    return float(min(phase, 24) / 24.0)


def extract_features_numpy(board: chess.Board) -> np.ndarray:
    """
    Extracts the full 808-dimensional tapered feature vector:
      x_tapered = concat(rho * x_base, (1.0 - rho) * x_base)
    Linear dot product w^T x_tapered automatically applies:
      w_mg^T (rho * x_base) + w_eg^T ((1 - rho) * x_base)
    Guaranteed exact antisymmetry: extract(board) == -extract(board.mirror()).
    """
    x_base = extract_base_features_numpy(board)
    rho = compute_game_phase(board)
    
    x_tapered = np.empty(TOTAL_FEATURE_DIM, dtype=np.float32)
    x_tapered[:BASE_FEATURE_DIM] = rho * x_base
    x_tapered[BASE_FEATURE_DIM:] = (1.0 - rho) * x_base
    return x_tapered


def extract_features(board: chess.Board) -> torch.Tensor:
    """Extracts features as a 1D PyTorch float32 tensor."""
    return torch.from_numpy(extract_features_numpy(board))


def get_default_heuristic_weights() -> torch.Tensor:
    """
    Returns standard classical chess heuristic weights for the 808-dimensional tapered model:
      [0..403]: Middlegame weights (w_mg)
      [404..807]: Endgame weights (w_eg)
    """
    w = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    
    # Material values (Pawn=100, Knight=320, Bishop=330, Rook=500, Queen=900)
    # Middlegame
    w[0] = 1.0     # Pawn
    w[1] = 3.2     # Knight
    w[2] = 3.3     # Bishop
    w[3] = 5.0     # Rook
    w[4] = 9.0     # Queen
    
    # Endgame (Pawns and Rooks slightly more valuable)
    w[BASE_FEATURE_DIM + 0] = 1.2     # Pawn
    w[BASE_FEATURE_DIM + 1] = 3.1     # Knight
    w[BASE_FEATURE_DIM + 2] = 3.3     # Bishop
    w[BASE_FEATURE_DIM + 3] = 5.2     # Rook
    w[BASE_FEATURE_DIM + 4] = 9.0     # Queen
    
    # Passed Pawn Rank Buckets: ranks 2..7
    # Middlegame: [0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
    # Endgame:    [0.2, 0.4, 0.8, 1.6, 3.2, 6.0]
    mg_passed = [0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
    eg_passed = [0.2, 0.4, 0.8, 1.6, 3.2, 6.0]
    for b in range(6):
        w[389 + b] = mg_passed[b]
        w[BASE_FEATURE_DIM + 389 + b] = eg_passed[b]
        
    # Strategic features:
    # Bishop pair
    w[395] = 0.5                      # MG
    w[BASE_FEATURE_DIM + 395] = 0.6   # EG
    
    # Rook on open / semi-open
    w[396] = 0.4                      # MG open
    w[BASE_FEATURE_DIM + 396] = 0.4   # EG open
    w[397] = 0.2                      # MG semi
    w[BASE_FEATURE_DIM + 397] = 0.2   # EG semi
    
    # Tempo
    w[398] = 0.1                      # MG
    w[BASE_FEATURE_DIM + 398] = 0.1   # EG
    
    # Doubled pawns penalty
    w[399] = -0.2                     # MG
    w[BASE_FEATURE_DIM + 399] = -0.3  # EG
    
    # Isolated pawns penalty
    w[400] = -0.25                    # MG
    w[BASE_FEATURE_DIM + 400] = -0.35 # EG
    
    # Center control
    w[401] = 0.15                     # MG
    w[BASE_FEATURE_DIM + 401] = 0.05  # EG
    
    # King shield
    w[402] = 0.3                      # MG (crucial in middlegame)
    w[BASE_FEATURE_DIM + 402] = 0.0   # EG (unconstrained in endgame)
    
    # Pseudo-mobility
    w[403] = 0.04                     # MG
    w[BASE_FEATURE_DIM + 403] = 0.06  # EG
    
    return torch.from_numpy(w)
