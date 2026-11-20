"""
feature_extractor.py
--------------------
Upgraded Classical Feature Extractor for Bumbot V2 (Phases P1 & P3):
  - Strict Generalized Linear Model (GLM) formulation:
      V_hat(s) = g^-1(w^T x_tapered(s))
  - Antisymmetric Piece-Square Tables (PST) with dead pawn squares pruned:
      Pawns only exist on Ranks 2-7 (48 squares instead of 64).
      Knights, Bishops, Rooks, Queens, Kings have 64 squares each.
      Total PST features: 48 + 5 x 64 = 368 features.
  - Material Imbalance (5 base features: P, N, B, R, Q).
  - Passed Pawn Rank Progression (6 base features: ranks 2 through 7).
  - Established Strategic Features (8 base features):
      Bishop pair, Open file rook, Semi-open file rook, Tempo,
      Doubled pawns, Isolated pawns, Center control, King shield.
  - Orthogonal High-Signal Basis Features (P3 Upgrades, 13 base features):
      1. Safe per-piece mobility (4 features: N, B, R, Q excluding enemy pawn attacks).
      2. Endgame mop-up (3 features: losing King to edge, King-to-King proximity, King-to-passer).
      3. King-zone attack pressure (1 feature: enemy pieces attacking ring around King).
      4. Knight outposts (1 feature: supported by friendly pawn, immune to enemy pawns).
      5. Protected passed pawns (1 feature: passer defended by friendly pawn).
      6. Connected pawn phalanxes (1 feature: friendly pawns side-by-side on same rank).
      7. Density interactions (2 features: Knight in closed, Rook in open positions).

Base Feature Dimension: 400
Tapered Game Phase Feature Dimension: 2 x 400 = 800
  concat(rho * x_base, (1.0 - rho) * x_base)

Guarantees 100% exact mathematical antisymmetry:
  extract(board) == -extract(board.mirror())

Provides Selective L2 Regularization (Hierarchical Shrinkage) mask:
  - Base material counts (idx 0..4 and 400..404) are unpenalized (mask = 0.0).
  - Positional/PST parameters are regularized (mask = 1.0) to zero mean,
    breaking exact multicollinearity.
"""

from typing import List, Tuple
import numpy as np
import torch
import chess

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

# King zone masks: 3x3 box around King
KING_ZONE_MASKS = [0] * 64
for sq in range(64):
    kf = chess.square_file(sq)
    kr = chess.square_rank(sq)
    mask = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = kf + df, kr + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                mask |= (1 << chess.square(nf, nr))
    KING_ZONE_MASKS[sq] = mask

# Center squares: d4 (27), e4 (28), d5 (35), e5 (36)
CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)

BASE_FEATURE_DIM = 400
TOTAL_FEATURE_DIM = 800  # 400 middlegame + 400 endgame


def get_base_feature_names() -> List[str]:
    """Returns human-readable names for the 400 base features."""
    names: List[str] = []
    
    # 1. Material Imbalances (0..4)
    for pt in MATERIAL_PIECE_TYPES:
        names.append(f"mat_{PIECE_NAMES[pt]}_diff")
        
    # 2. Antisymmetric Piece-Square Tables (5..372)
    # Pawn PST: 48 squares (Ranks 2-7, squares 8..55)
    for sq in range(8, 56):
        names.append(f"pst_pawn_{chess.square_name(sq)}")
        
    # Other pieces: 64 squares each (Knights, Bishops, Rooks, Queens, Kings)
    for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        pt_name = PIECE_NAMES[pt]
        for sq in range(64):
            names.append(f"pst_{pt_name}_{chess.square_name(sq)}")
            
    # 3. Passed Pawn Rank Buckets (373..378: ranks 2..7)
    for r in range(2, 8):
        names.append(f"passed_pawn_rank_{r}")
        
    # 4. Established Strategic features (379..386)
    names.append("bishop_pair_diff")
    names.append("rook_open_file_diff")
    names.append("rook_semi_open_file_diff")
    names.append("tempo")
    names.append("doubled_pawns_diff")
    names.append("isolated_pawns_diff")
    names.append("center_control_diff")
    names.append("king_shield_diff")
    
    # 5. Upgraded Orthogonal features (P3, 387..399)
    names.append("safe_mobility_knight")
    names.append("safe_mobility_bishop")
    names.append("safe_mobility_rook")
    names.append("safe_mobility_queen")
    names.append("endgame_losing_king_to_edge")
    names.append("endgame_king_proximity")
    names.append("endgame_king_to_passer")
    names.append("king_zone_attack_pressure")
    names.append("knight_outpost_diff")
    names.append("protected_passed_pawn_diff")
    names.append("pawn_phalanx_diff")
    names.append("knight_closed_density_interaction")
    names.append("rook_open_density_interaction")
    
    return names


def get_feature_names() -> List[str]:
    """Returns human-readable names for all 800 tapered features."""
    base_names = get_base_feature_names()
    mg_names = [f"mg_{name}" for name in base_names]
    eg_names = [f"eg_{name}" for name in base_names]
    return mg_names + eg_names


def get_regularization_mask() -> np.ndarray:
    """
    Returns an 800-dimensional mask for Selective L2 Regularization (Hierarchical Shrinkage).
    Material features (0..4 in MG and 400..404 in EG) are unpenalized (0.0).
    All PST and positional features are regularized (1.0) to enforce zero-mean offsets.
    """
    mask = np.ones(TOTAL_FEATURE_DIM, dtype=np.float32)
    mask[0:5] = 0.0
    mask[BASE_FEATURE_DIM:BASE_FEATURE_DIM + 5] = 0.0
    return mask


def extract_base_features_numpy(board: chess.Board) -> np.ndarray:
    """
    Extracts the 400 base features from chess.Board.
    Guarantees exact antisymmetry: extract(board) == -extract(board.mirror()).
    """
    arr = np.zeros(BASE_FEATURE_DIM, dtype=np.float32)
    
    white_occupied = board.occupied_co[chess.WHITE]
    black_occupied = board.occupied_co[chess.BLACK]
    
    w_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)
    
    # 1. Material Imbalances (5 features, idx: 0..4)
    for i, pt in enumerate(MATERIAL_PIECE_TYPES):
        w_cnt = board.pieces_mask(pt, chess.WHITE).bit_count()
        b_cnt = board.pieces_mask(pt, chess.BLACK).bit_count()
        arr[i] = float(w_cnt - b_cnt)
        
    # 2. Antisymmetric PSTs (368 features, idx: 5..372)
    # 2a. Pawns (48 squares, ranks 2-7: squares 8..55, idx: 5..52)
    for sq in chess.scan_forward(w_pawns):
        if 8 <= sq <= 55:
            arr[5 + (sq - 8)] += 1.0
    for sq in chess.scan_forward(b_pawns):
        mirror_sq = sq ^ 56
        if 8 <= mirror_sq <= 55:
            arr[5 + (mirror_sq - 8)] -= 1.0
            
    # 2b. Knights, Bishops, Rooks, Queens, Kings (64 squares each, idx: 53..372)
    pst_offset = 53
    for pt_idx, pt in enumerate([chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]):
        base_pt = pst_offset + pt_idx * 64
        # White pieces
        for sq in chess.scan_forward(board.pieces_mask(pt, chess.WHITE)):
            arr[base_pt + sq] += 1.0
        # Black pieces (mirrored square)
        for sq in chess.scan_forward(board.pieces_mask(pt, chess.BLACK)):
            mirror_sq = sq ^ 56
            arr[base_pt + mirror_sq] -= 1.0
            
    # 3. Passed Pawn Rank Progression (6 features, idx: 373..378: ranks 2..7)
    passed_offset = 373
    w_passed_mask = 0
    b_passed_mask = 0
    for sq in chess.scan_forward(w_pawns):
        if (WHITE_PASSED_SPAN[sq] & b_pawns) == 0:
            w_passed_mask |= (1 << sq)
            rank = sq // 8
            bucket = rank - 1
            if 0 <= bucket <= 5:
                arr[passed_offset + bucket] += 1.0
                
    for sq in chess.scan_forward(b_pawns):
        if (BLACK_PASSED_SPAN[sq] & w_pawns) == 0:
            b_passed_mask |= (1 << sq)
            rank = sq // 8
            adv_rank = 7 - rank
            bucket = adv_rank - 1
            if 0 <= bucket <= 5:
                arr[passed_offset + bucket] -= 1.0
                
    # 4. Established Strategic Features (8 features, idx: 379..386)
    # 4a. Bishop pair (idx: 379)
    w_bishops = board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count()
    b_bishops = board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count()
    arr[379] = float((1 if w_bishops >= 2 else 0) - (1 if b_bishops >= 2 else 0))
    
    # 4b. Rook on Open & Semi-Open Files (idx: 380, 381)
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
        
        w_r_cnt = (w_rooks & f_mask).bit_count()
        b_r_cnt = (b_rooks & f_mask).bit_count()
        
        if is_open:
            w_open_rooks += w_r_cnt
            b_open_rooks += b_r_cnt
        else:
            if w_semi:
                w_semi_rooks += w_r_cnt
            if b_semi:
                b_semi_rooks += b_r_cnt
                
    arr[380] = float(w_open_rooks - b_open_rooks)
    arr[381] = float(w_semi_rooks - b_semi_rooks)
    
    # 4c. Tempo (idx: 382)
    arr[382] = 1.0 if board.turn == chess.WHITE else -1.0
    
    # 4d. Doubled Pawns & Isolated Pawns (idx: 383, 384)
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
            
        adj_mask = 0
        if f > 0:
            adj_mask |= chess.BB_FILES[f - 1]
        if f < 7:
            adj_mask |= chess.BB_FILES[f + 1]
            
        if w_cnt > 0 and (w_pawns & adj_mask) == 0:
            w_isolated += w_cnt
        if b_cnt > 0 and (b_pawns & adj_mask) == 0:
            b_isolated += b_cnt
            
    arr[383] = float(w_doubled - b_doubled)
    arr[384] = float(w_isolated - b_isolated)
    
    # 4e. Center Control (idx: 385)
    w_center_att, b_center_att = 0, 0
    for sq in CENTER_SQUARES:
        w_center_att += board.attackers_mask(chess.WHITE, sq).bit_count()
        b_center_att += board.attackers_mask(chess.BLACK, sq).bit_count()
    arr[385] = float(w_center_att - b_center_att)
    
    # 4f. King Shield Integrity (idx: 386)
    w_shield, b_shield = 0, 0
    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)
    
    if w_king_sq is not None:
        kr = w_king_sq // 8
        kf = w_king_sq % 8
        if kr <= 3:
            for fi in range(max(0, kf - 1), min(8, kf + 2)):
                for ri in range(kr + 1, min(8, kr + 3)):
                    if (w_pawns & (1 << chess.square(fi, ri))) != 0:
                        w_shield += 1
                        
    if b_king_sq is not None:
        kr = b_king_sq // 8
        kf = b_king_sq % 8
        if kr >= 4:
            for fi in range(max(0, kf - 1), min(8, kf + 2)):
                for ri in range(max(0, kr - 2), kr):
                    if (b_pawns & (1 << chess.square(fi, ri))) != 0:
                        b_shield += 1
                        
    arr[386] = float(w_shield - b_shield)
    
    # 5. Upgraded Orthogonal Features (P3, idx: 387..399)
    # 5a. Safe per-piece mobility (idx: 387..390)
    # Pawn attack bitboards (direction-sensitive)
    w_pawn_attacks = ((w_pawns & ~chess.BB_FILE_A) << 7) | ((w_pawns & ~chess.BB_FILE_H) << 9)
    b_pawn_attacks = ((b_pawns & ~chess.BB_FILE_A) >> 9) | ((b_pawns & ~chess.BB_FILE_H) >> 7)
    
    w_safe_squares = ~white_occupied & ~b_pawn_attacks
    b_safe_squares = ~black_occupied & ~w_pawn_attacks
    
    # Knight mobility (idx: 387)
    w_n_mob = sum((chess.BB_KNIGHT_ATTACKS[sq] & w_safe_squares).bit_count() for sq in chess.scan_forward(board.pieces_mask(chess.KNIGHT, chess.WHITE)))
    b_n_mob = sum((chess.BB_KNIGHT_ATTACKS[sq] & b_safe_squares).bit_count() for sq in chess.scan_forward(board.pieces_mask(chess.KNIGHT, chess.BLACK)))
    arr[387] = float(w_n_mob - b_n_mob) / 8.0
    
    # Bishop mobility (idx: 388)
    w_b_mob = sum((board.attacks_mask(sq) & w_safe_squares).bit_count() for sq in chess.scan_forward(board.pieces_mask(chess.BISHOP, chess.WHITE)))
    b_b_mob = sum((board.attacks_mask(sq) & b_safe_squares).bit_count() for sq in chess.scan_forward(board.pieces_mask(chess.BISHOP, chess.BLACK)))
    arr[388] = float(w_b_mob - b_b_mob) / 8.0
    
    # Rook mobility (idx: 389)
    w_r_mob = sum((board.attacks_mask(sq) & w_safe_squares).bit_count() for sq in chess.scan_forward(board.pieces_mask(chess.ROOK, chess.WHITE)))
    b_r_mob = sum((board.attacks_mask(sq) & b_safe_squares).bit_count() for sq in chess.scan_forward(board.pieces_mask(chess.ROOK, chess.BLACK)))
    arr[389] = float(w_r_mob - b_r_mob) / 14.0
    
    # Queen mobility (idx: 390)
    w_q_mob = sum((board.attacks_mask(sq) & w_safe_squares).bit_count() for sq in chess.scan_forward(board.pieces_mask(chess.QUEEN, chess.WHITE)))
    b_q_mob = sum((board.attacks_mask(sq) & b_safe_squares).bit_count() for sq in chess.scan_forward(board.pieces_mask(chess.QUEEN, chess.BLACK)))
    arr[390] = float(w_q_mob - b_q_mob) / 28.0
    
    # Net material imbalance for mop-up gating
    mat_net = (
        1.0 * arr[0]
        + 3.2 * arr[1]
        + 3.3 * arr[2]
        + 5.0 * arr[3]
        + 9.0 * arr[4]
    )
    
    # 5b. Endgame Mop-Up (3 features, idx: 391..393)
    if w_king_sq is not None and b_king_sq is not None:
        # Distance to center: max is 7.0 (at corners: |0-3.5| + |0-3.5| = 7.0)
        w_dist_center = abs(w_king_sq // 8 - 3.5) + abs(w_king_sq % 8 - 3.5)
        b_dist_center = abs(b_king_sq // 8 - 3.5) + abs(b_king_sq % 8 - 3.5)
        
        # 391: Losing King to edge
        if mat_net >= 2.0:
            arr[391] = b_dist_center / 7.0
        elif mat_net <= -2.0:
            arr[391] = -w_dist_center / 7.0
            
        # 392: King-to-King proximity (max Manhattan distance is 14)
        dist_kings = abs(w_king_sq // 8 - b_king_sq // 8) + abs(w_king_sq % 8 - b_king_sq % 8)
        proximity = (14.0 - dist_kings) / 14.0
        if mat_net >= 2.0:
            arr[392] = proximity
        elif mat_net <= -2.0:
            arr[392] = -proximity
            
        # 393: King-to-Passer proximity
        w_passer_score = 0.0
        if w_passed_mask != 0:
            # Most advanced White passer
            adv_sq = max(chess.scan_forward(w_passed_mask), key=lambda s: s // 8)
            pr, pf = adv_sq // 8, adv_sq % 8
            w_k_dist = max(abs(w_king_sq // 8 - pr), abs(w_king_sq % 8 - pf))
            b_k_dist = max(abs(b_king_sq // 8 - pr), abs(b_king_sq % 8 - pf))
            w_passer_score = (b_k_dist - w_k_dist) / 7.0
            
        b_passer_score = 0.0
        if b_passed_mask != 0:
            # Most advanced Black passer (smallest rank)
            adv_sq = min(chess.scan_forward(b_passed_mask), key=lambda s: s // 8)
            pr, pf = adv_sq // 8, adv_sq % 8
            w_k_dist = max(abs(w_king_sq // 8 - pr), abs(w_king_sq % 8 - pf))
            b_k_dist = max(abs(b_king_sq // 8 - pr), abs(b_king_sq % 8 - pf))
            b_passer_score = (w_k_dist - b_k_dist) / 7.0
            
        arr[393] = float(w_passer_score - b_passer_score)
        
    # 5c. King-Zone Attack Pressure (idx: 394)
    w_att_zone, b_att_zone = 0, 0
    if b_king_sq is not None:
        zone_b = KING_ZONE_MASKS[b_king_sq]
        # White pieces attacking Black king zone
        for sq in chess.scan_forward(board.occupied_co[chess.WHITE] & ~w_pawns):
            if (board.attacks_mask(sq) & zone_b) != 0:
                w_att_zone += 1
    if w_king_sq is not None:
        zone_w = KING_ZONE_MASKS[w_king_sq]
        # Black pieces attacking White king zone
        for sq in chess.scan_forward(board.occupied_co[chess.BLACK] & ~b_pawns):
            if (board.attacks_mask(sq) & zone_w) != 0:
                b_att_zone += 1
    arr[394] = float(w_att_zone - b_att_zone)
    
    # 5d. Knight Outposts (idx: 395)
    # Knight defended by friendly pawn, immune to enemy pawns on adjacent files
    w_outposts, b_outposts = 0, 0
    for sq in chess.scan_forward(board.pieces_mask(chess.KNIGHT, chess.WHITE)):
        r = sq // 8
        f = sq % 8
        if 3 <= r <= 5:  # Ranks 4, 5, 6
            if (w_pawn_attacks & (1 << sq)) != 0:
                adj_files = 0
                if f > 0: adj_files |= chess.BB_FILES[f - 1]
                if f < 7: adj_files |= chess.BB_FILES[f + 1]
                # Check if enemy pawns can attack this square from ahead
                enemy_attack_span = 0
                for ri in range(r + 1, 8):
                    enemy_attack_span |= (adj_files & chess.BB_RANKS[ri])
                if (b_pawns & enemy_attack_span) == 0:
                    w_outposts += 1
                    
    for sq in chess.scan_forward(board.pieces_mask(chess.KNIGHT, chess.BLACK)):
        r = sq // 8
        f = sq % 8
        if 2 <= r <= 4:  # Ranks 3, 4, 5 for Black
            if (b_pawn_attacks & (1 << sq)) != 0:
                adj_files = 0
                if f > 0: adj_files |= chess.BB_FILES[f - 1]
                if f < 7: adj_files |= chess.BB_FILES[f + 1]
                enemy_attack_span = 0
                for ri in range(0, r):
                    enemy_attack_span |= (adj_files & chess.BB_RANKS[ri])
                if (w_pawns & enemy_attack_span) == 0:
                    b_outposts += 1
                    
    arr[395] = float(w_outposts - b_outposts)
    
    # 5e. Protected Passed Pawns (idx: 396)
    w_prot_passers = (w_passed_mask & w_pawn_attacks).bit_count()
    b_prot_passers = (b_passed_mask & b_pawn_attacks).bit_count()
    arr[396] = float(w_prot_passers - b_prot_passers)
    
    # 5f. Connected Pawn Phalanxes (idx: 397)
    # Friendly pawns on the same rank on adjacent files
    w_phalanxes, b_phalanxes = 0, 0
    for f in range(7):
        mask_f = chess.BB_FILES[f]
        mask_next = chess.BB_FILES[f + 1]
        for r in range(1, 7):
            rank_mask = chess.BB_RANKS[r]
            if (w_pawns & mask_f & rank_mask) and (w_pawns & mask_next & rank_mask):
                w_phalanxes += 1
            if (b_pawns & mask_f & rank_mask) and (b_pawns & mask_next & rank_mask):
                b_phalanxes += 1
    arr[397] = float(w_phalanxes - b_phalanxes)
    
    # 5g. GLM-Legal Piece-Pawn Density Interactions (idx: 398, 399)
    total_pawns = (w_pawns | b_pawns).bit_count()
    arr[398] = float(arr[1]) * (total_pawns / 16.0)          # Knight increases value in closed games
    arr[399] = float(arr[3]) * ((16.0 - total_pawns) / 16.0) # Rook increases value in open games
    
    return arr


def compute_game_phase(board: chess.Board) -> float:
    """Computes symmetric game phase factor rho in [0.0, 1.0]."""
    knights = board.pieces_mask(chess.KNIGHT, chess.WHITE).bit_count() + board.pieces_mask(chess.KNIGHT, chess.BLACK).bit_count()
    bishops = board.pieces_mask(chess.BISHOP, chess.WHITE).bit_count() + board.pieces_mask(chess.BISHOP, chess.BLACK).bit_count()
    rooks   = board.pieces_mask(chess.ROOK, chess.WHITE).bit_count() + board.pieces_mask(chess.ROOK, chess.BLACK).bit_count()
    queens  = board.pieces_mask(chess.QUEEN, chess.WHITE).bit_count() + board.pieces_mask(chess.QUEEN, chess.BLACK).bit_count()
    phase = 1 * knights + 1 * bishops + 2 * rooks + 4 * queens
    return float(min(phase, 24) / 24.0)


def extract_features_numpy(board: chess.Board) -> np.ndarray:
    """Extracts 800-dimensional tapered feature vector."""
    x_base = extract_base_features_numpy(board)
    rho = compute_game_phase(board)
    x_tapered = np.empty(TOTAL_FEATURE_DIM, dtype=np.float32)
    x_tapered[:BASE_FEATURE_DIM] = rho * x_base
    x_tapered[BASE_FEATURE_DIM:] = (1.0 - rho) * x_base
    return x_tapered


def extract_features(board: chess.Board) -> torch.Tensor:
    """Extracts features as PyTorch tensor."""
    return torch.from_numpy(extract_features_numpy(board))


def get_default_heuristic_weights() -> torch.Tensor:
    """Default textbook weights for 800-dimensional tapered model."""
    w = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    
    # 1. Base Material values (MG / EG)
    # Pawn, Knight, Bishop, Rook, Queen
    w[0] = 1.0; w[1] = 3.2; w[2] = 3.3; w[3] = 5.0; w[4] = 9.0
    w[BASE_FEATURE_DIM + 0] = 1.2
    w[BASE_FEATURE_DIM + 1] = 3.1
    w[BASE_FEATURE_DIM + 2] = 3.3
    w[BASE_FEATURE_DIM + 3] = 5.2
    w[BASE_FEATURE_DIM + 4] = 9.0
    
    # 2. Passed pawn rank buckets (ranks 2..7)
    mg_passed = [0.1, 0.2, 0.4, 0.8, 1.5, 3.0]
    eg_passed = [0.2, 0.4, 0.8, 1.6, 3.2, 6.0]
    for b in range(6):
        w[373 + b] = mg_passed[b]
        w[BASE_FEATURE_DIM + 373 + b] = eg_passed[b]
        
    # 3. Established Strategic features
    w[379] = 0.50; w[BASE_FEATURE_DIM + 379] = 0.60  # Bishop pair
    w[380] = 0.40; w[BASE_FEATURE_DIM + 380] = 0.40  # Open rook
    w[381] = 0.20; w[BASE_FEATURE_DIM + 381] = 0.20  # Semi-open rook
    w[382] = 0.10; w[BASE_FEATURE_DIM + 382] = 0.05  # Tempo
    w[383] = -0.20; w[BASE_FEATURE_DIM + 383] = -0.30 # Doubled pawns
    w[384] = -0.25; w[BASE_FEATURE_DIM + 384] = -0.35 # Isolated pawns
    w[385] = 0.15; w[BASE_FEATURE_DIM + 385] = 0.05  # Center control
    w[386] = 0.30; w[BASE_FEATURE_DIM + 386] = 0.00  # King shield
    
    # 4. Orthogonal Upgraded features (P3)
    # Safe mobility
    w[387] = 0.10; w[BASE_FEATURE_DIM + 387] = 0.10  # Knight safe mob
    w[388] = 0.10; w[BASE_FEATURE_DIM + 388] = 0.10  # Bishop safe mob
    w[389] = 0.10; w[BASE_FEATURE_DIM + 389] = 0.10  # Rook safe mob
    w[390] = 0.05; w[BASE_FEATURE_DIM + 390] = 0.05  # Queen safe mob
    
    # Endgame Mop-Up
    w[391] = 0.00; w[BASE_FEATURE_DIM + 391] = 0.80  # Losing King to edge
    w[392] = 0.00; w[BASE_FEATURE_DIM + 392] = 0.60  # King-to-King proximity
    w[393] = 0.10; w[BASE_FEATURE_DIM + 393] = 0.50  # King-to-passer
    
    # King-zone attack pressure
    w[394] = 0.25; w[BASE_FEATURE_DIM + 394] = 0.05
    
    # Outposts & pawn structure
    w[395] = 0.30; w[BASE_FEATURE_DIM + 395] = 0.20  # Knight outpost
    w[396] = 0.25; w[BASE_FEATURE_DIM + 396] = 0.50  # Protected passer
    w[397] = 0.15; w[BASE_FEATURE_DIM + 397] = 0.20  # Pawn phalanx
    
    # Density interactions
    w[398] = 0.20; w[BASE_FEATURE_DIM + 398] = 0.10  # Knight in closed
    w[399] = 0.20; w[BASE_FEATURE_DIM + 399] = 0.30  # Rook in open
    
    return torch.from_numpy(w)
