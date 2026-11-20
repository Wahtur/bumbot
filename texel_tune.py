"""
texel_tune.py
-------------
Offline Supervised Logistic Regression (Texel's Tuning Method) for Bumbot V2.
Fits linear weights w to quiet board positions extracted from games against
game outcome z in {1.0 (White win), 0.5 (Draw), 0.0 (Black win)} using
L-BFGS optimization with L2 regularization:
    min_w  -sum [ z_i log(sigmoid(w^T x_i / K)) + (1 - z_i) log(1 - sigmoid(w^T x_i / K)) ] + C * ||w||^2
"""

import os
import sys
import argparse
import random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn

import feature_extractor


def extract_quiet_positions_from_pgn(
    pgn_path: str,
    max_positions: int = 50000,
    skip_early_plies: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses PGN file and extracts (feature_vector, outcome) for quiet positions
    (non-check, non-tactical) to optimize Generalized Linear Model value function.
    """
    positions = []
    labels = []
    
    if not os.path.exists(pgn_path):
        print(f"[Error] PGN file not found: {pgn_path}")
        return np.empty((0, feature_extractor.TOTAL_FEATURE_DIM)), np.empty(0)

    print(f"Parsing PGN file: {pgn_path}...")
    with open(pgn_path, "r") as f:
        game_count = 0
        while len(positions) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            result_str = game.headers.get("Result", "*")
            if result_str == "1-0":
                outcome = 1.0
            elif result_str == "0-1":
                outcome = 0.0
            elif result_str == "1/2-1/2":
                outcome = 0.5
            else:
                continue

            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply >= skip_early_plies and not board.is_check():
                    feat = feature_extractor.extract_features_numpy(board)
                    positions.append(feat)
                    labels.append(outcome)
                    if len(positions) >= max_positions:
                        break

            game_count += 1

    print(f"Extracted {len(positions):,} quiet positions from {game_count} games.")
    return np.array(positions, dtype=np.float32), np.array(labels, dtype=np.float32)


def apply_texel_constraints(linear: nn.Linear):
    """Enforces domain-correct piece material anchors and non-negativity constraints."""
    with torch.no_grad():
        w = linear.weight
        D = feature_extractor.BASE_FEATURE_DIM

        # Anchored Material bounds (MG / EG)
        w[0, 0].clamp_(min=0.8, max=1.5)
        w[0, 1].clamp_(min=2.6, max=3.8)
        w[0, 2].clamp_(min=2.8, max=4.0)
        w[0, 3].clamp_(min=4.2, max=5.8)
        w[0, 4].clamp_(min=8.0, max=10.5)

        w[0, D+0].clamp_(min=0.9, max=1.8)
        w[0, D+1].clamp_(min=2.5, max=3.6)
        w[0, D+2].clamp_(min=2.8, max=4.0)
        w[0, D+3].clamp_(min=4.5, max=6.2)
        w[0, D+4].clamp_(min=8.0, max=10.5)

        # Passed pawns: positive monotonic progression
        w[0, 373:379].clamp_(min=0.01)
        w[0, D+373:D+379].clamp_(min=0.05)

        # Strategic features
        w[0, 379].clamp_(min=0.01)       # Bishop pair
        w[0, D+379].clamp_(min=0.01)
        w[0, 380:382].clamp_(min=0.01)   # Open / semi-open rooks
        w[0, D+380:D+382].clamp_(min=0.01)
        w[0, 383:385].clamp_(max=-0.01)  # Doubled / isolated pawns
        w[0, D+383:D+385].clamp_(max=-0.01)


def run_texel_tuning(
    features: np.ndarray,
    labels: np.ndarray,
    c_l2: float = 1e-3,
    max_iter: int = 100,
    scale: float = 4.0,
    normalize_pawn: bool = True,
    output_path: str = "texel_tuned_weights.pt",
):
    """
    Executes L-BFGS optimization on the BCE loss with Selective L2 Regularization
    and Projected Gradient Domain Constraints.
    """
    print(f"Running Constrained Texel Tuning (L-BFGS) on {len(features):,} positions (Selective L2={c_l2})...")
    X = torch.from_numpy(features)
    y = torch.from_numpy(labels)

    reg_mask = torch.from_numpy(feature_extractor.get_regularization_mask()).view(1, -1)

    # Initialize weights from default heuristics
    init_w = feature_extractor.get_default_heuristic_weights()
    linear = nn.Linear(feature_extractor.TOTAL_FEATURE_DIM, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(init_w.view(1, -1))

    # Outer projected iterations
    outer_steps = 5
    iters_per_step = max_iter // outer_steps

    bce = nn.BCELoss()

    for outer in range(outer_steps):
        optimizer = optim.LBFGS(linear.parameters(), lr=0.8, max_iter=iters_per_step, line_search_fn="strong_wolfe")
        
        def closure():
            optimizer.zero_grad()
            p = torch.sigmoid(linear(X).squeeze(-1) / scale)
            # Selective L2 regularization on positional features
            loss = bce(p, y) + 0.5 * c_l2 * torch.sum((linear.weight * reg_mask)**2)
            loss.backward()
            return loss

        loss_val = optimizer.step(closure)
        apply_texel_constraints(linear)
        print(f"Projected Step {(outer + 1) * iters_per_step:3d} | BCE Loss: {loss_val.item():.5f} | W_Norm: {torch.norm(linear.weight):.3f}", flush=True)

    tuned_w = linear.weight.squeeze(0).detach()

    # Normalize relative to learned middlegame pawn weight so 1.0 == 1 pawn
    if normalize_pawn:
        pawn_mg = float(tuned_w[0].item())
        if pawn_mg > 0.05:
            scale_factor = 1.0 / pawn_mg
            tuned_w = tuned_w * scale_factor
            print(f"[Normalization] Scaled weights by {scale_factor:.3f} (Pawn MG = 1.00)")

    print(f"Texel Tuning Complete! Final Weight Norm: {torch.norm(tuned_w):.4f}")

    torch.save({
        "weights": tuned_w,
        "source": "texel_lbfgs_projected",
        "feature_dim": feature_extractor.TOTAL_FEATURE_DIM,
        "model_type": "tapered_logistic_glm",
    }, output_path)
    print(f"Saved tuned weights to: {output_path}")
    return tuned_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texel Offline Logistic Regression Tuner")
    parser.add_argument("--pgn", type=str, required=True, help="Path to PGN game archive")
    parser.add_argument("--max-positions", type=int, default=20000)
    parser.add_argument("--output", type=str, default="texel_tuned_weights.pt")
    args = parser.parse_args()

    feats, labs = extract_quiet_positions_from_pgn(args.pgn, max_positions=args.max_positions)
    if len(feats) > 0:
        run_texel_tuning(feats, labs, output_path=args.output)
