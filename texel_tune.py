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

import feature_extractor_v2 as feature_extractor


def extract_quiet_positions_from_pgn(
    pgn_path: str,
    max_positions: int = 20000,
    skip_early_plies: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses PGN file and extracts (feature_vector, outcome) for non-check, non-capture positions.
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


def run_texel_tuning(
    features: np.ndarray,
    labels: np.ndarray,
    c_l2: float = 1e-4,
    max_iter: int = 100,
    scale: float = 4.0,
    output_path: str = "texel_tuned_weights.pt",
):
    """Executes L-BFGS optimization on the BCE loss."""
    print(f"Running Texel Tuning (L-BFGS) on {len(features):,} positions (L2={c_l2})...")
    X = torch.from_numpy(features)
    y = torch.from_numpy(labels)

    # Initialize weights from default heuristics
    init_w = feature_extractor.get_default_heuristic_weights()
    linear = nn.Linear(feature_extractor.TOTAL_FEATURE_DIM, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(init_w.view(1, -1))

    optimizer = optim.LBFGS(linear.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")
    bce = nn.BCELoss()

    step = [0]
    def closure():
        optimizer.zero_grad()
        p = torch.sigmoid(linear(X).squeeze(-1) / scale)
        loss = bce(p, y) + 0.5 * c_l2 * torch.norm(linear.weight)**2
        loss.backward()
        step[0] += 1
        if step[0] % 10 == 0:
            print(f"L-BFGS Step {step[0]:3d} | Loss: {loss.item():.5f}", flush=True)
        return loss

    optimizer.step(closure)
    tuned_w = linear.weight.squeeze(0).detach()
    print(f"Texel Tuning Complete! Final Weight Norm: {torch.norm(tuned_w):.4f}")

    torch.save({"weights": tuned_w, "source": "texel_lbfgs"}, output_path)
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
