"""
td_leaf_trainer.py
------------------
Upgraded TD-Leaf(lambda) Reinforcement Learning Engine for Bumbot V2 (Phase 2):
  - Logistic Regression (GLM) Value Function:
      p(s) = sigmoid( w^T x(s) / scale ) in (0, 1)
      Trained via Binary Cross-Entropy (BCE) Loss against bootstrapped targets in [0, 1]
      Eliminates vanishing gradient (1 - V^2) from tanh+MSE
  - Fixed Opening Exploration: Softmax temperature on pre-squashed linear scores
  - Synchronous On-Policy Batches: Eliminates off-policy replay buffer drift
  - Discount Factor: gamma = 0.99 (incentivizes active conversion of wins over draws)
  - Tapered Projected Gradient Descent (PGD):
      Enforces domain constraints on Middlegame vs Endgame slices separately
      (King shield unconstrained in Endgame slice to encourage king centralization)
"""

import os
import sys
import time
import signal
import argparse
import random
import glob
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess

import feature_extractor
import search_engine

stop_requested = False


def _signal_handler(signum, frame):
    global stop_requested
    print("\n[Signal] Interrupt received. Finishing current batch and saving checkpoint...", flush=True)
    stop_requested = True


try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    pass


class ChessLogisticEvaluator(nn.Module):
    """
    Generalized Linear Model (GLM) with logistic link function:
        p(White Win | s) = sigmoid( (w^T x(s)) / scale )
    Maps 808 tapered features to win probability in (0, 1).
    """
    def __init__(self, feature_dim: int = feature_extractor.TOTAL_FEATURE_DIM, scale: float = 4.0):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.linear(features)
        return torch.sigmoid(raw / self.scale)

    def forward_score(self, features: torch.Tensor) -> torch.Tensor:
        p = self.forward(features)
        return 2.0 * p - 1.0

    def get_weights(self) -> torch.Tensor:
        return self.linear.weight.squeeze(0).detach()

    def set_weights(self, weights: torch.Tensor):
        with torch.no_grad():
            self.linear.weight.copy_(weights.view(1, -1))


def play_self_play_game(
    weights_white,
    weights_black=None,
    search_depth: int = 2,
    max_plies: int = 200,
    temperature_cutoff: int = 20,
    initial_temperature: float = 0.5,
) -> Tuple[np.ndarray, float, str, int]:
    if weights_black is None:
        weights_black = weights_white

    w_w_np = weights_white if isinstance(weights_white, np.ndarray) else weights_white.cpu().numpy()
    w_b_np = weights_black if isinstance(weights_black, np.ndarray) else weights_black.cpu().numpy()

    board = chess.Board()
    leaf_features_list: List[np.ndarray] = []
    ply = 0

    while not board.is_game_over() and ply < max_plies:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        curr_w = w_w_np if board.turn == chess.WHITE else w_b_np
        
        if ply < temperature_cutoff:
            temp = max(0.01, initial_temperature * (1.0 - ply / float(temperature_cutoff)))
        else:
            temp = 0.0

        best_move, score, leaf_board, _ = search_engine.get_best_move_sampled(
            board, curr_w, depth=search_depth, temperature=temp
        )

        if best_move is None or best_move not in board.legal_moves:
            best_move = random.choice(legal_moves)
            board.push(best_move)
            leaf_feat = feature_extractor.extract_features_numpy(board)
        else:
            leaf_feat = feature_extractor.extract_features_numpy(leaf_board)
            board.push(best_move)

        leaf_features_list.append(leaf_feat)
        ply += 1

    if board.is_checkmate():
        outcome = 0.0 if board.turn == chess.WHITE else 1.0
        reason = "checkmate"
    elif board.is_stalemate():
        outcome = 0.5
        reason = "stalemate"
    elif board.is_insufficient_material():
        outcome = 0.5
        reason = "insufficient_material"
    elif board.is_fifty_moves():
        outcome = 0.5
        reason = "fifty_moves"
    elif board.is_repetition(3):
        outcome = 0.5
        reason = "repetition"
    else:
        mat_diff = float(leaf_features_list[-1][0]) if leaf_features_list else 0.0
        if abs(mat_diff) >= 5.0:
            outcome = 1.0 if mat_diff > 0 else 0.0
            reason = "adjudicated_material"
        else:
            outcome = 0.5
            reason = "max_plies_draw"

    features_array = np.array(leaf_features_list, dtype=np.float32) if leaf_features_list else np.empty((0, feature_extractor.TOTAL_FEATURE_DIM), dtype=np.float32)
    return features_array, outcome, reason, ply


def _worker_game_runner(args_tuple):
    w_w, w_b, depth, max_plies, temp_cutoff, init_temp = args_tuple
    return play_self_play_game(
        weights_white=w_w,
        weights_black=w_b,
        search_depth=depth,
        max_plies=max_plies,
        temperature_cutoff=temp_cutoff,
        initial_temperature=init_temp,
    )


def compute_td_leaf_bce_loss(
    evaluator: ChessLogisticEvaluator,
    leaf_features: np.ndarray,
    terminal_outcome: float,
    lambda_val: float = 0.7,
    gamma: float = 0.99,
) -> torch.Tensor:
    if len(leaf_features) == 0:
        return torch.tensor(0.0, requires_grad=True)

    feat_batch = torch.from_numpy(leaf_features)
    T = feat_batch.shape[0]

    p_preds = evaluator(feat_batch).squeeze(-1)
    p_vals = p_preds.detach().cpu().numpy()

    # Center probabilities around 0.5 (draw baseline) before applying temporal discounting.
    # This prevents gamma < 1 from decaying towards 0.0 (Black Win) and preserves exact color antisymmetry.
    p_centered = p_vals - 0.5
    outcome_centered = terminal_outcome - 0.5

    deltas = np.zeros(T, dtype=np.float32)
    for t in range(T - 1):
        deltas[t] = gamma * p_centered[t + 1] - p_centered[t]
    deltas[T - 1] = outcome_centered - p_centered[T - 1]

    returns = np.zeros(T, dtype=np.float32)
    running = 0.0
    decay = gamma * lambda_val
    for t in reversed(range(T)):
        running = deltas[t] + decay * running
        returns[t] = running

    # Re-center back to [0, 1] probability space: 0.5 + p_centered + returns
    targets_np = np.clip(0.5 + p_centered + returns, 0.001, 0.999)
    targets = torch.tensor(targets_np, dtype=torch.float32, device=p_preds.device)

    loss = nn.functional.binary_cross_entropy(p_preds, targets)
    return loss


def apply_tapered_pgd_constraints(evaluator: ChessLogisticEvaluator):
    with torch.no_grad():
        w = evaluator.linear.weight
        D = feature_extractor.BASE_FEATURE_DIM

        # Material >= 0.1
        w[0, 0:5].clamp_(min=0.1)
        w[0, D:D+5].clamp_(min=0.1)

        # Passed pawns >= 0.01 in MG, >= 0.05 in EG
        w[0, 373:379].clamp_(min=0.01)
        w[0, D+373:D+379].clamp_(min=0.05)

        # Bishop pair >= 0.01
        w[0, 379].clamp_(min=0.01)
        w[0, D+379].clamp_(min=0.01)

        # Rook open/semi-open >= 0.01
        w[0, 380:382].clamp_(min=0.01)
        w[0, D+380:D+382].clamp_(min=0.01)

        # Doubled & Isolated pawns <= -0.01
        w[0, 383].clamp_(max=-0.01)
        w[0, 384].clamp_(max=-0.01)
        w[0, D+383].clamp_(max=-0.01)
        w[0, D+384].clamp_(max=-0.01)

        # Center control >= 0.01 in MG
        w[0, 385].clamp_(min=0.01)

        # King shield: >= 0.01 in MG, UNCONSTRAINED in EG
        w[0, 386].clamp_(min=0.01)

        # Safe mobility >= 0.005
        w[0, 387:391].clamp_(min=0.005)
        w[0, D+387:D+391].clamp_(min=0.005)

        # Endgame mop-up (EG only) >= 0.01
        w[0, D+391:D+394].clamp_(min=0.01)

        # Outposts, Protected Passers, Phalanxes >= 0.01
        w[0, 395:398].clamp_(min=0.01)
        w[0, D+395:D+398].clamp_(min=0.01)

        # Density interactions >= 0.01
        w[0, 398:400].clamp_(min=0.01)
        w[0, D+398:D+400].clamp_(min=0.01)


def train_self_play(
    num_games: int = 100,
    search_depth: int = 2,
    lr: float = 0.001,
    lambda_val: float = 0.7,
    gamma: float = 0.99,
    l2_reg: float = 1e-4,
    checkpoint_interval: int = 40,
    save_dir: str = "checkpoints_v2",
    log_file: str = "training_log_v2.csv",
    workers: int = 4,
    resume: bool = True,
    batch_size: int = 16,
    initial_temperature: float = 0.5,
    temperature_cutoff: int = 20,
    league_prob: float = 0.20,
    heuristic_prob: float = 0.10,
    max_plies: int = 200,
) -> ChessLogisticEvaluator:
    os.makedirs(save_dir, exist_ok=True)
    evaluator = ChessLogisticEvaluator()
    # Selective L2 regularization: Adam weight_decay is 0; L2 applied via reg_mask
    optimizer = optim.Adam(evaluator.parameters(), lr=lr, weight_decay=0.0)

    latest_ckpt_path = os.path.join(save_dir, "weights_latest.pt")
    start_game = 0

    if resume and os.path.exists(latest_ckpt_path):
        print(f"[Resume] Loading checkpoint: {latest_ckpt_path}")
        d = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        ckpt_dim = len(d["weights"])
        if ckpt_dim == feature_extractor.TOTAL_FEATURE_DIM:
            evaluator.set_weights(d["weights"])
            if "optimizer_state" in d:
                optimizer.load_state_dict(d["optimizer_state"])
            start_game = d.get("game", 0)
            print(f"[Resume] Resuming from game {start_game}!")
        else:
            print(f"[Notice] Checkpoint dimension ({ckpt_dim}) does not match upgraded model dimension ({feature_extractor.TOTAL_FEATURE_DIM}).")
            print("[Init] Starting fresh training run with upgraded 800-dim orthogonal feature representation!")
            init_w = feature_extractor.get_default_heuristic_weights()
            evaluator.set_weights(init_w)
    else:
        init_w = feature_extractor.get_default_heuristic_weights()
        evaluator.set_weights(init_w)
        print("[Init] Starting fresh training from default heuristic weights.")

    log_mode = "a" if (resume and os.path.exists(log_file)) else "w"
    with open(log_file, log_mode) as f:
        if log_mode == "w":
            f.write("game,plies,outcome,reason,loss,weight_norm,time_sec\n")

    run_forever = (num_games <= 0)
    target_games = (start_game + num_games) if not run_forever else float("inf")
    games_completed = start_game

    print("=" * 70)
    print(" BUMBOT V2 ON-POLICY TD-LEAF TRAINING (LOGISTIC REGRESSION GLM)")
    print("=" * 70)
    print(f"Target Games     : {'Continuous' if run_forever else target_games}")
    print(f"Search Depth     : {search_depth}")
    print(f"Workers          : {workers} | Batch Size: {batch_size}")
    print(f"Loss Function    : Binary Cross-Entropy (BCE) with gamma={gamma}, lambda={lambda_val}")
    print(f"Feature Dimension: {feature_extractor.TOTAL_FEATURE_DIM} (Tapered MG + EG)")
    print(f"Checkpoints Dir  : {save_dir}")
    print(f"Log File         : {log_file}")
    print("-" * 70)

    pool = mp.Pool(processes=workers) if workers > 1 else None
    h_weights_np = feature_extractor.get_default_heuristic_weights().numpy()

    try:
        while (games_completed < target_games or run_forever) and not stop_requested:
            current_w_np = evaluator.get_weights().numpy()
            t0 = time.time()

            worker_tasks = []
            matchup_tags = []

            for _ in range(batch_size):
                roll = random.random()
                if roll < heuristic_prob:
                    if random.random() < 0.5:
                        w_w, w_b, tag = current_w_np, h_weights_np, "self_vs_heuristic"
                    else:
                        w_w, w_b, tag = h_weights_np, current_w_np, "heuristic_vs_self"
                elif roll < (heuristic_prob + league_prob):
                    past_ckpts = sorted(glob.glob(os.path.join(save_dir, "weights_game_*.pt")))
                    if past_ckpts:
                        chosen = random.choice(past_ckpts)
                        try:
                            d_past = torch.load(chosen, map_location="cpu", weights_only=False)
                            past_w = d_past["weights"].numpy() if isinstance(d_past["weights"], torch.Tensor) else np.array(d_past["weights"])
                            t_name = os.path.basename(chosen).replace(".pt", "")
                            if random.random() < 0.5:
                                w_w, w_b, tag = current_w_np, past_w, f"self_vs_{t_name}"
                            else:
                                w_w, w_b, tag = past_w, current_w_np, f"{t_name}_vs_self"
                        except Exception:
                            w_w, w_b, tag = current_w_np, current_w_np, "self_play"
                    else:
                        w_w, w_b, tag = current_w_np, current_w_np, "self_play"
                else:
                    w_w, w_b, tag = current_w_np, current_w_np, "self_play"

                worker_tasks.append((w_w, w_b, search_depth, max_plies, temperature_cutoff, initial_temperature))
                matchup_tags.append(tag)

            if pool is not None:
                batch_results = pool.map(_worker_game_runner, worker_tasks)
            else:
                batch_results = [_worker_game_runner(t) for t in worker_tasks]

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)

            for feat_arr, outcome, reason, plies in batch_results:
                loss = compute_td_leaf_bce_loss(evaluator, feat_arr, outcome, lambda_val=lambda_val, gamma=gamma)
                total_loss = total_loss + loss

            mean_loss = total_loss / len(batch_results)

            # Selective L2 Regularization (Hierarchical Shrinkage)
            reg_mask = torch.from_numpy(feature_extractor.get_regularization_mask()).to(evaluator.linear.weight.device)
            l2_loss = 0.5 * l2_reg * torch.sum((evaluator.linear.weight * reg_mask) ** 2)
            total_batch_loss = mean_loss + l2_loss

            total_batch_loss.backward()
            nn.utils.clip_grad_norm_(evaluator.parameters(), max_norm=5.0)
            optimizer.step()

            apply_tapered_pgd_constraints(evaluator)

            games_completed += len(batch_results)
            elapsed = time.time() - t0
            w_norm = torch.norm(evaluator.get_weights()).item()

            with open(log_file, "a") as f:
                for idx, (feat_arr, outcome, reason, plies) in enumerate(batch_results):
                    m_tag = matchup_tags[idx]
                    f.write(
                        f"{games_completed},{plies},{outcome:.2f},{reason}_{m_tag},"
                        f"{mean_loss.item():.6f},{w_norm:.4f},{elapsed/len(batch_results):.2f}\n"
                    )
                f.flush()

            last_res = batch_results[-1]
            last_tag = matchup_tags[-1]
            print(
                f"Game {games_completed:5d} | Plies: {last_res[3]:3d} | Match: {last_tag:18s} | "
                f"Outcome: {last_res[1]:3.1f} ({last_res[2]}) | BCE Loss: {mean_loss.item():.5f} | "
                f"W_Norm: {w_norm:.3f} | Batch: {elapsed:.2f}s",
                flush=True
            )

            if games_completed % checkpoint_interval == 0 or games_completed >= target_games or stop_requested:
                ckpt_data = {
                    "game": games_completed,
                    "weights": evaluator.get_weights(),
                    "optimizer_state": optimizer.state_dict(),
                    "feature_dim": feature_extractor.TOTAL_FEATURE_DIM,
                    "model_type": "tapered_logistic_glm",
                }
                c_path = os.path.join(save_dir, f"weights_game_{games_completed:04d}.pt")
                torch.save(ckpt_data, c_path)
                torch.save(ckpt_data, latest_ckpt_path)
                print(f"[Checkpoint] Saved generation {games_completed} to {c_path}", flush=True)

    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if games_completed > start_game:
            ckpt_data = {
                "game": games_completed,
                "weights": evaluator.get_weights(),
                "optimizer_state": optimizer.state_dict(),
                "feature_dim": feature_extractor.TOTAL_FEATURE_DIM,
                "model_type": "tapered_logistic_glm",
            }
            torch.save(ckpt_data, latest_ckpt_path)

    return evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bumbot V2 via On-Policy Logistic TD-Leaf")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-val", type=float, default=0.7)
    parser.add_argument("--l2-reg", type=float, default=1e-4, help="Selective L2 regularization strength")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=40)
    parser.add_argument("--save-dir", type=str, default="checkpoints_v2")
    parser.add_argument("--log-file", type=str, default="training_log_v2.csv")
    parser.add_argument("--max-plies", type=int, default=200, help="Maximum plies before draw adjudication")
    args = parser.parse_args()

    train_self_play(
        num_games=0 if args.forever else args.games,
        search_depth=args.depth,
        lr=args.lr,
        lambda_val=args.lambda_val,
        gamma=args.gamma,
        l2_reg=args.l2_reg,
        checkpoint_interval=args.checkpoint_interval,
        save_dir=args.save_dir,
        log_file=args.log_file,
        workers=args.workers,
        batch_size=args.batch_size,
        max_plies=args.max_plies,
    )
