"""
td_leaf_trainer.py
------------------
Owner: Partner B (ML & Self-Play)

Implements the TD-Leaf(lambda) reinforcement learning algorithm for the chess bot
using PyTorch, based on Baxter et al. (1998) "Learning to Play Chess Using Temporal Differences".

Key Concepts:
  - Evaluation Function: V(s) = tanh( (w . f(s)) / scale )
  - Minimax Leaf Selection: For move t, search to depth d yields leaf node s_leaf(t).
  - TD-Leaf Error:
        delta_t = V(s_leaf(t+1)) - V(s_leaf(t))     for t < N-1
        delta_{N-1} = z - V(s_leaf(N-1))            where z in {+1 (Win), 0 (Draw), -1 (Loss)}
  - TD(lambda) Target & Loss:
        Target Y_t = V_t + sum_{j=t}^{N-1} (lambda)^(j-t) * delta_j
        Loss = 0.5 * sum_t (V_t - Y_t.detach())^2
  - Self-Play Optimization: Runs games, backpropagates gradients, and saves checkpoints.
"""

import os
import sys
import time
import signal
import argparse
import random
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess

import feature_extractor
import search_engine

# Global flag for graceful shutdown
stop_requested = False


def _signal_handler(signum, frame):
    global stop_requested
    print("\n[Signal] Interrupt received. Finishing current batch and saving latest checkpoint...", flush=True)
    stop_requested = True


# Register signal handlers for graceful exit
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    pass


def _worker_self_play_game(args_tuple):
    """Worker function for multiprocessing pool using pure NumPy representation."""
    weights_np, depth, max_plies, random_opening_plies, eps = args_tuple
    return play_self_play_game(
        weights=weights_np,
        search_depth=depth,
        max_plies=max_plies,
        random_opening_plies=random_opening_plies,
        exploration_eps=eps,
    )


class ChessLinearEvaluator(nn.Module):
    """
    Classical linear evaluation function with hyperbolic tangent squashing:
        V(s) = tanh( w^T f(s) / scale )
    Maps board states to [-1.0, +1.0] where +1.0 = White win, -1.0 = Black win.
    """
    def __init__(self, feature_dim: int = feature_extractor.TOTAL_FEATURE_DIM, scale: float = 4.0):
        super().__init__()
        self.scale = scale
        self.linear = nn.Linear(feature_dim, 1, bias=False)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Computes squashed board value V(s) in (-1, +1)."""
        raw = self.linear(features)
        return torch.tanh(raw / self.scale)

    def get_weights(self) -> torch.Tensor:
        """Returns the weight vector as a 1D tensor."""
        return self.linear.weight.squeeze(0).detach()

    def set_weights(self, weights: torch.Tensor):
        """Sets the internal linear weights."""
        with torch.no_grad():
            self.linear.weight.copy_(weights.view(1, -1))


def play_self_play_game(
    weights,
    search_depth: int = 2,
    max_plies: int = 120,
    random_opening_plies: int = 2,
    exploration_eps: float = 0.05,
) -> Tuple[np.ndarray, float, str, int]:
    """
    Simulates a single self-play game using the search engine.
    Uses pure NumPy for the feature array representation for clean IPC across processes.
    
    Returns:
        leaf_features: 2D NumPy array of shape (plies, feature_dim)
        terminal_outcome: +1.0 (White win), 0.0 (Draw), -1.0 (Black win)
        termination_reason: Description of how the game ended
        total_plies: Number of plies played
    """
    if isinstance(weights, torch.Tensor):
        weights_np = weights.cpu().numpy()
    elif isinstance(weights, ChessLinearEvaluator):
        weights_np = weights.get_weights().cpu().numpy()
    else:
        weights_np = np.asarray(weights, dtype=np.float32)

    board = chess.Board()
    leaf_features_list: List[np.ndarray] = []
    
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
            
        # 1. Opening stochasticity / epsilon exploration
        if ply < random_opening_plies or random.random() < exploration_eps:
            move = random.choice(legal_moves)
            board.push(move)
            leaf_feat = feature_extractor.extract_features_numpy(board)
            leaf_features_list.append(leaf_feat)
        else:
            # Minimax Alpha-Beta search using numpy weights
            best_move, score, leaf_board, _ = search_engine.get_best_move(
                board, weights_np, depth=search_depth
            )
            if best_move is None:
                best_move = random.choice(legal_moves)
                board.push(best_move)
                leaf_feat = feature_extractor.extract_features_numpy(board)
            else:
                leaf_feat = feature_extractor.extract_features_numpy(leaf_board)
                board.push(best_move)
                
            leaf_features_list.append(leaf_feat)
            
        ply += 1

    # Determine game outcome from White's perspective
    if board.is_checkmate():
        outcome = -1.0 if board.turn == chess.WHITE else 1.0
        reason = "checkmate"
    elif board.is_stalemate():
        outcome = 0.0
        reason = "stalemate"
    elif board.is_insufficient_material():
        outcome = 0.0
        reason = "insufficient_material"
    elif board.is_fifty_moves():
        outcome = 0.0
        reason = "fifty_moves"
    elif board.is_repetition(3):
        outcome = 0.0
        reason = "threefold_repetition"
    else:
        reason = f"max_plies_{max_plies}"
        final_feats = feature_extractor.extract_features_numpy(board)
        final_raw = float(np.dot(weights_np, final_feats))
        final_val = float(np.tanh(final_raw / 4.0))
        outcome = max(-0.5, min(0.5, final_val))
        
    if leaf_features_list:
        leaf_features_arr = np.stack(leaf_features_list, axis=0)
    else:
        leaf_features_arr = np.zeros((0, feature_extractor.TOTAL_FEATURE_DIM), dtype=np.float32)

    return leaf_features_arr, outcome, reason, ply


def compute_td_leaf_loss(
    evaluator: ChessLinearEvaluator,
    leaf_features,
    terminal_outcome: float,
    lambda_val: float = 0.7,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Computes Baxter et al. TD-Leaf(lambda) loss using PyTorch autograd.
    Accepts 2D NumPy array or PyTorch Tensor (T, feature_dim).
    """
    if isinstance(leaf_features, np.ndarray):
        if leaf_features.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True)
        feat_batch = torch.from_numpy(leaf_features)
    elif isinstance(leaf_features, list):
        if len(leaf_features) == 0:
            return torch.tensor(0.0, requires_grad=True)
        feat_batch = torch.stack(leaf_features, dim=0)
    else:
        feat_batch = leaf_features
        if feat_batch.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True)

    T = feat_batch.shape[0]
    
    # Compute evaluations V(s_leaf)
    v_preds = evaluator(feat_batch).squeeze(-1)  # shape: (T,)

    # Compute TD errors delta_t in detached numpy/tensor space
    v_vals = v_preds.detach().cpu().numpy()
    deltas = np.zeros(T, dtype=np.float32)
    for t in range(T - 1):
        deltas[t] = gamma * v_vals[t + 1] - v_vals[t]
    # Terminal step
    deltas[T - 1] = terminal_outcome - v_vals[T - 1]

    # Compute accumulated discounted returns (TD-lambda targets)
    returns = np.zeros(T, dtype=np.float32)
    running_return = 0.0
    decay = gamma * lambda_val
    for t in reversed(range(T)):
        running_return = deltas[t] + decay * running_return
        returns[t] = running_return

    targets = torch.tensor(v_vals + returns, dtype=torch.float32, device=v_preds.device)

    # Mean squared error against detached targets
    loss = 0.5 * nn.functional.mse_loss(v_preds, targets, reduction="mean")
    return loss
    loss = 0.5 * nn.functional.mse_loss(v_preds, targets, reduction="mean")
    return loss


def train_self_play(
    num_games: int = 100,
    search_depth: int = 2,
    lr: float = 0.001,
    lambda_val: float = 0.7,
    checkpoint_interval: int = 25,
    save_dir: str = "checkpoints",
    log_file: str = "training_log.csv",
    initial_weights: Optional[torch.Tensor] = None,
    workers: int = 1,
    resume: bool = True,
) -> ChessLinearEvaluator:
    """
    Executes the TD-Leaf self-play training loop with progress tracking,
    multiprocessing parallelization, continuous running capability, and checkpointing.
    """
    os.makedirs(save_dir, exist_ok=True)
    evaluator = ChessLinearEvaluator()
    optimizer = optim.Adam(evaluator.parameters(), lr=lr, weight_decay=1e-4)

    latest_ckpt_path = os.path.join(save_dir, "weights_latest.pt")
    start_game = 0

    if resume and os.path.exists(latest_ckpt_path):
        print(f"[Resume] Loading checkpoint from: {latest_ckpt_path}")
        ckpt_data = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        evaluator.set_weights(ckpt_data["weights"])
        if "optimizer_state" in ckpt_data:
            optimizer.load_state_dict(ckpt_data["optimizer_state"])
        start_game = ckpt_data.get("game", 0)
        print(f"[Resume] Resuming training from game {start_game}!")
    elif initial_weights is not None:
        evaluator.set_weights(initial_weights)
    else:
        heuristic_w = feature_extractor.get_default_heuristic_weights()
        evaluator.set_weights(heuristic_w)

    log_mode = "a" if (resume and os.path.exists(log_file)) else "w"
    with open(log_file, log_mode) as f:
        if log_mode == "w":
            f.write("game,plies,outcome,reason,loss,weight_norm,time_sec\n")

    run_forever = (num_games <= 0)
    target_games = (start_game + num_games) if not run_forever else float("inf")
    games_completed = start_game

    print(f"=== Starting TD-Leaf Self-Play Training ===")
    target_str = "Forever (Continuous)" if run_forever else f"{target_games} total"
    print(f"Target: {target_str} | Search Depth: {search_depth} | LR: {lr} | Lambda: {lambda_val} | Workers: {workers}")
    print(f"Checkpoint Interval: Every {checkpoint_interval} games | Save Dir: {save_dir}")
    print(f"Initial Weight Norm: {torch.norm(evaluator.get_weights()).item():.4f}")

    def save_checkpoint(game_num: int):
        ckpt_payload = {
            "game": game_num,
            "weights": evaluator.get_weights(),
            "model_state": evaluator.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        v_path = os.path.join(save_dir, f"weights_game_{game_num:04d}.pt")
        l_path = os.path.join(save_dir, "weights_latest.pt")
        torch.save(ckpt_payload, v_path)
        torch.save(ckpt_payload, l_path)
        print(f"  [Checkpoint saved: game {game_num} -> {v_path}]", flush=True)

    start_total_time = time.time()
    pool = mp.Pool(processes=workers) if workers > 1 else None

    try:
        while not stop_requested and (run_forever or games_completed < target_games):
            remaining = (target_games - games_completed) if not run_forever else workers
            batch_size = min(workers if workers > 1 else 1, int(remaining))
            if batch_size <= 0:
                break

            t0 = time.time()
            current_w = evaluator.get_weights().cpu().numpy()
            if pool is not None:
                worker_args = [
                    (current_w, search_depth, 120, 2, 0.05) for _ in range(batch_size)
                ]
                batch_results = pool.map(_worker_self_play_game, worker_args)
            else:
                batch_results = [
                    play_self_play_game(current_w, search_depth=search_depth)
                ]

            optimizer.zero_grad()
            total_batch_loss = torch.tensor(0.0)
            
            for leaf_feats, outcome, reason, plies in batch_results:
                games_completed += 1
                loss = compute_td_leaf_loss(evaluator, leaf_feats, outcome, lambda_val=lambda_val)
                total_batch_loss = total_batch_loss + loss

            mean_loss = total_batch_loss / len(batch_results)
            mean_loss.backward()
            nn.utils.clip_grad_norm_(evaluator.parameters(), max_norm=5.0)
            optimizer.step()

            elapsed = time.time() - t0
            w_norm = torch.norm(evaluator.get_weights()).item()

            with open(log_file, "a") as f:
                for leaf_feats, outcome, reason, plies in batch_results:
                    f.write(
                        f"{games_completed},{plies},{outcome:.2f},{reason},"
                        f"{mean_loss.item():.6f},{w_norm:.4f},{elapsed/len(batch_results):.2f}\n"
                    )
                f.flush()

            if games_completed % max(1, (workers * 2)) == 0 or games_completed == target_games or stop_requested:
                last_res = batch_results[-1]
                print(
                    f"Game {games_completed:5d} | Plies: {last_res[3]:3d} | "
                    f"Outcome: {last_res[1]:+4.1f} ({last_res[2]}) | Loss: {mean_loss.item():.5f} | "
                    f"W_Norm: {w_norm:.3f} | Batch Time: {elapsed:.2f}s",
                    flush=True
                )

            if games_completed % checkpoint_interval == 0 or games_completed == target_games or stop_requested:
                save_checkpoint(games_completed)

    finally:
        if pool is not None:
            pool.close()
            pool.join()
        # Ensure latest state is persisted upon exit
        if games_completed > start_game:
            save_checkpoint(games_completed)

    print(f"Training loop exited. Total elapsed: {(time.time() - start_total_time):.1f}s")
    return evaluator
    return evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classical Chess Bot via TD-Leaf(lambda)")
    parser.add_argument("--games", type=int, default=100, help="Number of self-play games (set to 0 for infinite training)")
    parser.add_argument("--forever", action="store_true", help="Run self-play continuously until interrupted")
    parser.add_argument("--depth", type=int, default=2, help="Minimax search depth")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lambda-val", type=float, default=0.7, help="TD trace decay lambda")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Save checkpoint every N games")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-file", type=str, default="training_log.csv", help="CSV log filename")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint; start fresh")
    
    args = parser.parse_args()
    total_games = 0 if args.forever else args.games

    train_self_play(
        num_games=total_games,
        search_depth=args.depth,
        lr=args.lr,
        lambda_val=args.lambda_val,
        checkpoint_interval=args.checkpoint_interval,
        save_dir=args.save_dir,
        log_file=args.log_file,
        workers=args.workers,
        resume=(not args.no_resume),
    )
