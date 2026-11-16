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


def get_temperature_for_ply(ply: int, initial_temp: float = 1.0, cutoff_ply: int = 20) -> float:
    """
    AlphaZero Softmax Temperature Schedule:
    - Plies 0 to 10: High exploration (T ~ 1.0 -> 0.5) to vary opening books.
    - Plies 10 to 20: Gentle tapering (T ~ 0.5 -> 0.05).
    - Plies > 20: Deterministic greedy exploitation (T = 0.0) for tactical soundness.
    """
    if ply >= cutoff_ply:
        return 0.0
    fraction = ply / float(cutoff_ply)
    return max(0.0, initial_temp * (1.0 - fraction))


def _worker_self_play_game(args_tuple):
    """Worker function for multiprocessing pool using pure NumPy representation."""
    w_white, w_black, depth, max_plies, temp_cutoff, init_temp = args_tuple
    return play_self_play_game(
        weights_white=w_white,
        weights_black=w_black,
        search_depth=depth,
        max_plies=max_plies,
        temperature_cutoff=temp_cutoff,
        initial_temperature=init_temp,
    )


class ExperienceReplayBuffer:
    """
    AlphaZero-style FIFO Experience Replay Buffer for TD-Leaf training.
    
    Stores game trajectories (leaf_features, outcome) to decorrelate
    consecutive parameter updates and stabilize gradient variance.
    """
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, float, str, int]] = []

    def push(self, trajectory: Tuple[np.ndarray, float, str, int]):
        """Adds a completed game trajectory to the buffer."""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # FIFO eviction
        self.buffer.append(trajectory)

    def sample_training_batch(
        self, batch_size: int, recent_fraction: float = 0.5
    ) -> List[Tuple[np.ndarray, float, str, int]]:
        """
        Samples a balanced training batch combining both the most recent
        games and uniform random historical games from the buffer.
        """
        if not self.buffer:
            return []
        
        n_recent = min(len(self.buffer), max(1, int(batch_size * recent_fraction)))
        n_older = max(0, min(batch_size - n_recent, len(self.buffer) - n_recent))
        
        recent_items = self.buffer[-n_recent:]
        if n_older > 0:
            older_pool = self.buffer[:-n_recent]
            older_samples = random.sample(older_pool, min(n_older, len(older_pool)))
            return recent_items + older_samples
        return recent_items

    def __len__(self) -> int:
        return len(self.buffer)


def sample_league_matchup(
    current_w: np.ndarray,
    save_dir: str = "checkpoints",
    league_prob: float = 0.20,
    heuristic_prob: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    AlphaZero / AlphaStar League Opponent Sampling:
    - (1 - league_prob - heuristic_prob): Self-Play (current_w vs current_w)
    - league_prob: Plays against a random past checkpoint from checkpoints/
    - heuristic_prob: Plays against the baseline heuristic
    
    Returns:
        (w_white, w_black, matchup_type)
    """
    roll = random.random()
    h_weights = feature_extractor.get_default_heuristic_weights().numpy()

    if roll < heuristic_prob:
        if random.random() < 0.5:
            return current_w, h_weights, "self_vs_heuristic"
        else:
            return h_weights, current_w, "heuristic_vs_self"

    elif roll < (heuristic_prob + league_prob):
        ckpts = sorted(glob.glob(os.path.join(save_dir, "weights_game_*.pt")))
        if ckpts:
            chosen_ckpt = random.choice(ckpts)
            try:
                data = torch.load(chosen_ckpt, map_location="cpu", weights_only=False)
                past_w = data["weights"].numpy() if isinstance(data["weights"], torch.Tensor) else np.array(data["weights"])
                tag = os.path.basename(chosen_ckpt).replace(".pt", "")
                if random.random() < 0.5:
                    return current_w, past_w, f"self_vs_{tag}"
                else:
                    return past_w, current_w, f"{tag}_vs_self"
            except Exception:
                pass

    return current_w, current_w, "self_play"


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
    weights_white,
    weights_black=None,
    search_depth: int = 2,
    max_plies: int = 120,
    temperature_cutoff: int = 20,
    initial_temperature: float = 1.0,
) -> Tuple[np.ndarray, float, str, int]:
    """
    Simulates a match using AlphaZero-style Softmax Temperature exploration.
    Supports asymmetric league matchups (weights_white != weights_black).
    Uses pure NumPy representation for clean multiprocessing IPC across cores.
    
    Returns:
        leaf_features: 2D NumPy array of shape (plies, feature_dim)
        terminal_outcome: +1.0 (White win), 0.0 (Draw), -1.0 (Black win)
        termination_reason: Description of how the game ended
        total_plies: Number of plies played
    """
    if weights_black is None:
        weights_black = weights_white

    if isinstance(weights_white, torch.Tensor):
        w_white_np = weights_white.cpu().numpy()
    elif isinstance(weights_white, ChessLinearEvaluator):
        w_white_np = weights_white.get_weights().cpu().numpy()
    else:
        w_white_np = np.asarray(weights_white, dtype=np.float32)

    if isinstance(weights_black, torch.Tensor):
        w_black_np = weights_black.cpu().numpy()
    elif isinstance(weights_black, ChessLinearEvaluator):
        w_black_np = weights_black.get_weights().cpu().numpy()
    else:
        w_black_np = np.asarray(weights_black, dtype=np.float32)

    board = chess.Board()
    leaf_features_list: List[np.ndarray] = []

    ply = 0
    while not board.is_game_over() and ply < max_plies:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break

        curr_w = w_white_np if board.turn == chess.WHITE else w_black_np
        temp = get_temperature_for_ply(ply, initial_temp=initial_temperature, cutoff_ply=temperature_cutoff)

        # AlphaZero Softmax Temperature Root Sampling
        best_move, score, leaf_board, _ = search_engine.get_best_move_sampled(
            board, curr_w, depth=search_depth, temperature=temp
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
        final_raw = float(np.dot(w_white_np, final_feats))
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
    # AlphaZero Enhancements
    replay_capacity: int = 32,
    league_prob: float = 0.20,
    heuristic_prob: float = 0.10,
    temperature_cutoff: int = 20,
    initial_temperature: float = 1.0,
) -> ChessLinearEvaluator:
    """
    Executes the TD-Leaf self-play training loop with AlphaZero enhancements:
      1. Softmax Temperature Root Sampling (diverse opening exploration, zero blunders)
      2. League Opponent Sampling (robustness against past checkpoints & baseline)
      3. Experience Replay Buffer (decorrelated gradient descent & stable L2 norm)
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

    # Initialize AlphaZero Experience Replay Buffer
    replay_buffer = ExperienceReplayBuffer(capacity=replay_capacity)

    print(f"=== Starting AlphaZero-Enhanced TD-Leaf Training ===")
    target_str = "Forever (Continuous)" if run_forever else f"{target_games} total"
    print(f"Target: {target_str} | Search Depth: {search_depth} | LR: {lr} | Lambda: {lambda_val} | Workers: {workers}")
    print(f"AlphaZero Modules: Replay Buffer ({replay_capacity}) | League ({league_prob*100:.0f}%) | Heuristic ({heuristic_prob*100:.0f}%)")
    print(f"Temperature Schedule: T={initial_temperature} -> 0.0 over first {temperature_cutoff} plies")
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

        # Automatically update plots in plots/
        try:
            import plot_results
            plot_results.plot_training_telemetry(log_csv=log_file, save_path="plots/training_curves.png")
            plot_results.plot_weight_interpretation(checkpoint_path=l_path, save_path="plots/weight_interpretation.png")
        except Exception:
            pass

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

            # Prepare worker arguments with AlphaZero League Opponent Sampling
            worker_args = []
            matchups = []
            for _ in range(batch_size):
                w_w, w_b, match_tag = sample_league_matchup(
                    current_w,
                    save_dir=save_dir,
                    league_prob=league_prob,
                    heuristic_prob=heuristic_prob,
                )
                matchups.append(match_tag)
                worker_args.append((w_w, w_b, search_depth, 120, temperature_cutoff, initial_temperature))

            if pool is not None:
                batch_results = pool.map(_worker_self_play_game, worker_args)
            else:
                batch_results = [_worker_self_play_game(arg) for arg in worker_args]

            # Ingest new trajectories into the Experience Replay Buffer
            for res in batch_results:
                games_completed += 1
                replay_buffer.push(res)

            # Sample a decorrelated training batch from Replay Buffer
            train_batch = replay_buffer.sample_training_batch(batch_size=batch_size, recent_fraction=0.5)

            optimizer.zero_grad()
            total_batch_loss = torch.tensor(0.0)
            
            for leaf_feats, outcome, reason, plies in train_batch:
                loss = compute_td_leaf_loss(evaluator, leaf_feats, outcome, lambda_val=lambda_val)
                total_batch_loss = total_batch_loss + loss

            mean_loss = total_batch_loss / len(train_batch)
            mean_loss.backward()
            nn.utils.clip_grad_norm_(evaluator.parameters(), max_norm=5.0)
            optimizer.step()

            # Projected Gradient Descent: enforce domain sign constraints on strategic features
            with torch.no_grad():
                w_ref = evaluator.linear.weight
                w_ref[0, 0:5].clamp_(min=0.1)     # Material values must be positive
                w_ref[0, 645].clamp_(min=0.01)    # Mobility must be positive
                w_ref[0, 646].clamp_(min=0.01)    # Center control must be positive
                w_ref[0, 647].clamp_(max=-0.01)   # Doubled pawns penalty (negative)
                w_ref[0, 648].clamp_(max=-0.01)   # Isolated pawns penalty (negative)
                w_ref[0, 649].clamp_(min=0.01)    # Passed pawns bonus (positive)
                w_ref[0, 650].clamp_(min=0.01)    # King shield bonus (positive)

            elapsed = time.time() - t0
            w_norm = torch.norm(evaluator.get_weights()).item()

            with open(log_file, "a") as f:
                for idx, (leaf_feats, outcome, reason, plies) in enumerate(batch_results):
                    m_tag = matchups[idx]
                    f.write(
                        f"{games_completed},{plies},{outcome:.2f},{reason}_{m_tag},"
                        f"{mean_loss.item():.6f},{w_norm:.4f},{elapsed/len(batch_results):.2f}\n"
                    )
                f.flush()

            if games_completed % max(1, (workers * 2)) == 0 or games_completed == target_games or stop_requested:
                last_res = batch_results[-1]
                last_match = matchups[-1]
                print(
                    f"Game {games_completed:5d} | Plies: {last_res[3]:3d} | Match: {last_match:16s} | "
                    f"Outcome: {last_res[1]:+4.1f} ({last_res[2]}) | Loss: {mean_loss.item():.5f} | "
                    f"W_Norm: {w_norm:.3f} | Batch: {elapsed:.2f}s",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classical Chess Bot via AlphaZero-Enhanced TD-Leaf")
    parser.add_argument("--games", type=int, default=100, help="Number of self-play games (set to 0 for infinite training)")
    parser.add_argument("--forever", action="store_true", help="Run self-play continuously until interrupted")
    parser.add_argument("--depth", type=int, default=2, help="Minimax search depth")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lambda-val", type=float, default=0.7, help="TD trace decay lambda")
    parser.add_argument("--checkpoint-interval", type=int, default=40, help="Save checkpoint every N games")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-file", type=str, default="training_log.csv", help="CSV log filename")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint; start fresh")
    # AlphaZero arguments
    parser.add_argument("--league-ratio", type=float, default=0.20, help="Ratio of games against past checkpoints")
    parser.add_argument("--heuristic-ratio", type=float, default=0.10, help="Ratio of games against baseline heuristic")
    parser.add_argument("--buffer-size", type=int, default=32, help="Experience replay buffer size")
    parser.add_argument("--temp-cutoff", type=int, default=20, help="Plies to apply softmax temperature exploration")
    parser.add_argument("--init-temp", type=float, default=1.0, help="Initial softmax exploration temperature")
    
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
        replay_capacity=args.buffer_size,
        league_prob=args.league_ratio,
        heuristic_prob=args.heuristic_ratio,
        temperature_cutoff=args.temp_cutoff,
        initial_temperature=args.init_temp,
    )
