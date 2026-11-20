"""
plot_results.py
---------------
Generates presentation-ready training curves and weight visualizations
from TD-Leaf training logs and checkpoint files. Supports both V2 (808-dim
tapered GLM) and V1 (651-dim classical) architectures.

Outputs:
  - plots/training_curves.png (or training_curves_v2.png): TD Loss, Game Length, and Weight Norm evolution.
  - plots/weight_interpretation.png (or weight_interpretation_v2.png): Comparison of learned material & positional weights.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

try:
    import feature_extractor
except ImportError:
    feature_extractor = None


def plot_training_telemetry(log_csv: str = "training_log_v2.csv", save_path: str = "plots/training_curves.png"):
    """Plots TD-Leaf loss, game length progression, and parameter norm."""
    if not os.path.exists(log_csv):
        print(f"[Warning] Log file not found: {log_csv}")
        return

    try:
        df = pd.read_csv(log_csv)
    except Exception as e:
        print(f"[Warning] Failed to read log file: {e}")
        return

    if df.empty:
        print("[Warning] Log file is empty.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    is_v2 = ("v2" in log_csv) or ("v2" in save_path)
    loss_label = "Binary Cross-Entropy Loss" if is_v2 else "Mean Squared Error (V_t - Target)"
    loss_title = "TD-Leaf(λ) BCE Loss Convergence (V2)" if is_v2 else "TD-Leaf(λ) MSE Loss Convergence (V1)"

    # 1. Loss with Rolling Average
    axes[0].plot(df["game"], df["loss"], alpha=0.25, color="gray", label="Raw TD Loss")
    rolling_window = max(3, min(25, len(df) // 10))
    rolling_loss = df["loss"].rolling(window=rolling_window, min_periods=1).mean()
    axes[0].plot(df["game"], rolling_loss, color="#1f77b4", linewidth=2.5, label=f"Moving Avg (w={rolling_window})")
    axes[0].set_title(loss_title, fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Game Index", fontsize=11)
    axes[0].set_ylabel(loss_label, fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Game Length (Plies)
    axes[1].plot(df["game"], df["plies"], alpha=0.25, color="gray", label="Raw Plies")
    rolling_plies = df["plies"].rolling(window=rolling_window, min_periods=1).mean()
    axes[1].plot(df["game"], rolling_plies, color="#10b981", linewidth=2.5, label=f"Moving Avg (w={rolling_window})")
    axes[1].set_title("Game Duration (Soundness Indicator)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Game Index", fontsize=11)
    axes[1].set_ylabel("Total Plies", fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Weight Norm Evolution
    axes[2].plot(df["game"], df["weight_norm"], color="#ef4444", linewidth=2.5)
    axes[2].set_title("L2 Weight Norm (Parameter Stability)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Game Index", fontsize=11)
    axes[2].set_ylabel("||w||_2", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {save_path}")


def plot_weight_interpretation(
    checkpoint_path: str,
    save_path: str = "plots/weight_interpretation.png"
):
    """
    Visualizes the weights learned by TD-Leaf for material imbalances and strategic features.
    Auto-detects whether the checkpoint is V2 (808-dim) or V1 (651-dim).
    """
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Checkpoint not found: {checkpoint_path}")
        return

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = data["weights"].numpy() if isinstance(data["weights"], torch.Tensor) else np.array(data["weights"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    if len(weights) in (800, 808):
        dim = len(weights)
        is_800 = (dim == 800)
        base_d = 400 if is_800 else 404
        w_mg = weights[:base_d]
        w_eg = weights[base_d:]

        fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))

        # 1. Material Values: Middlegame vs Endgame
        mat_labels = ["Pawn", "Knight", "Bishop", "Rook", "Queen"]
        mg_mat = w_mg[0:5]
        eg_mat = w_eg[0:5]

        x = np.arange(len(mat_labels))
        width = 0.35
        axes[0].bar(x - width/2, mg_mat, width, label="Middlegame (MG)", color="#3b82f6", edgecolor="black")
        axes[0].bar(x + width/2, eg_mat, width, label="Endgame (EG)", color="#10b981", edgecolor="black")
        axes[0].set_title(f"Learned Material Values ({'800-dim V2.1' if is_800 else '808-dim V2'})", fontsize=13, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(mat_labels, fontweight="bold")
        axes[0].set_ylabel("Weight Value", fontsize=11)
        axes[0].legend(frameon=True)
        axes[0].grid(axis="y", linestyle="--", alpha=0.6)
        for i in range(len(mat_labels)):
            axes[0].text(x[i] - width/2, mg_mat[i] + 0.05, f"{mg_mat[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold")
            axes[0].text(x[i] + width/2, eg_mat[i] + 0.05, f"{eg_mat[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold")

        # 2. Key Strategic Features (MG vs EG)
        if is_800:
            strat_labels = ["Bishop Pair", "Rook Open", "Semi-Open", "Tempo", "King Shield", "Outpost", "Center Ctrl"]
            strat_indices = [379, 380, 381, 382, 386, 395, 385]
        else:
            strat_labels = ["Bishop Pair", "Rook Open", "Semi-Open", "Tempo", "King Shield", "Mobility", "Center Ctrl"]
            strat_indices = [395, 396, 397, 398, 402, 403, 401]

        mg_strat = [w_mg[idx] for idx in strat_indices]
        eg_strat = [w_eg[idx] for idx in strat_indices]

        x_s = np.arange(len(strat_labels))
        axes[1].bar(x_s - width/2, mg_strat, width, label="Middlegame", color="#6366f1", edgecolor="black")
        axes[1].bar(x_s + width/2, eg_strat, width, label="Endgame", color="#f59e0b", edgecolor="black")
        axes[1].set_title("Strategic & Positional Weights (MG vs EG)", fontsize=13, fontweight="bold")
        axes[1].set_xticks(x_s)
        axes[1].set_xticklabels(strat_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
        axes[1].set_ylabel("Weight Value", fontsize=11)
        axes[1].axhline(0, color="black", linewidth=1.0)
        axes[1].legend(frameon=True)
        axes[1].grid(axis="y", linestyle="--", alpha=0.6)

        # 3. Passed Pawn Progression by Rank (Ranks 2-7)
        passed_ranks = ["Rank 2", "Rank 3", "Rank 4", "Rank 5", "Rank 6", "Rank 7"]
        passed_start = 373 if is_800 else 389
        mg_passed = w_mg[passed_start:passed_start + 6]
        eg_passed = w_eg[passed_start:passed_start + 6]
        x_p = np.arange(len(passed_ranks))
        axes[2].bar(x_p - width/2, mg_passed, width, label="Middlegame", color="#8b5cf6", edgecolor="black")
        axes[2].bar(x_p + width/2, eg_passed, width, label="Endgame", color="#ec4899", edgecolor="black")
        axes[2].set_title("Passed Pawn Values by Rank (MG vs EG)", fontsize=13, fontweight="bold")
        axes[2].set_xticks(x_p)
        axes[2].set_xticklabels(passed_ranks, fontweight="bold")
        axes[2].set_ylabel("Bonus Value", fontsize=11)
        axes[2].legend(frameon=True)
        axes[2].grid(axis="y", linestyle="--", alpha=0.6)
        for i in range(len(passed_ranks)):
            axes[2].text(x_p[i] + width/2, eg_passed[i] + 0.08, f"{eg_passed[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Plot Saved] {save_path}")

    else:
        # V1: 651 features
        mat_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen"]
        mat_weights = weights[0:5]

        strat_names = ["Mobility", "Center Ctrl", "Doubled P.", "Isolated P.", "Passed P.", "King Shield"]
        strat_weights = weights[645:651]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(mat_names, mat_weights, color=["#3274d9"]*5, edgecolor="black")
        axes[0].set_title("Learned Material Values (Relative Scale)", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Weight Value", fontsize=11)
        axes[0].grid(axis="y", linestyle="--", alpha=0.6)
        for i, v in enumerate(mat_weights):
            axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")

        colors = ["#2ca02c" if w >= 0 else "#d62728" for w in strat_weights]
        axes[1].bar(strat_names, strat_weights, color=colors, edgecolor="black")
        axes[1].set_title("Learned Strategic & Positional Weights", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Weight Value", fontsize=11)
        axes[1].axhline(0, color="black", linewidth=1.0)
        axes[1].grid(axis="y", linestyle="--", alpha=0.6)
        for i, v in enumerate(strat_weights):
            y_pos = v + 0.01 if v >= 0 else v - 0.03
            axes[1].text(i, y_pos, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Plot Saved] {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot TD-Leaf Training Telemetry & Weights")
    parser.add_argument("--log", type=str, default="training_log_v2.csv", help="Path to training log CSV")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--out-dir", type=str, default="plots", help="Directory to save figures")
    args = parser.parse_args()

    plot_training_telemetry(log_csv=args.log, save_path=os.path.join(args.out_dir, "training_curves.png"))
    
    ckpt = args.checkpoint
    if ckpt is None:
        for search_dir in ["checkpoints_v2", "checkpoints"]:
            if os.path.exists(search_dir):
                latest = os.path.join(search_dir, "weights_latest.pt")
                if os.path.exists(latest):
                    ckpt = latest
                    break
                ckpts = sorted([os.path.join(search_dir, f) for f in os.listdir(search_dir) if f.endswith(".pt")])
                if ckpts:
                    ckpt = ckpts[-1]
                    break
            
    if ckpt:
        plot_weight_interpretation(checkpoint_path=ckpt, save_path=os.path.join(args.out_dir, "weight_interpretation.png"))
