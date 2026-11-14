"""
plot_results.py
---------------
Generates presentation-ready training curves and weight visualizations
from TD-Leaf training logs and checkpoint files.

Outputs:
  - plots/training_curves.png: TD Loss, Game Length, and Weight Norm evolution.
  - plots/weight_interpretation.png: Comparison of learned material & positional weights.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import feature_extractor


def plot_training_telemetry(log_csv: str = "training_log.csv", save_path: str = "plots/training_curves.png"):
    """Plots TD-Leaf loss, game length progression, and parameter norm."""
    if not os.path.exists(log_csv):
        print(f"[Warning] Log file not found: {log_csv}")
        return

    df = pd.read_csv(log_csv)
    if df.empty:
        print("[Warning] Log file is empty.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss with Rolling Average
    axes[0].plot(df["game"], df["loss"], alpha=0.3, color="gray", label="Raw TD Loss")
    rolling_window = max(3, min(20, len(df) // 5))
    rolling_loss = df["loss"].rolling(window=rolling_window, min_periods=1).mean()
    axes[0].plot(df["game"], rolling_loss, color="#1f77b4", linewidth=2.5, label=f"Moving Avg (w={rolling_window})")
    axes[0].set_title("TD-Leaf(λ) Loss Convergence", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Game Index", fontsize=12)
    axes[0].set_ylabel("Mean Squared Error (V_t - Target)", fontsize=12)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Game Length (Plies)
    axes[1].plot(df["game"], df["plies"], alpha=0.3, color="gray", label="Raw Plies")
    rolling_plies = df["plies"].rolling(window=rolling_window, min_periods=1).mean()
    axes[1].plot(df["game"], rolling_plies, color="#2ca02c", linewidth=2.5, label=f"Moving Avg (w={rolling_window})")
    axes[1].set_title("Game Duration (Soundness Indicator)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Game Index", fontsize=12)
    axes[1].set_ylabel("Total Plies", fontsize=12)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Weight Norm Evolution
    axes[2].plot(df["game"], df["weight_norm"], color="#d62728", linewidth=2.5)
    axes[2].set_title("L2 Weight Norm (Parameter Stability)", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Game Index", fontsize=12)
    axes[2].set_ylabel("||w||_2", fontsize=12)
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
    """
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Checkpoint not found: {checkpoint_path}")
        return

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = data["weights"].numpy() if isinstance(data["weights"], torch.Tensor) else np.array(data["weights"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = feature_extractor.get_feature_names()

    # Material features (first 5)
    mat_names = ["Pawn", "Knight", "Bishop", "Rook", "Queen"]
    mat_weights = weights[0:5]

    # Strategic features (last 6: mobility, center, doubled, isolated, passed, king shield)
    strat_names = ["Mobility", "Center Ctrl", "Doubled P.", "Isolated P.", "Passed P.", "King Shield"]
    strat_weights = weights[645:651]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Material weights bar chart
    axes[0].bar(mat_names, mat_weights, color=["#3274d9", "#3274d9", "#3274d9", "#3274d9", "#3274d9"], edgecolor="black")
    axes[0].set_title("Learned Material Values (Relative Scale)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Weight Value", fontsize=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(mat_weights):
        axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")

    # Strategic features bar chart (positive green, negative red)
    colors = ["#2ca02c" if w >= 0 else "#d62728" for w in strat_weights]
    axes[1].bar(strat_names, strat_weights, color=colors, edgecolor="black")
    axes[1].set_title("Learned Strategic & Positional Weights", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Weight Value", fontsize=12)
    axes[1].axhline(0, color="black", linewidth=1.0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(strat_weights):
        y_pos = v + 0.01 if v >= 0 else v - 0.03
        axes[1].text(i, y_pos, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot TD-Leaf Training Telemetry & Weights")
    parser.add_argument("--log", type=str, default="training_log.csv", help="Path to training log CSV")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--out-dir", type=str, default="plots", help="Directory to save figures")
    args = parser.parse_args()

    plot_training_telemetry(log_csv=args.log, save_path=os.path.join(args.out_dir, "training_curves.png"))
    
    # Auto-find latest checkpoint if not specified
    ckpt = args.checkpoint
    if ckpt is None and os.path.exists("checkpoints"):
        ckpts = sorted([os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pt")])
        if ckpts:
            ckpt = ckpts[-1]
            
    if ckpt:
        plot_weight_interpretation(checkpoint_path=ckpt, save_path=os.path.join(args.out_dir, "weight_interpretation.png"))
