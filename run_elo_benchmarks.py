"""
run_elo_benchmarks.py
---------------------
Benchmarks the trained TD-Leaf model against known Elo baselines:
  1. GreedyBot (Anchor Elo: ~450)
  2. HeuristicBot (Gen 0 Minimax, Anchor Elo: ~1050)
  3. Historical Game 1000 Checkpoint (Early Stage)
Calculates exact win rates, draw rates, and estimated Elo ratings.
"""

import os
import sys
import numpy as np
import torch
import chess

import feature_extractor
import search_engine
import evaluate
from evaluate import MinimaxPlayer, GreedyPlayer, RandomPlayer, play_match, run_tournament

def main():
    print("==================================================")
    print(" Bumbot TD-Leaf Official Elo Benchmarking Suite   ")
    print("==================================================")

    ckpt_path = "/opt/bumbot/checkpoints/weights_latest.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/weights_latest.pt"
        
    print(f"Loading latest trained model from: {ckpt_path}")
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = data["weights"] if "weights" in data else data
    if isinstance(weights, dict) and "linear.weight" in weights:
        weights = weights["linear.weight"].squeeze(0)
    game_num = data.get("game", "Latest")

    trained_player = MinimaxPlayer(name=f"Bumbot_Game{game_num}_D2", weights=weights, depth=2)
    print(f"Loaded: {trained_player.name} (Trained on {game_num} games)")

    # 1. Benchmark vs GreedyBot (Anchor Elo: 450)
    print("\n>>> BENCHMARK 1: vs GreedyBot (1-ply capture bot, Anchor ~450 Elo)")
    greedy = GreedyPlayer(name="GreedyBot")
    res_greedy = run_tournament(trained_player, greedy, num_games=10, opp_elo=450.0)

    # 2. Benchmark vs HeuristicBot (Gen 0 Heuristic Baseline, Anchor ~1050 Elo)
    print("\n>>> BENCHMARK 2: vs HeuristicBot (Gen 0 Baseline, Anchor ~1050 Elo)")
    h_weights = feature_extractor.get_default_heuristic_weights()
    heuristic = MinimaxPlayer(name="Heuristic_Gen0_D2", weights=h_weights, depth=2)
    res_heuristic = run_tournament(trained_player, heuristic, num_games=10, opp_elo=1050.0)

    # 3. Print Final Report
    print("\n" + "=" * 60)
    print("             FINAL ELO BENCHMARK REPORT             ")
    print("=" * 60)
    print(f"Model Under Test : {trained_player.name}")
    print(f"Games Trained    : {game_num} games")
    print("-" * 60)
    print(f"vs GreedyBot (450 Elo)   : {res_greedy['p1_wins']}W - {res_greedy['p2_wins']}L - {res_greedy['draws']}D | Score: {res_greedy['score_pct']:.1f}% | Est. Elo: {res_greedy['est_elo']:.0f}")
    print(f"vs Heuristic (1050 Elo)  : {res_heuristic['p1_wins']}W - {res_heuristic['p2_wins']}L - {res_heuristic['draws']}D | Score: {res_heuristic['score_pct']:.1f}% | Est. Elo: {res_heuristic['est_elo']:.0f}")
    print("-" * 60)
    composite_elo = (res_greedy['est_elo'] + res_heuristic['est_elo']) / 2.0
    print(f"Composite Estimated Playing Strength: ~{composite_elo:.0f} Elo")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
