"""
finish_d3_bench.py
------------------
Runs the remaining 5 games of the Depth 3 Elo tournament:
  - 1 game vs HeuristicBot_D3 (Black)
  - 4 games vs Stockfish_Level_2 (2 White, 2 Black)
And combines them with the 15 verified games to output the complete
official Depth 3 Benchmark Leaderboard and Calibrated Elo Rating.
"""

import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import chess
import chess.engine

sys.path.insert(0, "/opt/bumbot")
import feature_extractor
import search_engine
from evaluate import MinimaxPlayer, GreedyPlayer, play_match
from benchmark_vs_stockfish import StockfishPlayer
from benchmark_depth3_comprehensive import run_single_match, compute_fide_performance

# 15 Verified completed games from overnight run
PREV_GAMES = [
    # GreedyBot
    {"tier": "GreedyBot_Baseline", "anchor_elo": 450.0, "color": "Black", "score": 1.0, "outcome": "WIN", "reason": "checkmate", "plies": 8},
    {"tier": "GreedyBot_Baseline", "anchor_elo": 450.0, "color": "White", "score": 1.0, "outcome": "WIN", "reason": "checkmate", "plies": 17},
    {"tier": "GreedyBot_Baseline", "anchor_elo": 450.0, "color": "White", "score": 1.0, "outcome": "WIN", "reason": "checkmate", "plies": 55},
    {"tier": "GreedyBot_Baseline", "anchor_elo": 450.0, "color": "Black", "score": 1.0, "outcome": "WIN", "reason": "checkmate", "plies": 60},
    # Stockfish Level 0
    {"tier": "Stockfish_Level_0", "anchor_elo": 850.0, "color": "White", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    {"tier": "Stockfish_Level_0", "anchor_elo": 850.0, "color": "White", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    {"tier": "Stockfish_Level_0", "anchor_elo": 850.0, "color": "Black", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    {"tier": "Stockfish_Level_0", "anchor_elo": 850.0, "color": "Black", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    # Stockfish Level 1
    {"tier": "Stockfish_Level_1", "anchor_elo": 1050.0, "color": "White", "score": 0.0, "outcome": "LOSS", "reason": "checkmate", "plies": 44},
    {"tier": "Stockfish_Level_1", "anchor_elo": 1050.0, "color": "White", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    {"tier": "Stockfish_Level_1", "anchor_elo": 1050.0, "color": "Black", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    {"tier": "Stockfish_Level_1", "anchor_elo": 1050.0, "color": "Black", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    # HeuristicBot D3 (Games 1-3)
    {"tier": "HeuristicBot_D3", "anchor_elo": 1050.0, "color": "White", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    {"tier": "HeuristicBot_D3", "anchor_elo": 1050.0, "color": "Black", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
    {"tier": "HeuristicBot_D3", "anchor_elo": 1050.0, "color": "White", "score": 0.5, "outcome": "DRAW", "reason": "max_plies_draw", "plies": 70},
]

def main():
    print("=" * 70, flush=True)
    print(" COMPLETING REMAINING DEPTH 3 BENCHMARK GAMES (5 GAMES)", flush=True)
    print("=" * 70, flush=True)

    ckpt_path = "/opt/bumbot/checkpoints/weights_latest.pt"
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = data["weights"] if "weights" in data else data
    if isinstance(weights, dict) and "linear.weight" in weights:
        weights = weights["linear.weight"].squeeze(0)
    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(weights, dtype=np.float32)

    depth = 3
    max_plies = 70

    tasks = [
        # Game 16: HeuristicBot D3 game 4 (Black)
        (16, "HeuristicBot_D3", "heuristic", None, 1050.0, False, depth, max_plies, weights_np),
        # Games 17-20: Stockfish Level 2 (2 White, 2 Black)
        (17, "Stockfish_Level_2", "stockfish", 2, 1200.0, True,  depth, max_plies, weights_np),
        (18, "Stockfish_Level_2", "stockfish", 2, 1200.0, False, depth, max_plies, weights_np),
        (19, "Stockfish_Level_2", "stockfish", 2, 1200.0, True,  depth, max_plies, weights_np),
        (20, "Stockfish_Level_2", "stockfish", 2, 1200.0, False, depth, max_plies, weights_np),
    ]

    new_results = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_single_match, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            new_results.append(res)
            score_tag = f"+{res['score']:.1f}" if res['score'] > 0 else " 0.0"
            print(
                f"[{res['game_num']:02d}/20] "
                f"{res['tier']:<18} "
                f"[{res['color']:<5}]: "
                f"{res['outcome']:<4} ({score_tag}) | "
                f"{res['reason']:<16} | "
                f"{res['plies']:2d} plies | "
                f"{res['elapsed']:5.1f}s",
                flush=True
            )

    all_games = PREV_GAMES + new_results

    # Tiers summary
    tier_order = [
        "GreedyBot_Baseline",
        "Stockfish_Level_0",
        "Stockfish_Level_1",
        "HeuristicBot_D3",
        "Stockfish_Level_2",
    ]

    print("\n" + "=" * 78, flush=True)
    print("               FINAL DEPTH 3 TOURNAMENT LEADERBOARD (20 GAMES)", flush=True)
    print("=" * 78, flush=True)
    print(f"{'Opponent Tier':<22} | {'Anchor':<7} | {'Record':<10} | {'Score %':<7} | {'Avg Plies':<9} | {'Perf. Elo'}", flush=True)
    print("-" * 78, flush=True)

    tier_ratings = []
    sf_ratings = []

    for tier_name in tier_order:
        games = [g for g in all_games if g["tier"] == tier_name]
        n = len(games)
        anchor_elo = games[0]["anchor_elo"]
        wins = sum(1 for g in games if g["score"] == 1.0)
        losses = sum(1 for g in games if g["score"] == 0.0)
        draws = sum(1 for g in games if g["score"] == 0.5)
        total_score = wins + 0.5 * draws
        score_pct = (total_score / n) * 100.0 if n > 0 else 0.0
        avg_plies = np.mean([g["plies"] for g in games]) if n > 0 else 0.0
        perf_elo = compute_fide_performance(total_score / n, anchor_elo) if n > 0 else 0.0

        rec_str = f"{wins}W-{losses}L-{draws}D"
        print(
            f"{tier_name:<22} | "
            f"~{anchor_elo:<6.0f} | "
            f"{rec_str:<10} | "
            f"{score_pct:>6.1f}% | "
            f"{avg_plies:>7.1f} pl | "
            f"~{perf_elo:.0f} Elo",
            flush=True
        )

        tier_ratings.append(perf_elo)
        if "Stockfish" in tier_name:
            sf_ratings.append(perf_elo)

    print("-" * 78, flush=True)
    avg_overall = float(np.mean(tier_ratings))
    avg_sf = float(np.mean(sf_ratings)) if sf_ratings else avg_overall
    print(f"CALIBRATED STOCKFISH ELO (D3)     : ~{avg_sf:.0f} Elo", flush=True)
    print(f"OVERALL TOURNAMENT PERFORMANCE    : ~{avg_overall:.0f} Elo", flush=True)
    print("=" * 78 + "\n", flush=True)


if __name__ == "__main__":
    main()
