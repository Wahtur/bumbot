"""
benchmark_depth3_comprehensive.py
---------------------------------
Comprehensive Elo Benchmark Tournament for Bumbot at Search Depth 3.
Runs parallel matches across calibrated anchor opponents:
  1. GreedyBot (Anchor: 450 Elo)
  2. HeuristicBot D3 (Anchor: 1050 Elo) - Minimax D3 with textbook heuristics
  3. Stockfish 16 Level 0 (Anchor: 850 Elo)
  4. Stockfish 16 Level 1 (Anchor: 1050 Elo)
  5. Stockfish 16 Level 2 (Anchor: 1200 Elo)

Uses ProcessPoolExecutor to distribute games across server CPU cores.
"""

import sys
import os
import time
import argparse
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import chess
import chess.engine

sys.path.insert(0, "/opt/bumbot")
import feature_extractor
import search_engine
from evaluate import BasePlayer, MinimaxPlayer, GreedyPlayer, play_match
from benchmark_vs_stockfish import StockfishPlayer


def run_single_match(args_tuple) -> Dict:
    """Executes a single game between Bumbot and an opponent inside a worker process."""
    (
        game_num,
        tier_name,
        opp_type,
        opp_param,
        anchor_elo,
        bumbot_is_white,
        depth,
        max_plies,
        weights_np,
    ) = args_tuple

    bumbot = MinimaxPlayer(name=f"Bumbot_D{depth}", weights=weights_np, depth=depth)

    # Initialize opponent
    sf_engine = None
    if opp_type == "stockfish":
        sf_player = StockfishPlayer(name=tier_name, skill_level=opp_param, movetime=0.03)
        opponent = sf_player
        sf_engine = sf_player
    elif opp_type == "heuristic":
        h_weights = feature_extractor.get_default_heuristic_weights().numpy()
        opponent = MinimaxPlayer(name=tier_name, weights=h_weights, depth=depth)
    elif opp_type == "greedy":
        opponent = GreedyPlayer(name=tier_name)
    else:
        raise ValueError(f"Unknown opponent type: {opp_type}")

    t0 = time.time()
    try:
        if bumbot_is_white:
            w_player, b_player = bumbot, opponent
            score, reason, plies = play_match(w_player, b_player, max_plies=max_plies)
            bumbot_score = score
            color = "White"
        else:
            w_player, b_player = opponent, bumbot
            score, reason, plies = play_match(w_player, b_player, max_plies=max_plies)
            bumbot_score = 1.0 - score
            color = "Black"
    finally:
        if sf_engine is not None:
            sf_engine.close()

    elapsed = time.time() - t0

    if bumbot_score == 1.0:
        outcome = "WIN"
    elif bumbot_score == 0.0:
        outcome = "LOSS"
    else:
        outcome = "DRAW"

    return {
        "game_num": game_num,
        "tier": tier_name,
        "anchor_elo": anchor_elo,
        "color": color,
        "score": bumbot_score,
        "outcome": outcome,
        "reason": reason,
        "plies": plies,
        "elapsed": elapsed,
    }


def compute_fide_performance(score_fraction: float, anchor_elo: float) -> float:
    """Calculates FIDE performance rating from score fraction and opponent anchor."""
    # Clamp to avoid infinity on 100% or 0%
    clamped = max(0.02, min(0.98, score_fraction))
    delta = -400.0 * np.log10((1.0 - clamped) / clamped)
    return anchor_elo + delta


def main():
    parser = argparse.ArgumentParser(description="Full Elo Determination for Bumbot Depth 3")
    parser.add_argument("--checkpoint", type=str, default="/opt/bumbot/checkpoints/weights_latest.pt")
    parser.add_argument("--depth", type=int, default=3, help="Search depth (default: 3)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes (default: 4)")
    parser.add_argument("--max-plies", type=int, default=80, help="Max plies per game (default: 80)")
    parser.add_argument("--games-per-tier", type=int, default=4, help="Games per tier, must be even (default: 4)")
    args = parser.parse_args()

    print("=" * 70)
    print("      BUMBOT DEPTH 3 COMPREHENSIVE ELO TOURNAMENT")
    print("=" * 70)
    print(f"Search Depth      : {args.depth}")
    print(f"Parallel Workers  : {args.workers}")
    print(f"Max Plies / Game  : {args.max_plies}")
    print(f"Games Per Tier    : {args.games_per_tier} (balanced colors)")

    # Load weights
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = data["weights"] if "weights" in data else data
    if isinstance(weights, dict) and "linear.weight" in weights:
        weights = weights["linear.weight"].squeeze(0)
    if isinstance(weights, torch.Tensor):
        weights_np = weights.detach().cpu().numpy()
    else:
        weights_np = np.asarray(weights, dtype=np.float32)

    game_num_ckpt = data.get("game", "Latest")
    print(f"Loaded Checkpoint : Game {game_num_ckpt} ({args.checkpoint})")

    # Define Tiers
    # (tier_name, opp_type, opp_param, anchor_elo)
    tiers = [
        ("GreedyBot_Baseline", "greedy", None, 450.0),
        ("Stockfish_Level_0", "stockfish", 0, 850.0),
        ("Stockfish_Level_1", "stockfish", 1, 1050.0),
        ("HeuristicBot_D3", "heuristic", None, 1050.0),
        ("Stockfish_Level_2", "stockfish", 2, 1200.0),
    ]

    # Build task list with balanced alternating colors
    tasks = []
    g_count = 1
    for tier_name, opp_type, opp_param, anchor_elo in tiers:
        for g in range(args.games_per_tier):
            bumbot_white = (g % 2 == 0)
            tasks.append((
                g_count,
                tier_name,
                opp_type,
                opp_param,
                anchor_elo,
                bumbot_white,
                args.depth,
                args.max_plies,
                weights_np,
            ))
            g_count += 1

    total_games = len(tasks)
    print(f"Total Match Games : {total_games}")
    print("-" * 70)

    start_time = time.time()
    results_by_tier: Dict[str, List[Dict]] = {t[0]: [] for t in tiers}
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_match, t): t for t in tasks}

        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            results_by_tier[res["tier"]].append(res)
            
            score_tag = f"+{res['score']:.1f}" if res['score'] > 0 else " 0.0"
            print(
                f"[{completed:02d}/{total_games:02d}] "
                f"{res['tier']:<18} "
                f"[{res['color']:<5}]: "
                f"{res['outcome']:<4} ({score_tag}) | "
                f"{res['reason']:<16} | "
                f"{res['plies']:2d} plies | "
                f"{res['elapsed']:5.1f}s",
                flush=True
            )

    total_elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Tournament completed in {total_elapsed / 60.0:.2f} minutes ({total_elapsed:.1f}s total)\n")

    # Summarize Results per Tier
    print("=" * 76)
    print("                     DEPTH 3 TOURNAMENT LEADERBOARD")
    print("=" * 76)
    print(f"{'Opponent Tier':<22} | {'Anchor':<7} | {'Record':<10} | {'Score %':<7} | {'Avg Plies':<9} | {'Perf. Elo'}")
    print("-" * 76)

    tier_ratings = []
    sf_ratings = []

    for tier_name, opp_type, opp_param, anchor_elo in tiers:
        games = results_by_tier[tier_name]
        n = len(games)
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
            f"~{perf_elo:.0f} Elo"
        )

        tier_ratings.append(perf_elo)
        if "Stockfish" in tier_name:
            sf_ratings.append(perf_elo)

    print("-" * 76)
    avg_overall = float(np.mean(tier_ratings))
    avg_sf = float(np.mean(sf_ratings)) if sf_ratings else avg_overall
    print(f"STOCKFISH-CALIBRATED RATING (D3)  : ~{avg_sf:.0f} Elo")
    print(f"OVERALL TOURNAMENT PERFORMANCE    : ~{avg_overall:.0f} Elo")
    print("=" * 76)


if __name__ == "__main__":
    main()
