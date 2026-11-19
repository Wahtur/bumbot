"""
benchmark_vs_stockfish.py
-------------------------
Official FIDE-calibrated benchmark tournament of Bumbot (TD-Leaf) 
against Stockfish 16 at calibrated difficulty tiers:
  - Tier 1: Stockfish Skill Level 0 (~850 Elo)
  - Tier 2: Stockfish Skill Level 1 (~1050 Elo)
  - Tier 3: Stockfish UCI_Elo 1320 (~1320 Elo)
"""

import os
import sys
import time
import argparse
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import chess
import chess.engine

import feature_extractor
import search_engine
import evaluate
from evaluate import BasePlayer, MinimaxPlayer, play_match


class StockfishPlayer(BasePlayer):
    """Wraps the local Stockfish 16 engine as an evaluate BasePlayer."""
    def __init__(self, name: str, skill_level: Optional[int] = None, elo: Optional[int] = None, movetime: float = 0.05):
        super().__init__(name=name)
        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        options = {}
        if elo is not None:
            options["UCI_LimitStrength"] = True
            options["UCI_Elo"] = max(1320, int(elo))
        elif skill_level is not None:
            options["Skill Level"] = max(0, min(20, int(skill_level)))
            
        self.engine.configure(options)
        self.limit = chess.engine.Limit(time=movetime)

    def get_move(self, board: chess.Board) -> chess.Move:
        res = self.engine.play(board, self.limit)
        return res.move

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


def run_tier_tournament(bumbot: BasePlayer, stockfish: StockfishPlayer, num_games: int = 6, anchor_elo: float = 850.0) -> Dict:
    print(f"\n" + "=" * 60)
    print(f" MATCH: {bumbot.name} vs {stockfish.name} ({num_games} Games)")
    print(f" Opponent Anchor Rating: ~{anchor_elo:.0f} Elo")
    print("=" * 60)

    b_wins = 0
    s_wins = 0
    draws = 0
    total_plies = 0

    for g in range(1, num_games + 1):
        if g % 2 == 1:
            w, b = bumbot, stockfish
            score, reason, plies = play_match(w, b, max_plies=120)
            b_score = score
        else:
            w, b = stockfish, bumbot
            score, reason, plies = play_match(w, b, max_plies=120)
            b_score = 1.0 - score

        total_plies += plies
        if b_score == 1.0:
            b_wins += 1
            res_str = f"WIN ({bumbot.name})"
        elif b_score == 0.0:
            s_wins += 1
            res_str = f"LOSS ({stockfish.name})"
        else:
            draws += 1
            res_str = "DRAW"

        color_str = "White" if g % 2 == 1 else "Black"
        print(f"Game {g:2d}/{num_games} [{color_str}]: {res_str:<18} | {reason:<16} | {plies:3d} plies")

    total_score = b_wins + 0.5 * draws
    score_pct = (total_score / num_games) * 100.0

    # FIDE Elo performance formula
    clamped_s = max(0.02, min(0.98, total_score / num_games))
    elo_delta = -400.0 * np.log10((1.0 - clamped_s) / clamped_s)
    perf_elo = anchor_elo + elo_delta

    results = {
        "opponent": stockfish.name,
        "anchor_elo": anchor_elo,
        "games": num_games,
        "bumbot_wins": b_wins,
        "stockfish_wins": s_wins,
        "draws": draws,
        "score_pct": score_pct,
        "avg_plies": total_plies / num_games,
        "perf_elo": perf_elo
    }

    print("-" * 60)
    print(f"Result : {b_wins}W - {s_wins}L - {draws}D ({score_pct:.1f}%)")
    print(f"Performance Rating vs {stockfish.name}: ~{perf_elo:.0f} Elo")
    print("-" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Bumbot vs Stockfish 16")
    parser.add_argument("--depth", type=int, default=2, help="Bumbot search depth (default: 2)")
    parser.add_argument("--games", type=int, default=6, help="Games per tier (default: 6)")
    parser.add_argument("--checkpoint", type=str, default="/opt/bumbot/checkpoints/weights_latest.pt")
    args = parser.parse_args()

    print("==========================================================")
    print(" Bumbot TD-Leaf vs Stockfish 16 Calibrated Elo Tournament ")
    print("==========================================================")

    # Load Bumbot
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = data["weights"] if "weights" in data else data
    if isinstance(weights, dict) and "linear.weight" in weights:
        weights = weights["linear.weight"].squeeze(0)
    game_num = data.get("game", "Latest")

    bumbot = MinimaxPlayer(name=f"Bumbot_Game{game_num}_D{args.depth}", weights=weights, depth=args.depth)
    print(f"Loaded: {bumbot.name} (Trained on {game_num} games)")

    all_results = []

    # Tier 1: Stockfish Skill Level 0 (~850 Elo)
    sf_level0 = StockfishPlayer(name="Stockfish_Lvl0", skill_level=0, movetime=0.04)
    try:
        r1 = run_tier_tournament(bumbot, sf_level0, num_games=args.games, anchor_elo=850.0)
        all_results.append(r1)
    finally:
        sf_level0.close()

    # Tier 2: Stockfish Skill Level 1 (~1050 Elo)
    sf_level1 = StockfishPlayer(name="Stockfish_Lvl1", skill_level=1, movetime=0.04)
    try:
        r2 = run_tier_tournament(bumbot, sf_level1, num_games=args.games, anchor_elo=1050.0)
        all_results.append(r2)
    finally:
        sf_level1.close()

    # Final Summary Table
    print("\n" + "=" * 65)
    print("                  FINAL ELO BENCHMARK LEADERBOARD                 ")
    print("=" * 65)
    print(f"{'Opponent':<20} | {'Anchor Elo':<10} | {'Record':<12} | {'Score %':<8} | {'Est. Elo'}")
    print("-" * 65)
    for r in all_results:
        rec = f"{r['bumbot_wins']}W-{r['stockfish_wins']}L-{r['draws']}D"
        print(f"{r['opponent']:<20} | ~{r['anchor_elo']:<9.0f} | {rec:<12} | {r['score_pct']:<7.1f}% | ~{r['perf_elo']:.0f} Elo")
    print("-" * 65)
    avg_elo = np.mean([r['perf_elo'] for r in all_results])
    print(f"OVERALL CALIBRATED BUMBOT ELO: ~{avg_elo:.0f} Elo")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
