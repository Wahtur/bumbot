"""
benchmark_depth3_comprehensive.py
---------------------------------
Comprehensive Elo Benchmark Tournament for Bumbot V2 at Search Depth 3:
  - Supports 40 games per tier (20 balanced 2-ply opening book pairs)
  - Alternates colors on each opening pair (Bumbot White, then Black)
    to eliminate opening and first-move color bias.
  - Computes FIDE performance ratings and overall calibrated Elo.
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

import feature_extractor
import search_engine

# 20 Balanced 2-ply Opening Positions (UCI moves)
OPENING_BOOK_2PLY = [
    ["e2e4", "e7e5"],  # King's Pawn
    ["e2e4", "c7c5"],  # Sicilian Defense
    ["e2e4", "e7e6"],  # French Defense
    ["e2e4", "c7c6"],  # Caro-Kann Defense
    ["e2e4", "d7d5"],  # Scandinavian Defense
    ["e2e4", "d7d6"],  # Pirc Defense
    ["d2d4", "d7d5"],  # Queen's Gambit / Closed
    ["d2d4", "g8f6"], # Indian Defenses
    ["d2d4", "e7e6"],  # Queen's Pawn / Franco-Indian
    ["d2d4", "f7f5"],  # Dutch Defense
    ["c2c4", "e7e5"],  # English: Reversed Sicilian
    ["c2c4", "c7c5"],  # Symmetrical English
    ["c2c4", "g8f6"], # English: Anglo-Indian
    ["g1f3", "d7d5"],  # Reti Opening
    ["g1f3", "g8f6"], # King's Indian Attack setup
    ["g1f3", "c7c5"],  # Reti: Sicilian Invitation
    ["b2b3", "e7e5"],  # Nimzo-Larsen Attack
    ["g2g3", "d7d5"],  # King's Fianchetto
    ["f2f4", "d7d5"],  # Bird's Opening
    ["d2d4", "c7c5"],  # Benoni Defense setup
]


class BasePlayer:
    def __init__(self, name: str):
        self.name = name

    def get_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError


class GreedyPlayer(BasePlayer):
    def __init__(self, name: str = "GreedyBot"):
        super().__init__(name)
        self.piece_vals = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000
        }

    def get_move(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        best_val = -1
        best_moves = []
        for m in moves:
            val = 0
            if board.is_capture(m):
                cap = board.piece_at(m.to_square)
                val = self.piece_vals.get(cap.piece_type, 100) if cap else 100
            elif m.promotion:
                val = 800
            if val > best_val:
                best_val = val
                best_moves = [m]
            elif val == best_val:
                best_moves.append(m)
        return best_moves[0] if best_moves else moves[0]


class MinimaxPlayer(BasePlayer):
    def __init__(self, name: str, weights: np.ndarray, depth: int = 3):
        super().__init__(name)
        self.weights = weights
        self.depth = depth

    def get_move(self, board: chess.Board) -> chess.Move:
        move, _, _, _ = search_engine.get_best_move(board, self.weights, depth=self.depth)
        return move if move and move in board.legal_moves else list(board.legal_moves)[0]


class StockfishPlayer(BasePlayer):
    def __init__(self, name: str, skill_level: Optional[int] = None, movetime: float = 0.03):
        super().__init__(name)
        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        if skill_level is not None:
            self.engine.configure({"Skill Level": max(0, min(20, skill_level))})
        self.limit = chess.engine.Limit(time=movetime)

    def get_move(self, board: chess.Board) -> chess.Move:
        res = self.engine.play(board, self.limit)
        return res.move

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


def play_opening_game(
    w_player: BasePlayer,
    b_player: BasePlayer,
    opening_moves: List[str],
    max_plies: int = 200,
) -> Tuple[float, str, int]:
    board = chess.Board()
    for uci_str in opening_moves:
        board.push(chess.Move.from_uci(uci_str))

    ply = len(opening_moves)
    while not board.is_game_over() and ply < max_plies:
        curr = w_player if board.turn == chess.WHITE else b_player
        move = curr.get_move(board)
        board.push(move)
        ply += 1

    if board.is_checkmate():
        score = 0.0 if board.turn == chess.WHITE else 1.0
        reason = "checkmate"
    elif board.is_stalemate():
        score = 0.5
        reason = "stalemate"
    elif board.is_insufficient_material():
        score = 0.5
        reason = "insufficient_material"
    elif board.is_fifty_moves():
        score = 0.5
        reason = "fifty_moves"
    elif board.is_repetition(3):
        score = 0.5
        reason = "repetition"
    else:
        score = 0.5
        reason = "max_plies_draw"

    return score, reason, ply


def run_single_match(args_tuple) -> Dict:
    (
        game_id, tier_name, opp_type, opp_param, anchor_elo,
        bumbot_white, opening_moves, depth, max_plies, weights_np
    ) = args_tuple

    bumbot = MinimaxPlayer(name=f"Bumbot_D{depth}", weights=weights_np, depth=depth)
    sf_engine = None

    if opp_type == "stockfish":
        sf_player = StockfishPlayer(name=tier_name, skill_level=opp_param, movetime=0.03)
        opponent = sf_player
        sf_engine = sf_player
    elif opp_type == "heuristic":
        h_w = feature_extractor.get_default_heuristic_weights().numpy()
        opponent = MinimaxPlayer(name=tier_name, weights=h_w, depth=depth)
    elif opp_type == "greedy":
        opponent = GreedyPlayer(name=tier_name)
    else:
        raise ValueError(f"Unknown opponent: {opp_type}")

    t0 = time.time()
    try:
        if bumbot_white:
            score, reason, plies = play_opening_game(bumbot, opponent, opening_moves, max_plies)
            b_score = score
            color = "White"
        else:
            score, reason, plies = play_opening_game(opponent, bumbot, opening_moves, max_plies)
            b_score = 1.0 - score
            color = "Black"
    finally:
        if sf_engine:
            sf_engine.close()

    elapsed = time.time() - t0
    outcome = "WIN" if b_score == 1.0 else ("DRAW" if b_score == 0.5 else "LOSS")
    return {
        "game_id": game_id,
        "tier": tier_name,
        "anchor_elo": anchor_elo,
        "color": color,
        "score": b_score,
        "outcome": outcome,
        "reason": reason,
        "plies": plies,
        "elapsed": elapsed,
        "opening": " ".join(opening_moves),
    }


def compute_fide_performance(score_fraction: float, anchor_elo: float) -> float:
    clamped = max(0.01, min(0.99, score_fraction))
    delta = -400.0 * np.log10((1.0 - clamped) / clamped)
    return anchor_elo + delta


def run_tournament(checkpoint_path: str, games_per_tier: int = 40, depth: int = 3, workers: int = 4, max_plies: int = 200):
    d = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = d["weights"] if "weights" in d else d
    if isinstance(weights, dict) and "linear.weight" in weights:
        weights = weights["linear.weight"].squeeze(0)
    w_np = weights.numpy() if isinstance(weights, torch.Tensor) else np.array(weights)

    tiers = [
        ("GreedyBot_Baseline", "greedy", None, 450.0),
        ("Stockfish_Level_0", "stockfish", 0, 850.0),
        ("Stockfish_Level_1", "stockfish", 1, 1050.0),
        ("HeuristicBot_D3", "heuristic", None, 1050.0),
        ("Stockfish_Level_2", "stockfish", 2, 1200.0),
    ]

    tasks = []
    g_id = 1
    num_openings = max(1, games_per_tier // 2)
    openings = (OPENING_BOOK_2PLY * ((num_openings // len(OPENING_BOOK_2PLY)) + 1))[:num_openings]

    for tier_name, opp_type, opp_param, anchor in tiers:
        for op in openings:
            tasks.append((g_id, tier_name, opp_type, opp_param, anchor, True, op, depth, max_plies, w_np))
            g_id += 1
            tasks.append((g_id, tier_name, opp_type, opp_param, anchor, False, op, depth, max_plies, w_np))
            g_id += 1

    print("=" * 80)
    print(f" BUMBOT V2 STATISTICAL TOURNAMENT ({len(tasks)} TOTAL GAMES, {games_per_tier}/TIER)")
    print(f" Checkpoint: {checkpoint_path} | Depth: {depth} | Workers: {workers}")
    print("=" * 80)

    results_by_tier = {t[0]: [] for t in tiers}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_match, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            res = f.result()
            done += 1
            results_by_tier[res["tier"]].append(res)
            print(
                f"[{done:03d}/{len(tasks):03d}] {res['tier']:<18} [{res['color']:<5}] "
                f"{res['outcome']:<4} | {res['reason']:<16} | {res['plies']:2d} pl | {res['elapsed']:4.1f}s | {res['opening']}",
                flush=True
            )

    print("\n" + "=" * 80)
    print("                     FINAL TOURNAMENT LEADERBOARD")
    print("=" * 80)
    print(f"{'Opponent Tier':<22} | {'Anchor':<7} | {'Record':<12} | {'Score %':<7} | {'Avg Pl':<7} | {'Perf Elo'}")
    print("-" * 80)

    all_perf = []
    sf_perf = []
    for tier_name, _, _, anchor in tiers:
        games = results_by_tier[tier_name]
        n = len(games)
        wins = sum(1 for g in games if g["score"] == 1.0)
        losses = sum(1 for g in games if g["score"] == 0.0)
        draws = sum(1 for g in games if g["score"] == 0.5)
        score = (wins + 0.5 * draws) / n
        avg_pl = np.mean([g["plies"] for g in games])
        perf = compute_fide_performance(score, anchor)
        all_perf.append(perf)
        if "Stockfish" in tier_name:
            sf_perf.append(perf)
        print(f"{tier_name:<22} | ~{anchor:<6.0f} | {wins}W-{losses}L-{draws}D | {score*100:>5.1f}% | {avg_pl:>5.1f} | ~{perf:.0f} Elo")

    print("-" * 80)
    avg_sf = np.mean(sf_perf) if sf_perf else np.mean(all_perf)
    avg_all = np.mean(all_perf)
    print(f"CALIBRATED STOCKFISH ELO (D3)     : ~{avg_sf:.0f} Elo")
    print(f"OVERALL TOURNAMENT PERFORMANCE    : ~{avg_all:.0f} Elo")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/opt/bumbot/checkpoints_v2/weights_latest.pt")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_tournament(args.checkpoint, games_per_tier=args.games, depth=args.depth, workers=args.workers)
