"""
evaluate.py
-----------
Evaluation and Benchmarking Suite for TD-Leaf Chess Bot

Provides:
  - RandomBot: Plays random legal moves (Baseline Elo ~100)
  - GreedyBot: 1-ply greedy captures (Baseline Elo ~400)
  - HeuristicBot: Minimax with un-trained standard chess heuristic weights
  - TrainedBot: Loads a trained TD-Leaf checkpoint
  - Head-to-head tournament runner with Elo estimation and match statistics.
"""

import os
import argparse
import random
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import chess

import feature_extractor
import search_engine


class BasePlayer:
    """Abstract interface for a chess player agent."""
    def __init__(self, name: str):
        self.name = name

    def get_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError


class RandomPlayer(BasePlayer):
    """Plays uniformly random legal moves."""
    def __init__(self, name: str = "RandomBot"):
        super().__init__(name)

    def get_move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


class GreedyPlayer(BasePlayer):
    """1-ply greedy capture bot: captures highest value victim, else random move."""
    def __init__(self, name: str = "GreedyBot"):
        super().__init__(name)
        self.piece_vals = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000,
        }

    def get_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        best_val = -1
        best_moves = []
        
        for m in legal_moves:
            val = 0
            if board.is_capture(m):
                captured = board.piece_at(m.to_square)
                val = self.piece_vals.get(captured.piece_type, 100) if captured else 100
            elif m.promotion:
                val = 800
                
            if val > best_val:
                best_val = val
                best_moves = [m]
            elif val == best_val:
                best_moves.append(m)
                
        return random.choice(best_moves) if best_moves else random.choice(legal_moves)


class MinimaxPlayer(BasePlayer):
    """Minimax player powered by weights tensor and specified search depth."""
    def __init__(self, name: str, weights: torch.Tensor, depth: int = 2):
        super().__init__(name)
        self.weights = weights
        self.depth = depth

    def get_move(self, board: chess.Board) -> chess.Move:
        best_move, _, _, _ = search_engine.get_best_move(
            board, self.weights, depth=self.depth
        )
        if best_move is None or best_move not in board.legal_moves:
            return random.choice(list(board.legal_moves))
        return best_move


def load_player_from_checkpoint(ckpt_path: str, depth: int = 2, name: Optional[str] = None) -> MinimaxPlayer:
    """Loads weights from a saved .pt checkpoint and returns a MinimaxPlayer."""
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = data["weights"] if "weights" in data else data
    if isinstance(weights, dict) and "linear.weight" in weights:
        weights = weights["linear.weight"].squeeze(0)
    player_name = name or f"TDLeaf_Game{data.get('game', 'Custom')}_D{depth}"
    return MinimaxPlayer(name=player_name, weights=weights, depth=depth)


def play_match(
    white_player: BasePlayer,
    black_player: BasePlayer,
    max_plies: int = 150,
) -> Tuple[float, str, int]:
    """
    Plays a single game between White and Black.
    Returns:
        score: +1.0 (White win), 0.5 (Draw), 0.0 (Black win)
        termination_reason: reason string
        plies: total plies
    """
    board = chess.Board()
    ply = 0
    
    while not board.is_game_over() and ply < max_plies:
        current_player = white_player if board.turn == chess.WHITE else black_player
        move = current_player.get_move(board)
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


def run_tournament(
    player1: BasePlayer,
    player2: BasePlayer,
    num_games: int = 20,
    opp_elo: float = 100.0,
) -> Dict:
    """
    Plays a balanced tournament between two players (alternating colors)
    and estimates player1's performance and Elo relative to player2.
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_plies = 0

    print(f"\n--- Tournament: {player1.name} vs {player2.name} ({num_games} games) ---")
    
    for g in range(1, num_games + 1):
        # Alternate colors
        if g % 2 == 1:
            w, b = player1, player2
            score, reason, plies = play_match(w, b)
            p1_score = score
        else:
            w, b = player2, player1
            score, reason, plies = play_match(w, b)
            p1_score = 1.0 - score
            
        total_plies += plies
        if p1_score == 1.0:
            p1_wins += 1
            res_str = f"{player1.name} won"
        elif p1_score == 0.0:
            p2_wins += 1
            res_str = f"{player2.name} won"
        else:
            draws += 1
            res_str = "Draw"
            
        print(f"Game {g:2d}/{num_games}: {res_str} ({reason}, {plies} plies)")

    total_score = p1_wins + 0.5 * draws
    score_pct = (total_score / num_games) * 100.0
    
    # Elo calculation: R1 = R2 - 400 * log10((1 - S) / S)
    # Clamped for extreme scores
    clamped_s = max(0.01, min(0.99, total_score / num_games))
    elo_delta = -400.0 * np.log10((1.0 - clamped_s) / clamped_s)
    est_elo = opp_elo + elo_delta

    results = {
        "p1_name": player1.name,
        "p2_name": player2.name,
        "games": num_games,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "score_pct": score_pct,
        "avg_plies": total_plies / num_games,
        "est_elo": est_elo,
    }

    print("\n=== Tournament Summary ===")
    print(f"{player1.name}: {p1_wins}W - {p2_wins}L - {draws}D ({score_pct:.1f}%)")
    print(f"Average Game Length: {results['avg_plies']:.1f} plies")
    print(f"Estimated Elo (vs {player2.name} @ {opp_elo:.0f}): {est_elo:.1f} Elo")
    print("==========================\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Chess Bot against Baselines")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--opponent", type=str, choices=["random", "greedy", "heuristic"], default="random")
    parser.add_argument("--games", type=int, default=10, help="Number of match games")
    parser.add_argument("--depth", type=int, default=2, help="Search depth for bot")
    
    args = parser.parse_args()

    # Setup player 1
    if args.checkpoint and os.path.exists(args.checkpoint):
        p1 = load_player_from_checkpoint(args.checkpoint, depth=args.depth)
    else:
        print("[Notice] No checkpoint specified or found. Using default Heuristic Bot as Player 1.")
        h_weights = feature_extractor.get_default_heuristic_weights()
        p1 = MinimaxPlayer(name=f"Heuristic_D{args.depth}", weights=h_weights, depth=args.depth)

    # Setup player 2
    if args.opponent == "random":
        p2 = RandomPlayer()
        opp_elo = 100.0
    elif args.opponent == "greedy":
        p2 = GreedyPlayer()
        opp_elo = 400.0
    else:
        h_weights = feature_extractor.get_default_heuristic_weights()
        p2 = MinimaxPlayer(name="Heuristic_D1", weights=h_weights, depth=1)
        opp_elo = 600.0

    run_tournament(p1, p2, num_games=args.games, opp_elo=opp_elo)
