"""
eval_latest_match.py
--------------------
Plays a 4-game match between Trained Game 4000 and HeuristicBot
(2 games as White, 2 games as Black) at Search Depth 3 with
P0 and P2 enhancements active.
"""

import sys
import os
import torch
import chess
import feature_extractor
import search_engine

ckpt_path = "/opt/bumbot/checkpoints_v2/texel_tuned_weights.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = "/opt/bumbot/checkpoints_v2/weights_latest.pt"

data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
w_trained = data["weights"].numpy() if isinstance(data["weights"], torch.Tensor) else data["weights"]
w_heur = feature_extractor.get_default_heuristic_weights().numpy()

openings = [
    ("King's Pawn", ["e2e4", "e7e5"]),
    ("Queen's Pawn", ["d2d4", "d7d5"]),
]

print(f"Loaded checkpoint: {ckpt_path} (dim {len(w_trained)})")
print("======================================================================")
print(" 4-GAME MATCH: Bumbot V2.1 (Texel L-BFGS Tuned) vs Heuristic Baseline (Depth 3)")
print("======================================================================")

results = []

for op_name, op_moves in openings:
    for swap in [False, True]:
        white_is_trained = not swap
        white_name = "Bumbot_V2_1_Texel" if white_is_trained else "HeuristicBot"
        black_name = "HeuristicBot" if white_is_trained else "Bumbot_V2_1_Texel"
        
        w_white = w_trained if white_is_trained else w_heur
        w_black = w_heur if white_is_trained else w_trained
        
        board = chess.Board()
        for uci in op_moves:
            board.push_uci(uci)
            
        move_list = list(op_moves)
        print(f"\n--- Game: {op_name} | White: {white_name} vs Black: {black_name} ---")
        
        ply = len(move_list)
        while not board.is_game_over() and ply < 200:
            w_curr = w_white if board.turn == chess.WHITE else w_black
            move, score, _, nodes = search_engine.get_best_move(board, w_curr, depth=3)
            if move is None:
                break
            move_list.append(board.san(move))
            board.push(move)
            ply += 1
            
        res = board.result()
        term = "Checkmate" if board.is_checkmate() else ("Stalemate" if board.is_stalemate() else ("Max Plies" if ply >= 200 else "Draw"))
        print(f"Result: {res} ({term}) in {ply} plies")
        
        if white_is_trained:
            score_trained = 1.0 if res == "1-0" else (0.5 if res == "1/2-1/2" else 0.0)
        else:
            score_trained = 1.0 if res == "0-1" else (0.5 if res == "1/2-1/2" else 0.0)
            
        results.append(score_trained)
        print(f"Bumbot Score: {score_trained} | Moves: {' '.join(move_list[:20])}...")

print("\n======================================================================")
print(f" MATCH FINISHED: Bumbot V2 scored {sum(results)} / {len(results)} ({sum(results)/len(results)*100:.1f}%)")
print("======================================================================")
