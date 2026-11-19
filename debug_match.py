import sys
import os
import torch
import chess
import feature_extractor
import search_engine
import evaluate

d = torch.load("/opt/bumbot/checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w_trained = d["weights"]
w_heur = feature_extractor.get_default_heuristic_weights()

# Evaluate at Depth 3 (the depth the model was trained at!)
p_trained = evaluate.MinimaxPlayer("Trained_D3", w_trained, depth=3)
p_heur = evaluate.MinimaxPlayer("Heur_D3", w_heur, depth=3)

board = chess.Board()
moves = []
while not board.is_game_over() and len(moves) < 100:
    curr = p_trained if board.turn == chess.WHITE else p_heur
    m = curr.get_move(board)
    moves.append(board.san(m))
    board.push(m)

print("=== Match at Depth 3 ===")
print("Game Result:", board.result())
print("Termination:", "Checkmate" if board.is_checkmate() else ("Stalemate" if board.is_stalemate() else "Other"))
print("Total Plies:", len(moves))
print("Moves:", " ".join(moves))
