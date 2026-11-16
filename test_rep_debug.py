import chess
import feature_extractor
import search_engine
import evaluate
import torch, os

w_latest = feature_extractor.get_default_heuristic_weights().numpy()
if os.path.exists("checkpoints/weights_latest.pt"):
    d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
    w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

w_heur = feature_extractor.get_default_heuristic_weights().numpy()

p_latest_w = evaluate.MinimaxPlayer("Latest_W", w_latest, depth=2)
p_heur_b = evaluate.MinimaxPlayer("Heur_B", w_heur, depth=2)

board = chess.Board()
moves = []
ply = 0
while not board.is_game_over() and ply < 60:
    curr = p_latest_w if board.turn == chess.WHITE else p_heur_b
    mv = curr.get_move(board)
    san = board.san(mv)
    board.push(mv)
    moves.append(san)
    ply += 1

print("Game stopped at ply:", ply)
print("Game over?", board.is_game_over())
print("Is repetition(3)?", board.is_repetition(3))
print("Is checkmate?", board.is_checkmate())
print("Moves played:", " ".join(moves))
