import chess
import feature_extractor
import search_engine
import evaluate
import torch, os

d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]
w_heur = feature_extractor.get_default_heuristic_weights().numpy()

# Test 1: With Temperature at opening (plies 1-4)
print("=== Test 1: Opening Temperature Exploration (T=0.8 for plies 0-3, then greedy) ===")
b = chess.Board()
moves = []
for ply in range(30):
    if b.is_game_over():
        break
    temp = 0.8 if ply < 4 else 0.0
    w = w_latest if b.turn == chess.WHITE else w_heur
    m, sc, _, _ = search_engine.get_best_move_sampled(b, w, depth=2, temperature=temp)
    moves.append(b.san(m))
    b.push(m)

print("Opening:", " ".join(moves[:10]))
print("All 30 moves:", " ".join(moves))
sc = search_engine.evaluate_board(b, w_heur)
print("Eval after 30 plies (White perspective):", f"{sc:+.2f}")
