import chess
import feature_extractor
import search_engine
import evaluate
import torch, os

d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w_fixed = d["weights"].numpy().copy() if isinstance(d["weights"], torch.Tensor) else d["weights"].copy()
w_fixed[645] = max(0.02, float(w_fixed[645]))
w_fixed[646] = max(0.05, float(w_fixed[646]))

w_heur = feature_extractor.get_default_heuristic_weights().numpy()

p_white = evaluate.MinimaxPlayer("White_LatestFixed", w_fixed, depth=2)
p_black = evaluate.MinimaxPlayer("Black_Heuristic", w_heur, depth=2)

b = chess.Board()
moves = []
for ply in range(60):
    if b.is_game_over():
        break
    curr = p_white if b.turn == chess.WHITE else p_black
    mv = curr.get_move(b)
    moves.append(b.san(mv))
    b.push(mv)

print(f"Match ended at ply {len(moves)}")
print("Moves:", " ".join(moves[:20]))
print("Game Over?", b.is_game_over())
print("Checkmate?", b.is_checkmate())
sc = search_engine.evaluate_board(b, w_heur)
print("Final eval (White perspective):", f"{sc:+.2f}")
