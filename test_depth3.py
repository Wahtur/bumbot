import chess
import feature_extractor
import search_engine
import torch, os

w = feature_extractor.get_default_heuristic_weights().numpy()

b = chess.Board()
b.push_san("c4")
b.push_san("e6")
b.push_san("Nf3")
b.push_san("Qf6")

print("--- With Depth 3 Search ---")
m3, sc3, _, _ = search_engine.get_best_move(b, w, depth=3)
print("Depth 3 chose:", b.san(m3), "eval:", sc3)

# Test candidate moves at Depth 3
for mv in search_engine.order_moves(b):
    b.push(mv)
    val, reply, _ = search_engine.alpha_beta(b, w, depth=2, alpha=-float("inf"), beta=float("inf"), is_maximizing=False)
    b.pop()
    if b.san(mv) in ["Nc3", "d4", "e3", "g3", "Ng1"]:
        print(f"  Depth 3: White move {b.san(mv)} -> eval={val:.2f}")
