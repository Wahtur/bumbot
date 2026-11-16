import chess
import feature_extractor
import search_engine
import torch, os

w_latest = feature_extractor.get_default_heuristic_weights().numpy()
if os.path.exists("checkpoints/weights_latest.pt"):
    d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
    w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

w_heur = feature_extractor.get_default_heuristic_weights().numpy()

b = chess.Board()
b.push_san("c4")
b.push_san("e6")
b.push_san("Nf3")
b.push_san("Qf6")

print("Position after 1. c4 e6 2. Nf3 Qf6:")
print(b)

m_latest, sc_latest, _, _ = search_engine.get_best_move(b, w_latest, depth=2)
print("Latest chose:", b.san(m_latest), "eval:", sc_latest)

m_heur, sc_heur, _, _ = search_engine.get_best_move(b, w_heur, depth=2)
print("Heuristic chose:", b.san(m_heur), "eval:", sc_heur)

# Let's inspect why Ng1 was chosen!
for mv in search_engine.order_moves(b):
    b.push(mv)
    # What does Black reply at depth 1?
    val, reply, _ = search_engine.alpha_beta(b, w_heur, depth=1, alpha=-float("inf"), beta=float("inf"), is_maximizing=False)
    b.pop()
    if b.san(mv) in ["Ng1", "Nc3", "d4", "e3", "g3"]:
        print(f"  White move {b.san(mv)} -> Black best reply {b.san(reply) if reply else 'None'}, eval={val:.2f}")
