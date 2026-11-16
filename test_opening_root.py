import chess
import feature_extractor
import search_engine
import torch, os

d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]
w_heur = feature_extractor.get_default_heuristic_weights().numpy()

b = chess.Board()
print("Starting board candidate moves at Depth 2 for LATEST weights:")
for m in search_engine.order_moves(b):
    b.push(m)
    val, reply, _ = search_engine.alpha_beta(b, w_latest, depth=1, alpha=-float("inf"), beta=float("inf"), is_maximizing=False)
    b.pop()
    print(f"  {b.san(m):4s} -> eval={val:+.3f} (Black reply: {b.san(reply) if reply else 'None'})")
