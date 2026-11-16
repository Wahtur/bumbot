import chess
import feature_extractor
import search_engine
import torch, os

d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w = d["weights"].numpy().copy() if isinstance(d["weights"], torch.Tensor) else d["weights"].copy()

# Fix the inverted mobility weight to positive
w[645] = 0.03  # Positive mobility!
w[646] = 0.08  # Positive center control!

b = chess.Board()
print("Starting board candidate moves with POSITIVE mobility (+0.03):")
scores = []
for m in search_engine.order_moves(b):
    b.push(m)
    val, reply, _ = search_engine.alpha_beta(b, w, depth=1, alpha=-float("inf"), beta=float("inf"), is_maximizing=False)
    b.pop()
    scores.append((m, val, reply))

scores.sort(key=lambda x: x[1], reverse=True)
for m, sc, reply in scores[:8]:
    print(f"  {b.san(m):4s} -> eval={sc:+.3f} (Black reply: {b.san(reply) if reply else 'None'})")
