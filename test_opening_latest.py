import chess
import feature_extractor
import search_engine
import torch, os

d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]
w_heur = feature_extractor.get_default_heuristic_weights().numpy()

b = chess.Board("rnb1kbnr/pppp1ppp/4pq2/8/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 2 3")
print("Position: 1. c4 e6 2. Nf3 Qf6")

print("Candidate moves and scores with LATEST weights (w_latest):")
moves = []
for m in search_engine.order_moves(b):
    b.push(m)
    # 1-ply eval from Black's response
    min_sc = float("inf")
    best_reply = None
    for b_m in search_engine.order_moves(b):
        b.push(b_m)
        sc = search_engine.evaluate_board(b, w_latest)
        b.pop()
        if sc < min_sc:
            min_sc = sc
            best_reply = b_m
    b.pop()
    moves.append((m, min_sc, best_reply))

moves.sort(key=lambda x: x[1], reverse=True)
for m, sc, reply in moves[:10]:
    print(f"  {b.san(m):6s} -> eval={sc:+.3f} (Black reply: {b.san(m) if reply else 'None'} -> {reply.uci() if reply else ''})")
