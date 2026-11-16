import chess
import feature_extractor
import search_engine
import torch, os

w_latest = feature_extractor.get_default_heuristic_weights().numpy()
if os.path.exists("checkpoints/weights_latest.pt"):
    d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
    w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

# Board right before White's blunder: after 13... Qxg2
moves = "e3 Nf6 Qf3 Ng8 Nc3 f6 Qh5+ g6 Qd5 Nh6 Nf3 d6 Bb5+ c6 Ba4 Nd7 Qc4 Qa5 Bxc6 bxc6 b4 Qf5 Nd4 Qg4 f4 Qxg2".split()
b = chess.Board()
for m in moves:
    b.push_san(m)

print("Position FEN:", b.fen())
print("White turn:", b.turn == chess.WHITE)

# Let's inspect all legal moves and their depth 2 alpha-beta scores
ordered = search_engine.order_moves(b)
print("Evaluating all White candidate moves at depth 2:")
scores = []
for mv in ordered:
    b.push(mv)
    sc, _, _ = search_engine.alpha_beta(b, w_latest, depth=1, alpha=-float("inf"), beta=float("inf"), is_maximizing=False)
    b.pop()
    scores.append((mv, sc))

scores.sort(key=lambda x: x[1], reverse=True)
for mv, sc in scores[:10]:
    print(f"  Move {mv.uci()} ({b.san(mv)}): eval={sc:.2f}")

print("\nLet's also evaluate with Heuristic weights:")
w_heur = feature_extractor.get_default_heuristic_weights().numpy()
scores_h = []
for mv in ordered:
    b.push(mv)
    sc, _, _ = search_engine.alpha_beta(b, w_heur, depth=1, alpha=-float("inf"), beta=float("inf"), is_maximizing=False)
    b.pop()
    scores_h.append((mv, sc))

scores_h.sort(key=lambda x: x[1], reverse=True)
for mv, sc in scores_h[:10]:
    print(f"  Heuristic Move {mv.uci()} ({b.san(mv)}): eval={sc:.2f}")
