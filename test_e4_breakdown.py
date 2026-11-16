import chess
import feature_extractor
import search_engine
import torch, os

d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]
w_heur = feature_extractor.get_default_heuristic_weights().numpy()

b = chess.Board()
feat_start = feature_extractor.extract_features_numpy(b)

b.push_san("e4")
feat_e4 = feature_extractor.extract_features_numpy(b)

names = feature_extractor.get_feature_names()
print("Feature differences after 1. e4:")
diff = feat_e4 - feat_start
changed = [(names[i], diff[i], w_latest[i], diff[i] * w_latest[i]) for i in range(len(diff)) if diff[i] != 0]

for name, d_val, w_val, contribution in changed:
    print(f"  {name:30s} | diff={d_val:+.1f} | weight={w_val:+.3f} | impact={contribution:+.3f}")

print("\nTotal static eval start:", float(np.dot(w_latest, feat_start)))
print("Total static eval 1. e4:", float(np.dot(w_latest, feat_e4)))

# Now what does Black reply to 1. e4 at depth 1?
min_sc = float("inf")
best_reply = None
for m in b.legal_moves:
    b.push(m)
    sc = search_engine.evaluate_board(b, w_latest)
    b.pop()
    if sc < min_sc:
        min_sc = sc
        best_reply = m
print(f"Black reply at depth 1 to 1. e4: {b.san(best_reply)} -> eval={min_sc:+.3f}")
