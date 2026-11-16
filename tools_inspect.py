import torch
import numpy as np
import feature_extractor

d = torch.load('/home/antigravity/bumbot/checkpoints/weights_latest.pt', map_location='cpu')
w = d['weights'].numpy() if hasattr(d['weights'], 'numpy') else np.array(d['weights'])
h = feature_extractor.get_default_heuristic_weights().numpy()
names = feature_extractor.get_feature_names()

print("=== Checkpoint: Game", d.get("game", "unknown"), "===")
print("Weight Vector Dimension:", len(w))
print("L2 Norm:", round(float(np.linalg.norm(w)), 4))

print("\n--- 1. Material Imbalance Weights (in pawn units) ---")
for i in range(5):
    print(f"{names[i]:<22} | Heuristic: {h[i]:+.3f} | Learned: {w[i]:+.4f} | Shift: {w[i]-h[i]:+.4f}")

print("\n--- 2. High-Level Positional Features ---")
for i in range(645, 651):
    print(f"{names[i]:<22} | Heuristic: {h[i]:+.3f} | Learned: {w[i]:+.4f} | Shift: {w[i]-h[i]:+.4f}")

w_pst = w[5:645]
h_pst = h[5:645]
print("\n--- 3. Piece-Square Tables (640 features) ---")
print(f"PST min: {w_pst.min():.4f}, max: {w_pst.max():.4f}, mean: {w_pst.mean():.4f}, std: {w_pst.std():.4f}")

top_pos_idx = np.argsort(w_pst)[-8:][::-1]
print("\nTop rewarded piece squares (Learned):")
for idx in top_pos_idx:
    feat_idx = 5 + idx
    print(f"  {names[feat_idx]:<25}: {w[feat_idx]:+.4f} (Heuristic was: {h[feat_idx]:+.4f})")

top_neg_idx = np.argsort(w_pst)[:8]
print("\nTop penalized piece squares (Learned):")
for idx in top_neg_idx:
    feat_idx = 5 + idx
    print(f"  {names[feat_idx]:<25}: {w[feat_idx]:+.4f} (Heuristic was: {h[feat_idx]:+.4f})")
