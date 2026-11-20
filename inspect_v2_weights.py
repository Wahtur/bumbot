"""
inspect_v2_weights.py
---------------------
Inspects and compares learned weights from checkpoints_v2/weights_game_0512.pt
against initial baseline heuristic weights, showing:
  1. Material values (MG vs EG)
  2. Passed Pawn Rank Buckets (MG vs EG)
  3. King Shield (MG vs EG)
  4. Strategic features (Bishop pair, Rooks on open files, etc.)
"""

import sys
import torch
import numpy as np

sys.path.insert(0, "/opt/bumbot")
import feature_extractor

ckpt_path = "/opt/bumbot/checkpoints_v2/weights_game_0512.pt"
d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
w_learned = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else np.array(d["weights"])
w_init = feature_extractor.get_default_heuristic_weights().numpy()

D = feature_extractor.BASE_FEATURE_DIM

print("=" * 70)
print("       BUMBOT V2 WEIGHT INSPECTION: GENERATION 512 vs INITIAL")
print("=" * 70)

# 1. Material
pieces = ["Pawn", "Knight", "Bishop", "Rook", "Queen"]
print(f"{'Feature':<20} | {'MG (Init)':<10} | {'MG (Gen 512)':<12} | {'EG (Init)':<10} | {'EG (Gen 512)':<12}")
print("-" * 70)
for i, p in enumerate(pieces):
    mg_i = w_init[i]
    mg_l = w_learned[i]
    eg_i = w_init[D + i]
    eg_l = w_learned[D + i]
    print(f"{p:<20} | {mg_i:<10.3f} | {mg_l:<12.3f} | {eg_i:<10.3f} | {eg_l:<12.3f}")

# 2. Passed Pawn Rank Buckets (ranks 2..7)
print("\n" + "=" * 70)
print(" PASSED PAWN RANK BUCKETS (RANKS 2..7)")
print("=" * 70)
print(f"{'Rank':<20} | {'MG (Init)':<10} | {'MG (Gen 512)':<12} | {'EG (Init)':<10} | {'EG (Gen 512)':<12}")
print("-" * 70)
for r in range(2, 8):
    b = r - 2
    idx = 389 + b
    mg_i = w_init[idx]
    mg_l = w_learned[idx]
    eg_i = w_init[D + idx]
    eg_l = w_learned[D + idx]
    print(f"Rank {r:<15} | {mg_i:<10.3f} | {mg_l:<12.3f} | {eg_i:<10.3f} | {eg_l:<12.3f}")

# 3. Strategic Features
print("\n" + "=" * 70)
print(" STRATEGIC EVALUATION FEATURES")
print("=" * 70)
strat_names = [
    ("Bishop Pair", 395),
    ("Rook Open File", 396),
    ("Rook Semi-Open File", 397),
    ("Tempo", 398),
    ("Doubled Pawns", 399),
    ("Isolated Pawns", 400),
    ("Center Control", 401),
    ("King Shield", 402),
    ("Pseudo-Mobility", 403),
]
print(f"{'Feature':<22} | {'MG (Init)':<10} | {'MG (Gen 512)':<12} | {'EG (Init)':<10} | {'EG (Gen 512)':<12}")
print("-" * 70)
for name, idx in strat_names:
    mg_i = w_init[idx]
    mg_l = w_learned[idx]
    eg_i = w_init[D + idx]
    eg_l = w_learned[D + idx]
    print(f"{name:<22} | {mg_i:<10.3f} | {mg_l:<12.3f} | {eg_i:<10.3f} | {eg_l:<12.3f}")
print("=" * 70)
