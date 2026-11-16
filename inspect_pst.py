import torch
import feature_extractor
import numpy as np

d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

idx = 5
print("PST Weights analysis:")
for pt_name in ["Pawn", "Knight", "Bishop", "Rook", "Queen"]:
    w_pst = w[idx:idx+64]
    b_pst = w[idx+320:idx+320+64]
    print(f"  {pt_name:7s} | White mean: {w_pst.mean():+.4f}, min: {w_pst.min():+.4f}, max: {w_pst.max():+.4f} | Black mean: {b_pst.mean():+.4f}, min: {b_pst.min():+.4f}, max: {b_pst.max():+.4f}")
    idx += 64
