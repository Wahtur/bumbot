import sys
import torch
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_v2/weights_game_4000.pt"
d = torch.load(path, map_location="cpu", weights_only=False)
w = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

print(f"=== PST Analysis for {path} ===")
idx = 5
for pt_name in ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]:
    mg_pst = w[idx:idx+64]
    eg_pst = w[404+idx:404+idx+64]
    print(f"  {pt_name:7s} | MG mean: {mg_pst.mean():+.4f}, min: {mg_pst.min():+.4f}, max: {mg_pst.max():+.4f} | EG mean: {eg_pst.mean():+.4f}, min: {eg_pst.min():+.4f}, max: {eg_pst.max():+.4f}")
    idx += 64
