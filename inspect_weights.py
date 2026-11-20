import sys
import os
import torch
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_v2/weights_latest.pt"
if not os.path.exists(path):
    print(f"File not found: {path}")
    sys.exit(1)

d = torch.load(path, map_location="cpu", weights_only=False)
w = d["weights"]
if isinstance(w, torch.Tensor):
    w = w.numpy()

print(f"=== Checkpoint: {path} (dim {len(w)}) ===")
print("Game:", d.get("game", "unknown"))
print("L2 Norm:", np.linalg.norm(w))

if len(w) == 808:
    mg = w[:404]
    eg = w[404:]
    print("Middlegame Material [P, N, B, R, Q]:", np.round(mg[0:5], 3))
    print("Endgame Material    [P, N, B, R, Q]:", np.round(eg[0:5], 3))
    print("MG Strategic [BishopPair, OpenRook, SemiRook, Tempo, Doubled, Isolated, Center, Shield, Mob]:")
    print("  ", np.round(mg[395:404], 3))
    print("EG Strategic:")
    print("  ", np.round(eg[395:404], 3))
    print("MG Passed Pawns (Ranks 2-7):", np.round(mg[389:395], 3))
    print("EG Passed Pawns (Ranks 2-7):", np.round(eg[389:395], 3))
elif len(w) == 800:
    mg = w[:400]
    eg = w[400:]
    print("Middlegame Material [P, N, B, R, Q]:", np.round(mg[0:5], 3))
    print("Endgame Material    [P, N, B, R, Q]:", np.round(eg[0:5], 3))
    print("MG Strategic [BishopPair, OpenRook, SemiRook, Tempo, Doubled, Isolated, Center, Shield]:")
    print("  ", np.round(mg[379:387], 3))
    print("MG P3 Upgrades [SafeMob(4), EdgeKing, KingProx, PasserProx, KingZone, Outpost, ProtPasser, Phalanx, NClosed, ROpen]:")
    print("  ", np.round(mg[387:400], 3))
