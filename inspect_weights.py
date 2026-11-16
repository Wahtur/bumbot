import torch
import feature_extractor

d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
w = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

print("Game:", d.get("game"))
print("Material weights:")
for name, val in zip(["Pawn", "Knight", "Bishop", "Rook", "Queen"], w[:5]):
    print(f"  {name}: {val:.3f}")

print("\nStrategic weights:")
for name, val in zip(["Mobility", "Center", "Doubled", "Isolated", "Passed", "King Shield"], w[645:651]):
    print(f"  {name}: {val:.3f}")

# Let's inspect PST weights for White vs Black
print("\nPST samples:")
w_pawns_pst = w[5:69]
b_pawns_pst = w[69:133]
print("White pawns PST mean:", w_pawns_pst.mean())
print("Black pawns PST mean:", b_pawns_pst.mean())
