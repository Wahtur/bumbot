import sys
import torch

f = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/weights_latest.pt"
d = torch.load(f, map_location="cpu", weights_only=False)
print("File:", f, "Game:", d.get("game"), "Norm:", torch.norm(d["weights"]).item())
