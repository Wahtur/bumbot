import torch
import os, glob

save_dir = "checkpoints"
latest_path = os.path.join(save_dir, "weights_latest.pt")

if os.path.exists(latest_path):
    data = torch.load(latest_path, map_location="cpu", weights_only=False)
    w = data["weights"]
    if isinstance(w, torch.Tensor):
        w = w.clone()
    else:
        w = torch.tensor(w, dtype=torch.float32)

    # Enforce domain sign constraints
    w[0:5].clamp_(min=0.1)     # Material
    w[645].clamp_(min=0.03)    # Mobility (positive!)
    w[646].clamp_(min=0.08)    # Center control (positive!)
    w[647].clamp_(max=-0.05)   # Doubled pawns (negative)
    w[648].clamp_(max=-0.05)   # Isolated pawns (negative)
    w[649].clamp_(min=0.05)    # Passed pawns (positive)
    w[650].clamp_(min=0.05)    # King shield (positive)

    data["weights"] = w
    if "model_state" in data and "linear.weight" in data["model_state"]:
        data["model_state"]["linear.weight"] = w.view(1, -1)

    torch.save(data, latest_path)
    print(f"Updated {latest_path} with domain-constrained weights:")
    print(f"  Mobility: {w[645].item():.4f}")
    print(f"  Center: {w[646].item():.4f}")
    print(f"  Passed: {w[649].item():.4f}")
