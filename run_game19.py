import sys
import time
import numpy as np
import torch
import chess
import chess.engine

sys.path.insert(0, "/opt/bumbot")
import feature_extractor
import search_engine
from evaluate import MinimaxPlayer, play_match
from benchmark_vs_stockfish import StockfishPlayer

ckpt_path = "/opt/bumbot/checkpoints/weights_latest.pt"
data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
weights = data["weights"] if "weights" in data else data
if isinstance(weights, dict) and "linear.weight" in weights:
    weights = weights["linear.weight"].squeeze(0)
if isinstance(weights, torch.Tensor):
    weights_np = weights.detach().cpu().numpy()
else:
    weights_np = np.asarray(weights, dtype=np.float32)

bumbot = MinimaxPlayer(name="Bumbot_D3", weights=weights_np, depth=3)
sf = StockfishPlayer(name="Stockfish_L2", skill_level=2, movetime=0.03)

print("Starting Game 19: Bumbot [White] vs Stockfish Level 2 [Black] (max 70 plies)...", flush=True)
t0 = time.time()
try:
    score, reason, plies = play_match(bumbot, sf, max_plies=70)
finally:
    sf.close()

elapsed = time.time() - t0
outcome = "WIN" if score == 1.0 else ("DRAW" if score == 0.5 else "LOSS")
print(f"Result Game 19: {outcome} ({score}) | {reason} | {plies} plies | {elapsed:.1f}s", flush=True)
