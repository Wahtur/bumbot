import sys
import time
import torch
import chess
import chess.engine
sys.path.insert(0, "/opt/bumbot")
import feature_extractor
import search_engine
from evaluate import MinimaxPlayer, play_match
from benchmark_vs_stockfish import StockfishPlayer

data = torch.load("/opt/bumbot/checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
weights = data["weights"] if "weights" in data else data
if isinstance(weights, dict) and "linear.weight" in weights:
    weights = weights["linear.weight"].squeeze(0)
if isinstance(weights, torch.Tensor):
    weights = weights.detach().cpu().numpy()

bumbot = MinimaxPlayer(name="Bumbot_D3", weights=weights, depth=3)
sf = StockfishPlayer(name="Stockfish_L0", skill_level=0, movetime=0.03)

t0 = time.time()
score, reason, plies = play_match(bumbot, sf, max_plies=100)
elapsed = time.time() - t0
print(f"Game 1 (White): Score={score}, Reason={reason}, Plies={plies}, Time={elapsed:.2f}s ({elapsed/plies:.3f}s/ply)")
sf.close()
