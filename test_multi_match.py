import chess
import feature_extractor
import search_engine
from evaluate import MinimaxPlayer, play_match
import torch, os

w_latest = feature_extractor.get_default_heuristic_weights().numpy()
if os.path.exists("checkpoints/weights_latest.pt"):
    d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
    w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

w_heur = feature_extractor.get_default_heuristic_weights().numpy()

p_latest_w = MinimaxPlayer("Latest_W", w_latest, depth=2)
p_heur_b = MinimaxPlayer("Heur_B", w_heur, depth=2)
p_heur_w = MinimaxPlayer("Heur_W", w_heur, depth=2)
p_latest_b = MinimaxPlayer("Latest_B", w_latest, depth=2)

print("=== 5 Matches: White=Latest vs Black=Heuristic ===")
w_wins = 0
b_wins = 0
draws = 0
for i in range(5):
    res, reason, plies = play_match(p_latest_w, p_heur_b, max_plies=100)
    print(f"Game {i+1}: Result={res} ({reason}) in {plies} plies")
    if res == 1.0: w_wins += 1
    elif res == 0.0: b_wins += 1
    else: draws += 1
print(f"Summary White=Latest: {w_wins}W - {b_wins}L - {draws}D")

print("\n=== 5 Matches: White=Heuristic vs Black=Latest ===")
w_wins = 0
b_wins = 0
draws = 0
for i in range(5):
    res, reason, plies = play_match(p_heur_w, p_latest_b, max_plies=100)
    print(f"Game {i+1}: Result={res} ({reason}) in {plies} plies")
    if res == 1.0: w_wins += 1
    elif res == 0.0: b_wins += 1
    else: draws += 1
print(f"Summary Black=Latest: {b_wins}W - {w_wins}L - {draws}D")
