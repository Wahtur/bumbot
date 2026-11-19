import sys
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import chess
import chess.engine
import numpy as np

sys.path.insert(0, "/opt/bumbot")
import feature_extractor
import search_engine
from evaluate import MinimaxPlayer, play_match
from benchmark_vs_stockfish import StockfishPlayer

def play_one_game(task):
    game_id, opp_name, skill_level, bumbot_is_white, depth, max_plies = task
    # Load weights
    data = torch.load("/opt/bumbot/checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
    weights = data["weights"] if "weights" in data else data
    if isinstance(weights, dict) and "linear.weight" in weights:
        weights = weights["linear.weight"].squeeze(0)
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    bumbot = MinimaxPlayer(name=f"Bumbot_D{depth}", weights=weights, depth=depth)
    sf = StockfishPlayer(name=opp_name, skill_level=skill_level, movetime=0.03)

    t0 = time.time()
    try:
        if bumbot_is_white:
            score, reason, plies = play_match(bumbot, sf, max_plies=max_plies)
            b_score = score
            color = "White"
        else:
            score, reason, plies = play_match(sf, bumbot, max_plies=max_plies)
            b_score = 1.0 - score
            color = "Black"
    finally:
        sf.close()

    elapsed = time.time() - t0
    return {
        "game_id": game_id,
        "opp": opp_name,
        "color": color,
        "score": b_score,
        "reason": reason,
        "plies": plies,
        "time": elapsed
    }

if __name__ == "__main__":
    tasks = [
        (1, "Stockfish_L0", 0, True, 3, 60),
        (2, "Stockfish_L0", 0, False, 3, 60),
    ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(play_one_game, t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            print(f"Done Game {res['game_id']} vs {res['opp']} [{res['color']}]: Score={res['score']}, Reason={res['reason']}, Plies={res['plies']}, Time={res['time']:.1f}s")
    print(f"Total parallel elapsed: {time.time()-t0:.1f}s")
