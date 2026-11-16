import chess
import feature_extractor
import search_engine
from evaluate import MinimaxPlayer, GreedyPlayer, RandomPlayer, play_match

w_latest = feature_extractor.get_default_heuristic_weights().numpy()
import torch, os
if os.path.exists("checkpoints/weights_latest.pt"):
    d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
    w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

w_heur = feature_extractor.get_default_heuristic_weights().numpy()

p_latest_w = MinimaxPlayer("Latest_W", w_latest, depth=2)
p_heur_b = MinimaxPlayer("Heur_B", w_heur, depth=2)
p_heur_w = MinimaxPlayer("Heur_W", w_heur, depth=2)
p_latest_b = MinimaxPlayer("Latest_B", w_latest, depth=2)

print("--- Test 1: Latest (White) vs Heuristic (Black) ---")
res1, moves1, reason1 = play_match(p_latest_w, p_heur_b, max_plies=120)
print(f"Result: {res1} ({reason1}) in {len(moves1)} plies")

print("--- Test 2: Heuristic (White) vs Latest (Black) ---")
res2, moves2, reason2 = play_match(p_heur_w, p_latest_b, max_plies=120)
print(f"Result: {res2} ({reason2}) in {len(moves2)} plies")

# Let's inspect the final board of Test 1
b1 = chess.Board()
for m in moves1:
    b1.push(m)
print("Final board Test 1 FEN:", b1.fen())
print("White material:", sum(b1.pieces_mask(pt, chess.WHITE).bit_count() * val for pt, val in zip(feature_extractor.EVAL_PIECE_TYPES, [1,3,3,5,9])))
print("Black material:", sum(b1.pieces_mask(pt, chess.BLACK).bit_count() * val for pt, val in zip(feature_extractor.EVAL_PIECE_TYPES, [1,3,3,5,9])))

