import time
import chess
import numpy as np
import feature_extractor
import search_engine
import td_leaf_trainer
import benchmark_opt
import benchmark_move_order

w = feature_extractor.get_default_heuristic_weights().numpy()

# Test game before
t0 = time.time()
res1 = td_leaf_trainer.play_self_play_game(w, search_depth=2, max_plies=100)
t_orig = time.time() - t0
print(f"Current self_play game (100 plies): {t_orig:.2f}s ({res1[3] / t_orig:.1f} plies/sec)")

# Apply fast extract_features and fast order_moves
feature_extractor.extract_features_numpy = benchmark_opt.extract_features_fast
search_engine.order_moves = benchmark_move_order.order_moves_fast

t0 = time.time()
res2 = td_leaf_trainer.play_self_play_game(w, search_depth=2, max_plies=100)
t_opt = time.time() - t0
print(f"Optimized self_play game (100 plies): {t_opt:.2f}s ({res2[3] / t_opt:.1f} plies/sec) -> Speedup: {t_orig / t_opt:.2f}x")
