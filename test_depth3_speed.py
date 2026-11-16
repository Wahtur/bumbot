import time
import feature_extractor
import search_engine
import td_leaf_trainer

w = feature_extractor.get_default_heuristic_weights().numpy()
t0 = time.time()
res = td_leaf_trainer.play_self_play_game(w, search_depth=3, max_plies=80)
t_game = time.time() - t0
print("Depth 3 Single Game:", res[3], "plies in", round(t_game, 2), "s (", round(res[3]/t_game, 1), "plies/sec )")
