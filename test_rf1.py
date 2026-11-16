import chess
import feature_extractor
import search_engine

w = feature_extractor.get_default_heuristic_weights().numpy()
b = chess.Board("r1b1kb1r/p2np2p/2pp1ppn/8/1PQN1P2/2N1P3/P1PP2qP/R1B1K2R w KQkq - 0 14")
b.push_san("Rf1")

black_moves = []
for m in b.legal_moves:
    san = b.san(m)
    b.push(m)
    val = search_engine.evaluate_board(b, w)
    b.pop()
    black_moves.append((san, val))

black_moves.sort(key=lambda x: x[1])  # Black minimizes eval
print("Top 5 Black responses after 14. Rf1:")
for san, val in black_moves[:5]:
    print(f"  {san}: eval={val:.2f}")
