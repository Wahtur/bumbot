import chess
import feature_extractor
import search_engine

w = feature_extractor.get_default_heuristic_weights().numpy()

b = chess.Board("r1b1kb1r/p2np2p/2pp1ppn/8/1PQN1P2/2N1P3/P1PP2qP/R1B1K2R w KQkq - 0 14")

# Test move 1: Nxc6
b.push_san("Nxc6")
print("After 14. Nxc6:")
for m in b.legal_moves:
    b.push(m)
    val = search_engine.evaluate_board(b, w)
    b.pop()
    if b.san(m) in ["Qxh1+", "Qxc6"]:
        print(f"  Black response {b.san(m)}: leaf eval = {val:.2f}")
b.pop()

# Test move 2: Rf1
b.push_san("Rf1")
print("\nAfter 14. Rf1:")
for m in b.legal_moves:
    b.push(m)
    val = search_engine.evaluate_board(b, w)
    b.pop()
    if b.san(m) in ["Bh3", "Qxh2", "Nf7"]:
        print(f"  Black response {b.san(m)}: leaf eval = {val:.2f}")
b.pop()
