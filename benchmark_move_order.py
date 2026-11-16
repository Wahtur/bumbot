import time
import chess
import search_engine

b = chess.Board()

# Push a few moves to get a rich tactical position
b.push_san("e4")
b.push_san("e5")
b.push_san("Nf3")
b.push_san("Nc6")
b.push_san("Bc4")
b.push_san("Nf6")
b.push_san("Ng5")
b.push_san("d5")
b.push_san("exd5")
b.push_san("Na5")

t0 = time.time()
for _ in range(5000):
    mvs = search_engine.order_moves(b)
t_curr = time.time() - t0
print("Current order_moves (with gives_check):", round(5000 / t_curr), "ops/sec")

def order_moves_fast(board):
    moves = list(board.legal_moves)
    if len(moves) <= 1:
        return moves
    
    # Fast scoring without expensive gives_check make/unmake
    def fast_score(m):
        if board.is_capture(m):
            v = board.piece_at(m.to_square)
            v_val = search_engine.VICTIM_SCORES.get(v.piece_type, 0) if v else search_engine.VICTIM_SCORES[chess.PAWN]
            a = board.piece_at(m.from_square)
            a_val = search_engine.VICTIM_SCORES.get(a.piece_type, 0) if a else 0
            return 10000 + v_val * 10 - a_val
        elif m.promotion:
            return 9000
        return 0

    moves.sort(key=fast_score, reverse=True)
    return moves

t0 = time.time()
for _ in range(5000):
    mvs = order_moves_fast(b)
t_fast = time.time() - t0
print("Fast order_moves (MVV-LVA):", round(5000 / t_fast), "ops/sec (Speedup:", round(t_curr / t_fast, 2), "x)")
