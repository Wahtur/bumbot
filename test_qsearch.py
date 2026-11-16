import chess
import feature_extractor
import search_engine

w = feature_extractor.get_default_heuristic_weights().numpy()
b = chess.Board("r1b1kb1r/p2np2p/2pp1ppn/8/1PQN1P2/2N1P3/P1PP2qP/R1B1K2R w KQkq - 0 14")

def quiescence(board, weights, alpha, beta, is_maximizing, qdepth=4):
    stand_pat = search_engine.evaluate_board(board, weights)
    if is_maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        if qdepth <= 0:
            return alpha
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        captures.sort(key=lambda m: search_engine.score_move(board, m), reverse=True)
        for m in captures:
            board.push(m)
            sc = quiescence(board, weights, alpha, beta, False, qdepth - 1)
            board.pop()
            if sc >= beta:
                return beta
            alpha = max(alpha, sc)
        return alpha
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
        if qdepth <= 0:
            return beta
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        captures.sort(key=lambda m: search_engine.score_move(board, m), reverse=True)
        for m in captures:
            board.push(m)
            sc = quiescence(board, weights, alpha, beta, True, qdepth - 1)
            board.pop()
            if sc <= alpha:
                return alpha
            beta = min(beta, sc)
        return beta

# Evaluate White's candidate moves WITH quiescence search!
ordered = search_engine.order_moves(b)
print("Evaluating candidate moves WITH Quiescence Search:")
scores_q = []
for mv in ordered:
    b.push(mv)
    # Search Black reply with quiescence
    min_sc = float("inf")
    for b_mv in b.legal_moves:
        b.push(b_mv)
        val = quiescence(b, w, -float("inf"), float("inf"), is_maximizing=True)
        b.pop()
        min_sc = min(min_sc, val)
    b.pop()
    scores_q.append((mv, min_sc))

scores_q.sort(key=lambda x: x[1], reverse=True)
for mv, sc in scores_q[:5]:
    print(f"  Move {mv.uci()} ({b.san(mv)}): eval={sc:.2f}")
