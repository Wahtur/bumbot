import chess
import feature_extractor
import search_engine

w = feature_extractor.get_default_heuristic_weights().numpy()

b = chess.Board()
b.push_san("c4")
b.push_san("e6")
b.push_san("Nf3")
b.push_san("Qf6")

def quiescence(board, weights, alpha, beta, is_maximizing, max_qdepth=4):
    stand_pat = search_engine.evaluate_board(board, weights)
    if is_maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        if max_qdepth <= 0:
            return alpha
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if not captures:
            return alpha
        captures.sort(key=lambda m: search_engine.score_move(board, m), reverse=True)
        for move in captures:
            board.push(move)
            score = quiescence(board, weights, alpha, beta, False, max_qdepth - 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
        if max_qdepth <= 0:
            return beta
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if not captures:
            return beta
        captures.sort(key=lambda m: search_engine.score_move(board, m), reverse=True)
        for move in captures:
            board.push(move)
            score = quiescence(board, weights, alpha, beta, True, max_qdepth - 1)
            board.pop()
            if score <= alpha:
                return alpha
            beta = min(beta, score)
        return beta

# Search depth 2 WITH quiescence
best_m = None
max_sc = -float("inf")
for mv in search_engine.order_moves(b):
    b.push(mv)
    # 1-ply response for black
    min_sc = float("inf")
    for b_mv in search_engine.order_moves(b):
        b.push(b_mv)
        leaf_val = quiescence(b, w, -float("inf"), float("inf"), is_maximizing=True)
        b.pop()
        min_sc = min(min_sc, leaf_val)
    b.pop()
    if min_sc > max_sc:
        max_sc = min_sc
        best_m = mv

print(f"Depth 2 WITH Quiescence Search chose: {b.san(best_m)} with eval: {max_sc:.2f}")
