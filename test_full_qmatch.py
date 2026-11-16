import chess
import feature_extractor
import search_engine
import evaluate
import torch, os

# Monkey-patch quiescence into search_engine
def quiescence(board, weights, alpha, beta, is_maximizing, max_qdepth=3, node_counter=None):
    if node_counter is not None:
        node_counter[0] += 1
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
            score = quiescence(board, weights, alpha, beta, False, max_qdepth - 1, node_counter)
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
            score = quiescence(board, weights, alpha, beta, True, max_qdepth - 1, node_counter)
            board.pop()
            if score <= alpha:
                return alpha
            beta = min(beta, score)
        return beta

# Patch alpha_beta
orig_alpha_beta = search_engine.alpha_beta
def alpha_beta_with_q(board, weights, depth, alpha, beta, is_maximizing, node_counter=None):
    if node_counter is not None:
        node_counter[0] += 1
    if board.is_checkmate():
        val = -(search_engine.MATE_SCORE + depth) if is_maximizing else (search_engine.MATE_SCORE + depth)
        return val, None, board.copy()
    if (board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3)):
        return search_engine.DRAW_SCORE, None, board.copy()
    if depth <= 0:
        val = quiescence(board, weights, alpha, beta, is_maximizing, max_qdepth=3, node_counter=node_counter)
        return val, None, board.copy()
    ordered_moves = search_engine.order_moves(board)
    if not ordered_moves:
        val = -(search_engine.MATE_SCORE + depth) if is_maximizing else (search_engine.MATE_SCORE + depth)
        return val, None, board.copy()
    best_move = None
    best_leaf = board
    if is_maximizing:
        max_eval = -float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score, _, leaf = alpha_beta_with_q(board, weights, depth - 1, alpha, beta, False, node_counter)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                best_leaf = leaf
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move, best_leaf
    else:
        min_eval = float("inf")
        for move in ordered_moves:
            board.push(move)
            eval_score, _, leaf = alpha_beta_with_q(board, weights, depth - 1, alpha, beta, True, node_counter)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                best_leaf = leaf
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move, best_leaf

search_engine.alpha_beta = alpha_beta_with_q

w_latest = feature_extractor.get_default_heuristic_weights().numpy()
if os.path.exists("checkpoints/weights_latest.pt"):
    d = torch.load("checkpoints/weights_latest.pt", map_location="cpu", weights_only=False)
    w_latest = d["weights"].numpy() if isinstance(d["weights"], torch.Tensor) else d["weights"]

w_heur = feature_extractor.get_default_heuristic_weights().numpy()

p_latest_w = evaluate.MinimaxPlayer("Latest_W", w_latest, depth=2)
p_heur_b = evaluate.MinimaxPlayer("Heur_B", w_heur, depth=2)

board = chess.Board()
moves = []
ply = 0
while not board.is_game_over() and ply < 50:
    curr = p_latest_w if board.turn == chess.WHITE else p_heur_b
    mv = curr.get_move(board)
    san = board.san(mv)
    board.push(mv)
    moves.append(san)
    ply += 1

print(f"Match completed: {ply} plies")
print("Moves:", " ".join(moves[:20]))
print("Checkmate?", board.is_checkmate())
if board.is_checkmate():
    print("Winner:", "White" if board.turn == chess.BLACK else "Black")
else:
    # Final eval from White's perspective
    sc = search_engine.evaluate_board(board, w_heur)
    print("Final eval (White perspective):", f"{sc:+.2f}")
