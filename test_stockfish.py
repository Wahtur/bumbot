import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
print("Stockfish connected successfully!")
engine.configure({"Skill Level": 0})
board = chess.Board()
result = engine.play(board, chess.engine.Limit(time=0.05))
print("Stockfish Move 1:", board.san(result.move))
engine.quit()
print("Stockfish disconnected cleanly.")
