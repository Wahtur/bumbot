import chess
b = chess.Board()
print("Attack methods:", [m for m in dir(b) if 'attack' in m])
print("Attacks on E4:", b.attacks(chess.E4), type(b.attacks(chess.E4)))
print("Attacks mask on E4:", b.attacks_mask(chess.E4), type(b.attacks_mask(chess.E4)))
