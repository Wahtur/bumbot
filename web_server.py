"""
web_server.py
-------------
Interactive Local Network Dashboard & Generation Arena for TD-Leaf Chess Bot.
Supports both V2 (808-dim Tapered GLM) and V1 (651-dim Classical) checkpoint lineages.

Features:
  - Generation Matchmaker: Match any two checkpoints (cross-version V2 vs V1 or vs Baselines)
    and simulate matches in real time.
  - Interactive Chessboard: Move scrubber, play/pause auto-play, clickable move list, and eval bar.
  - Live Telemetry: Shows active training daemon status, game count, and real-time training curves.
  - Live Games Feed: Real-time table of recent training games with outcomes and termination reasons.
"""

import os
import glob
import time
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import chess
import chess.pgn
from flask import Flask, render_template, jsonify, request, send_from_directory

import feature_extractor
import search_engine
from evaluate import RandomPlayer, GreedyPlayer, MinimaxPlayer

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR_V2 = os.path.join(BASE_DIR, "checkpoints_v2")
CHECKPOINT_DIR_V1 = os.path.join(BASE_DIR, "checkpoints")

LOG_CSV_V2 = os.path.join(BASE_DIR, "training_log_v2.csv")
LOG_CSV_V1 = os.path.join(BASE_DIR, "training_log.csv")

PID_FILE = os.path.join(BASE_DIR, "trainer.pid")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def get_available_generations() -> List[Dict]:
    """
    Scans both V2 and V1 checkpoint directories and returns available model generations,
    ordered newest V2 first, followed by V1 checkpoints, then baseline bots.
    """
    v2_checkpoints = []
    v1_checkpoints = []
    latest_v2_game = None
    latest_v1_game = None

    # Scan V2 Checkpoints
    if os.path.exists(CHECKPOINT_DIR_V2):
        latest_path_v2 = os.path.join(CHECKPOINT_DIR_V2, "weights_latest.pt")
        if os.path.exists(latest_path_v2):
            try:
                d = torch.load(latest_path_v2, map_location="cpu", weights_only=False)
                latest_v2_game = d.get("game", None)
            except Exception:
                pass

        files = glob.glob(os.path.join(CHECKPOINT_DIR_V2, "weights_game_*.pt"))
        for f in files:
            basename = os.path.basename(f)
            try:
                num_str = basename.replace("weights_game_", "").replace(".pt", "")
                game_num = int(num_str)
                v2_checkpoints.append({
                    "id": f"v2_game_{game_num}",
                    "name": f"⚡ V2 Gen {game_num} (Game {game_num})",
                    "game": game_num,
                    "path": f,
                    "version": "v2",
                    "type": "checkpoint"
                })
            except ValueError:
                pass

    # Scan V1 Checkpoints
    if os.path.exists(CHECKPOINT_DIR_V1):
        latest_path_v1 = os.path.join(CHECKPOINT_DIR_V1, "weights_latest.pt")
        if os.path.exists(latest_path_v1):
            try:
                d = torch.load(latest_path_v1, map_location="cpu", weights_only=False)
                latest_v1_game = d.get("game", None)
            except Exception:
                pass

        files = glob.glob(os.path.join(CHECKPOINT_DIR_V1, "weights_game_*.pt"))
        for f in files:
            basename = os.path.basename(f)
            try:
                num_str = basename.replace("weights_game_", "").replace(".pt", "")
                game_num = int(num_str)
                v1_checkpoints.append({
                    "id": f"v1_game_{game_num}",
                    "name": f"📦 V1 Gen {game_num} (Game {game_num})",
                    "game": game_num,
                    "path": f,
                    "version": "v1",
                    "type": "checkpoint"
                })
            except ValueError:
                pass

    # Sort each descending
    v2_checkpoints.sort(key=lambda x: x["game"], reverse=True)
    v1_checkpoints.sort(key=lambda x: x["game"], reverse=True)

    gens = []

    # 1. Add V2 Latest at very top
    if os.path.exists(os.path.join(CHECKPOINT_DIR_V2, "weights_latest.pt")):
        latest_label = f"🔥 V2 Latest Active (Game {latest_v2_game})" if latest_v2_game else "🔥 V2 Latest Active Generation"
        gens.append({
            "id": "v2_latest",
            "name": latest_label,
            "game": latest_v2_game or 999999,
            "path": os.path.join(CHECKPOINT_DIR_V2, "weights_latest.pt"),
            "version": "v2",
            "type": "latest"
        })

    # 2. Add V2 checkpoints
    gens.extend(v2_checkpoints)

    # 3. Add V1 Latest & V1 checkpoints
    if os.path.exists(os.path.join(CHECKPOINT_DIR_V1, "weights_latest.pt")):
        v1_latest_label = f"📦 V1 Final Checkpoint (Game {latest_v1_game})" if latest_v1_game else "📦 V1 Final Checkpoint"
        gens.append({
            "id": "v1_latest",
            "name": v1_latest_label,
            "game": latest_v1_game or 13444,
            "path": os.path.join(CHECKPOINT_DIR_V1, "weights_latest.pt"),
            "version": "v1",
            "type": "checkpoint"
        })
    gens.extend(v1_checkpoints)

    # 4. Add baselines
    gens.extend([
        {"id": "heuristic", "name": "Generation 0 (Heuristic Baseline)", "game": 0, "type": "baseline"},
        {"id": "greedy", "name": "GreedyBot (1-ply Material Capture)", "game": 0, "type": "bot"},
        {"id": "random", "name": "RandomBot (Baseline Elo ~100)", "game": 0, "type": "bot"},
    ])

    return gens


def load_weights_for_id(gen_id: str) -> Tuple[Optional[np.ndarray], str]:
    """
    Loads weight vector as numpy array and returns (weights, display_name) for given generation ID.
    Supports V2 (808-dim), V1 (651-dim), and baselines.
    """
    if gen_id == "heuristic":
        return feature_extractor.get_default_heuristic_weights().numpy(), "Gen 0 (Heuristic)"

    # Alias "latest" to "v2_latest"
    if gen_id in ("latest", "v2_latest"):
        latest_path = os.path.join(CHECKPOINT_DIR_V2, "weights_latest.pt")
        if os.path.exists(latest_path):
            d = torch.load(latest_path, map_location="cpu", weights_only=False)
            w = d["weights"]
            g = d.get("game", "Latest")
            w_np = w.numpy() if isinstance(w, torch.Tensor) else np.array(w)
            return w_np, f"V2 Latest (Game {g})"
        # Fallback to V1 latest
        latest_path_v1 = os.path.join(CHECKPOINT_DIR_V1, "weights_latest.pt")
        if os.path.exists(latest_path_v1):
            d = torch.load(latest_path_v1, map_location="cpu", weights_only=False)
            w = d["weights"]
            g = d.get("game", "Latest")
            w_np = w.numpy() if isinstance(w, torch.Tensor) else np.array(w)
            return w_np, f"V1 Latest (Game {g})"
        return feature_extractor.get_default_heuristic_weights().numpy(), "Heuristic (Fallback)"

    if gen_id == "v1_latest":
        p = os.path.join(CHECKPOINT_DIR_V1, "weights_latest.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=False)
            w = d["weights"]
            g = d.get("game", "V1")
            w_np = w.numpy() if isinstance(w, torch.Tensor) else np.array(w)
            return w_np, f"V1 Gen {g}"

    # V2 Game checkpoint
    if gen_id.startswith("v2_game_"):
        try:
            num = int(gen_id.replace("v2_game_", ""))
            # Try 4-digit and raw formatting
            for pat in [f"weights_game_{num:04d}.pt", f"weights_game_{num}.pt"]:
                p = os.path.join(CHECKPOINT_DIR_V2, pat)
                if os.path.exists(p):
                    d = torch.load(p, map_location="cpu", weights_only=False)
                    w = d["weights"]
                    w_np = w.numpy() if isinstance(w, torch.Tensor) else np.array(w)
                    return w_np, f"V2 Gen {num}"
        except Exception as e:
            print(f"[Error] load_weights_for_id failed for {gen_id}: {e}")

    # V1 Game checkpoint
    if gen_id.startswith("v1_game_"):
        try:
            num = int(gen_id.replace("v1_game_", ""))
            for pat in [f"weights_game_{num:04d}.pt", f"weights_game_{num}.pt", f"weights_game_{num:05d}.pt"]:
                p = os.path.join(CHECKPOINT_DIR_V1, pat)
                if os.path.exists(p):
                    d = torch.load(p, map_location="cpu", weights_only=False)
                    w = d["weights"]
                    w_np = w.numpy() if isinstance(w, torch.Tensor) else np.array(w)
                    return w_np, f"V1 Gen {num}"
        except Exception as e:
            print(f"[Error] load_weights_for_id failed for {gen_id}: {e}")

    # Legacy game_ prefix: check V2 first, then V1
    if gen_id.startswith("game_"):
        try:
            num = int(gen_id.replace("game_", ""))
            for d_dir, prefix, tag in [(CHECKPOINT_DIR_V2, "weights_game_", "V2"), (CHECKPOINT_DIR_V1, "weights_game_", "V1")]:
                for pat in [f"{prefix}{num:04d}.pt", f"{prefix}{num}.pt"]:
                    p = os.path.join(d_dir, pat)
                    if os.path.exists(p):
                        d = torch.load(p, map_location="cpu", weights_only=False)
                        w = d["weights"]
                        w_np = w.numpy() if isinstance(w, torch.Tensor) else np.array(w)
                        return w_np, f"{tag} Gen {num}"
        except Exception:
            pass

    return feature_extractor.get_default_heuristic_weights().numpy(), f"Unknown ({gen_id})"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """
    Returns live trainer status, game count, throughput metrics, checkpoints,
    and a real-time list of recent games. Supports ?version=v2 or ?version=v1.
    """
    version = request.args.get("version", "v2").lower()
    if version not in ("v1", "v2"):
        version = "v2"

    is_running = False
    pid = None
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            if os.name == "nt":
                is_running = True
            else:
                import subprocess
                res = subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL)
                is_running = (res.returncode == 0)
        except Exception:
            is_running = False

    active_log = LOG_CSV_V2 if version == "v2" else LOG_CSV_V1

    total_games = 0
    latest_loss = 0.0
    latest_norm = 0.0
    latest_outcome = "N/A"
    games_per_sec = 0.0
    games_per_min = 0.0
    plies_per_sec = 0.0
    recent_games = []

    if os.path.exists(active_log):
        try:
            with open(active_log, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) > 1:
                total_games = len(lines) - 1
                parts = lines[-1].split(",")
                if len(parts) >= 6:
                    # game,plies,outcome,reason,loss,weight_norm,time_sec
                    latest_outcome = f"{parts[2]} ({parts[3]})"
                    latest_loss = float(parts[4])
                    latest_norm = float(parts[5])

                # Parse recent games (last 25 games, newest first)
                raw_recent = lines[max(1, len(lines) - 25):]
                for r in reversed(raw_recent):
                    p = r.split(",")
                    if len(p) >= 6:
                        try:
                            recent_games.append({
                                "game": int(p[0]),
                                "plies": int(p[1]),
                                "outcome": float(p[2]),
                                "reason": p[3],
                                "loss": float(p[4]),
                                "weight_norm": float(p[5]),
                                "time_sec": round(float(p[6]), 2) if len(p) > 6 else 0.0
                            })
                        except (ValueError, IndexError):
                            pass

                # Throughput from recent games (last 50)
                recent_lines = lines[-min(50, len(lines) - 1):]
                times = []
                plies = []
                for r in recent_lines:
                    p = r.split(",")
                    if len(p) >= 7:
                        try:
                            t_val = float(p[6])
                            pl_val = float(p[1])
                            if t_val > 0:
                                times.append(t_val)
                                plies.append(pl_val)
                        except ValueError:
                            pass
                if times:
                    avg_time = sum(times) / len(times)
                    if avg_time > 0:
                        games_per_sec = round(1.0 / avg_time, 2)
                        games_per_min = round(games_per_sec * 60.0, 1)
                        avg_ply = sum(plies) / len(plies)
                        plies_per_sec = round(avg_ply * games_per_sec, 1)
        except Exception as e:
            print(f"[Error] reading log {active_log}: {e}")

    # Total counts for both versions
    v2_count = 0
    if os.path.exists(LOG_CSV_V2):
        try:
            with open(LOG_CSV_V2, "r") as f:
                v2_count = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

    v1_count = 0
    if os.path.exists(LOG_CSV_V1):
        try:
            with open(LOG_CSV_V1, "r") as f:
                v1_count = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

    gens = get_available_generations()
    return jsonify({
        "version": version,
        "daemon_running": is_running,
        "pid": pid,
        "total_games": total_games,
        "v2_games_count": v2_count,
        "v1_games_count": v1_count,
        "latest_loss": latest_loss,
        "latest_norm": latest_norm,
        "latest_outcome": latest_outcome,
        "games_per_sec": games_per_sec,
        "games_per_min": games_per_min,
        "plies_per_sec": plies_per_sec,
        "generations": gens,
        "recent_games": recent_games,
    })


@app.route("/api/plots/<path:filename>")
def api_plot(filename):
    """
    Serves generated plot images, automatically re-generating them if stale (>10s old).
    Supports ?version=v2 or ?version=v1.
    """
    version = request.args.get("version", "v2").lower()
    if version not in ("v1", "v2"):
        version = "v2"

    active_log = LOG_CSV_V2 if version == "v2" else LOG_CSV_V1
    active_ckpt_dir = CHECKPOINT_DIR_V2 if version == "v2" else CHECKPOINT_DIR_V1

    # Map target filename to version-specific filename
    base, ext = os.path.splitext(filename)
    if not (base.endswith("_v1") or base.endswith("_v2")):
        versioned_filename = f"{base}_{version}{ext}"
    else:
        versioned_filename = filename

    filepath = os.path.join(PLOTS_DIR, versioned_filename)
    now = time.time()
    
    is_stale = (not os.path.exists(filepath)) or (now - os.path.getmtime(filepath) > 10)
    if is_stale and os.path.exists(active_log):
        try:
            import plot_results
            latest_ckpt = os.path.join(active_ckpt_dir, "weights_latest.pt")
            curves_path = os.path.join(PLOTS_DIR, f"training_curves_{version}.png")
            weights_path = os.path.join(PLOTS_DIR, f"weight_interpretation_{version}.png")

            plot_results.plot_training_telemetry(log_csv=active_log, save_path=curves_path)
            if os.path.exists(latest_ckpt):
                plot_results.plot_weight_interpretation(checkpoint_path=latest_ckpt, save_path=weights_path)
        except Exception as e:
            print(f"[Warning] Failed to auto-regenerate plot: {e}")

    if not os.path.exists(filepath):
        # Fallback to plain filename
        fallback_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(fallback_path):
            return send_from_directory(PLOTS_DIR, filename)
            
    return send_from_directory(PLOTS_DIR, versioned_filename)


@app.route("/api/simulate_match", methods=["POST"])
def api_simulate_match():
    """
    Simulates a match between two chosen generations / bots.
    Supports cross-version matches (e.g. V2 Gen 1200 vs V1 Gen 13400).
    Returns move list, FEN list, evaluations, and termination reason.
    """
    data = request.get_json(silent=True) or {}
    white_id = data.get("white_id", "v2_latest")
    black_id = data.get("black_id", "heuristic")
    depth = int(data.get("depth", 2))
    max_plies = min(250, int(data.get("max_plies", 200)))

    # White player
    if white_id == "random":
        p_white = RandomPlayer(name="White: RandomBot")
    elif white_id == "greedy":
        p_white = GreedyPlayer(name="White: GreedyBot")
    else:
        w, label = load_weights_for_id(white_id)
        p_white = MinimaxPlayer(name=f"White: {label}", weights=w, depth=depth)

    # Black player
    if black_id == "random":
        p_black = RandomPlayer(name="Black: RandomBot")
    elif black_id == "greedy":
        p_black = GreedyPlayer(name="Black: GreedyBot")
    else:
        w, label = load_weights_for_id(black_id)
        p_black = MinimaxPlayer(name=f"Black: {label}", weights=w, depth=depth)

    board = chess.Board()
    fens = [board.fen()]
    moves_san = []
    moves_uci = []
    evals = [0.0]

    ply = 0
    t0 = time.time()
    
    while not board.is_game_over() and ply < max_plies:
        curr_player = p_white if board.turn == chess.WHITE else p_black
        move = curr_player.get_move(board)
        san = board.san(move)
        uci = move.uci()
        board.push(move)
        
        # Calculate evaluation from White's perspective
        h_weights = feature_extractor.get_default_heuristic_weights().numpy()
        score = search_engine.evaluate_board(board, h_weights)

        moves_san.append(san)
        moves_uci.append(uci)
        fens.append(board.fen())
        evals.append(round(float(score), 2))
        ply += 1

    elapsed = time.time() - t0

    if board.is_checkmate():
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        reason = "Checkmate"
    elif board.is_stalemate():
        result = "1/2-1/2"
        reason = "Stalemate"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        reason = "Insufficient Material"
    elif board.is_fifty_moves():
        result = "1/2-1/2"
        reason = "50-Move Rule"
    elif board.is_repetition(3):
        result = "1/2-1/2"
        reason = "Threefold Repetition"
    else:
        result = "1/2-1/2"
        reason = f"Max Plies ({max_plies}) Reached"

    return jsonify({
        "white": p_white.name,
        "black": p_black.name,
        "result": result,
        "reason": reason,
        "total_plies": ply,
        "time_sec": round(elapsed, 2),
        "fens": fens,
        "moves_san": moves_san,
        "moves_uci": moves_uci,
        "evals": evals,
    })


@app.route("/api/bot_move", methods=["POST"])
def api_bot_move():
    """
    Computes a single response move for the selected bot given a board FEN.
    Supports both V2 and V1 bot models.
    """
    data = request.get_json(silent=True) or {}
    fen = data.get("fen", chess.STARTING_FEN)
    bot_id = data.get("bot_id", "v2_latest")
    depth = max(1, min(4, int(data.get("depth", 2))))

    try:
        board = chess.Board(fen)
    except Exception as e:
        return jsonify({"error": f"Invalid FEN: {e}"}), 400

    if board.is_game_over():
        h_weights = feature_extractor.get_default_heuristic_weights().numpy()
        score = round(float(search_engine.evaluate_board(board, h_weights)), 2)
        return jsonify({
            "game_over": True,
            "result": "1-0" if (board.is_checkmate() and board.turn == chess.BLACK) else ("0-1" if board.is_checkmate() else "1/2-1/2"),
            "reason": "Checkmate" if board.is_checkmate() else "Draw",
            "fen": board.fen(),
            "eval": score
        })

    t0 = time.time()
    nodes = 1
    if bot_id == "random":
        move = random.choice(list(board.legal_moves))
    elif bot_id == "greedy":
        player = GreedyPlayer()
        move = player.get_move(board)
        nodes = len(list(board.legal_moves))
    else:
        weights, _ = load_weights_for_id(bot_id)
        move, eval_val, leaf, nodes = search_engine.get_best_move(board, weights, depth=depth)

    if move is None:
        legal = list(board.legal_moves)
        if legal:
            move = legal[0]
        else:
            return jsonify({"game_over": True, "reason": "No legal moves available"})

    san = board.san(move)
    uci = move.uci()
    board.push(move)
    elapsed = time.time() - t0

    # Calculate evaluation from White's perspective
    h_weights = feature_extractor.get_default_heuristic_weights().numpy()
    eval_score = round(float(search_engine.evaluate_board(board, h_weights)), 2)

    game_over = board.is_game_over()
    result = None
    reason = None
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
        reason = "Checkmate"
    elif board.is_stalemate():
        result = "1/2-1/2"
        reason = "Stalemate"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        reason = "Insufficient Material"
    elif board.is_fifty_moves():
        result = "1/2-1/2"
        reason = "50-Move Rule"
    elif board.is_repetition(3):
        result = "1/2-1/2"
        reason = "Threefold Repetition"

    return jsonify({
        "bot_move_san": san,
        "bot_move_uci": uci,
        "from": uci[:2],
        "to": uci[2:4],
        "fen": board.fen(),
        "eval": eval_score,
        "game_over": game_over,
        "result": result,
        "reason": reason,
        "nodes": nodes,
        "time_sec": round(elapsed, 3),
        "is_check": board.is_check()
    })


@app.route("/api/evaluate_position", methods=["POST"])
def api_evaluate_position():
    """Returns evaluation score and game state for a given FEN."""
    data = request.get_json(silent=True) or {}
    fen = data.get("fen", chess.STARTING_FEN)
    try:
        board = chess.Board(fen)
        h_weights = feature_extractor.get_default_heuristic_weights().numpy()
        score = round(float(search_engine.evaluate_board(board, h_weights)), 2)
        return jsonify({
            "eval": score,
            "is_game_over": board.is_game_over(),
            "is_check": board.is_check(),
            "turn": "white" if board.turn == chess.WHITE else "black"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bumbot Web Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (0.0.0.0 for LAN)")
    args = parser.parse_args()

    print(f"=== Bumbot Web Visualizer Running ===")
    print(f"Access from your workstation browser at: http://192.168.1.135:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
