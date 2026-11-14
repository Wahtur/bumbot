#!/usr/bin/env bash
# start_daemon.sh
# Starts continuous TD-Leaf self-play training in the background
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PID_FILE="$DIR/trainer.pid"
LOG_DIR="$DIR/logs"
TRAINER_LOG="$LOG_DIR/trainer.log"
PYTHON="$DIR/venv/bin/python3"

mkdir -p checkpoints "$LOG_DIR" plots

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[Warning] Trainer is already running with PID $PID."
        echo "Use ./status.sh to check progress, or ./stop_daemon.sh to stop it."
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

WORKERS=${1:-10}
DEPTH=${2:-2}
CHECKPOINT_EVERY=${3:-50}

echo "=========================================================="
echo " Starting Continuous TD-Leaf Training Daemon"
echo " Workers: $WORKERS | Depth: $DEPTH | Checkpoints every: $CHECKPOINT_EVERY games"
echo " Log file: $TRAINER_LOG"
echo "=========================================================="

nohup "$PYTHON" -u td_leaf_trainer.py \
    --forever \
    --workers "$WORKERS" \
    --depth "$DEPTH" \
    --checkpoint-interval "$CHECKPOINT_EVERY" \
    --save-dir checkpoints \
    --log-file training_log.csv >> "$TRAINER_LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Trainer daemon started successfully with PID: $PID"
echo "Commands to monitor:"
echo "  ./status.sh         -> Check current progress"
echo "  tail -f logs/trainer.log -> Follow live training output"
echo "  ./stop_daemon.sh    -> Gracefully stop training and save"
