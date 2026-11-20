#!/usr/bin/env bash
set -e

DIR="/opt/bumbot"
cd "$DIR"

LOG_FILE="$DIR/tournament_v2.log"
PYTHON="$DIR/venv/bin/python3"
CKPT="$DIR/checkpoints_v2/weights_latest.pt"

echo "Launching 40-Game Statistical Tournament in background..."
nohup "$PYTHON" -u benchmark_depth3_comprehensive.py \
    --checkpoint "$CKPT" \
    --games 40 \
    --depth 3 \
    --workers 6 > "$LOG_FILE" 2>&1 &

PID=$!
echo "Tournament started with PID: $PID"
echo "$PID" > "$DIR/tournament.pid"
