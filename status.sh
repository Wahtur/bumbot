#!/usr/bin/env bash
# status.sh
# Displays current training status, PID, game count, and checkpoints
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PID_FILE="$DIR/trainer.pid"

echo "=== TD-Leaf Trainer Daemon Status ==="
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Status: RUNNING (PID: $PID)"
    else
        echo "Status: STOPPED (stale PID $PID)"
    fi
else
    echo "Status: STOPPED (not running)"
fi

CKPT_DIR="checkpoints_v2"
if [ ! -d "$CKPT_DIR" ]; then
    CKPT_DIR="checkpoints"
fi

LOG_FILE="training_log_v2.csv"
if [ ! -f "$LOG_FILE" ]; then
    LOG_FILE="training_log.csv"
fi

echo ""
echo "--- Latest Checkpoints in $CKPT_DIR/ ---"
ls -lht "$CKPT_DIR"/*.pt 2>/dev/null | head -n 5 || echo "No checkpoints found yet."

echo ""
echo "--- Recent Games from $LOG_FILE ---"
if [ -f "$LOG_FILE" ]; then
    tail -n 6 "$LOG_FILE"
else
    echo "No training log found yet."
fi

echo ""
echo "--- Recent Console Logs (logs/trainer.log) ---"
if [ -f "logs/trainer.log" ]; then
    tail -n 8 logs/trainer.log
else
    echo "No console logs yet."
fi
echo "========================================"
