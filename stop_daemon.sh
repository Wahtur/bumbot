#!/usr/bin/env bash
# stop_daemon.sh
# Gracefully stops the background TD-Leaf training daemon
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PID_FILE="$DIR/trainer.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No trainer.pid file found. Trainer daemon does not appear to be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Process $PID is not running. Cleaning up stale PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Sending graceful stop signal (SIGTERM) to trainer PID $PID..."
kill -TERM "$PID"

echo "Waiting for current batch to complete and checkpoint to be saved..."
while ps -p "$PID" > /dev/null 2>&1; do
    sleep 1
    printf "."
done

rm -f "$PID_FILE"
echo ""
echo "Trainer daemon stopped cleanly! Checkpoints updated."
