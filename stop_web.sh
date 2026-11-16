#!/usr/bin/env bash
# stop_web.sh
# Stops the Bumbot web visualizer dashboard
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PID_FILE="$DIR/web.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No web.pid found. Web dashboard does not appear to be running."
    exit 0
fi

PID=$(cat "$PID_FILE")
if ps -p "$PID" > /dev/null 2>&1; then
    echo "Stopping Web Visualizer (PID $PID)..."
    kill "$PID"
    rm -f "$PID_FILE"
    echo "Web dashboard stopped."
else
    echo "Process $PID not running. Cleaning up stale PID file."
    rm -f "$PID_FILE"
fi
