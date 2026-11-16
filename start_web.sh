#!/usr/bin/env bash
# start_web.sh
# Starts the Bumbot web visualizer dashboard on port 5000
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PID_FILE="$DIR/web.pid"
LOG_FILE="$DIR/logs/web.log"
PYTHON="$DIR/venv/bin/python3"
PORT=${1:-5000}

mkdir -p "$DIR/logs"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[Warning] Web visualizer is already running with PID $PID on port $PORT."
        echo "Access at: http://192.168.1.135:$PORT"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

echo "Starting Bumbot Web Visualizer on port $PORT..."
nohup "$PYTHON" web_server.py --port "$PORT" --host "0.0.0.0" >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Web dashboard started successfully with PID: $PID"
echo "Open in your browser from any local network device at:"
echo "    http://192.168.1.135:$PORT"
