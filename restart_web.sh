#!/usr/bin/env bash
# restart_web.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

pkill -f 'web_server.*5001' 2>/dev/null || true
if [ -f "web.pid" ]; then
    PID=$(cat web.pid)
    kill "$PID" 2>/dev/null || true
    rm -f web.pid
fi

PORT=5000
if ss -tulpn | grep -q ":5000 "; then
    echo "[Info] Port 5000 is currently occupied. Starting on port 5001."
    PORT=5001
else
    echo "[Info] Port 5000 is free! Starting on port 5000."
fi

nohup ./venv/bin/python3 web_server.py --port "$PORT" --host "0.0.0.0" >> logs/web.log 2>&1 &
echo $! > web.pid
echo "Web dashboard running at: http://192.168.1.135:$PORT"
