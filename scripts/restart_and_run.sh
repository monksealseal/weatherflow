#!/usr/bin/env bash
set -e
cd /mnt/c/Users/esima/.codex/weatherflow || exit 1
if [ -f server.pid ]; then printf 'Killing PID from server.pid: %s\n' "$(cat server.pid)"; kill "$(cat server.pid)" || true; rm -f server.pid; fi
pkill -f uvicorn || true
. .venv/bin/activate
nohup uvicorn weatherflow.server.app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 & echo $! > server.pid
sleep 3
printf '=== Uvicorn processes ===\n'
ps -ef | grep uvicorn | grep -v grep || true
printf '=== Listening sockets port 8000 ===\n'
ss -ltnp | grep 8000 || true
printf '=== Server log (last 120 lines) ===\n'
tail -n 120 server.log || true
printf '=== Running experiment ===\n'
python run_experiment.py
