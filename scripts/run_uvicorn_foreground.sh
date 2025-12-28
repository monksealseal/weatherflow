#!/usr/bin/env bash
set -e
cd /mnt/c/Users/esima/.codex/weatherflow || exit 1
pkill -f uvicorn || true
rm -f server.pid || true
. .venv/bin/activate
exec uvicorn weatherflow.server.app:app --host 0.0.0.0 --port 8000
