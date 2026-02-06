#!/usr/bin/env bash
# WeatherFlow Development Server Launcher
#
# Starts development servers for local development.
#
# Usage:
#   ./scripts/dev.sh              # Show options
#   ./scripts/dev.sh backend      # Start FastAPI backend (port 8000)
#   ./scripts/dev.sh frontend     # Start Vite dev server (port 5173)
#   ./scripts/dev.sh streamlit    # Start Streamlit app (port 8501)
#   ./scripts/dev.sh all          # Start backend + frontend concurrently

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-5173}
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}

cleanup() {
    echo ""
    echo -e "${BLUE}Shutting down servers...${RESET}"
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo -e "${GREEN}Done.${RESET}"
}

start_backend() {
    echo -e "${BLUE}Starting FastAPI backend on port ${BACKEND_PORT}...${RESET}"
    cd "$ROOT_DIR"
    python -m uvicorn weatherflow.server.app:app \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --reload \
        --reload-dir weatherflow
}

start_frontend() {
    echo -e "${BLUE}Starting Vite dev server on port ${FRONTEND_PORT}...${RESET}"
    cd "$ROOT_DIR/frontend"
    npm run dev -- --port "$FRONTEND_PORT"
}

start_streamlit() {
    echo -e "${BLUE}Starting Streamlit on port ${STREAMLIT_PORT}...${RESET}"
    cd "$ROOT_DIR"
    streamlit run streamlit_app/Home.py --server.port "$STREAMLIT_PORT"
}

start_all() {
    trap cleanup EXIT INT TERM

    echo -e "${BOLD}Starting all development servers...${RESET}"
    echo -e "  Backend:  http://localhost:${BACKEND_PORT}"
    echo -e "  Frontend: http://localhost:${FRONTEND_PORT}"
    echo ""

    start_backend &
    sleep 2
    start_frontend &

    echo -e "${GREEN}All servers running. Press Ctrl+C to stop.${RESET}"
    wait
}

show_help() {
    echo -e "${BOLD}WeatherFlow Development Servers${RESET}"
    echo "================================"
    echo ""
    echo "Usage: ./scripts/dev.sh [target]"
    echo ""
    echo "Targets:"
    echo "  backend     Start FastAPI backend (port ${BACKEND_PORT})"
    echo "  frontend    Start Vite dev server (port ${FRONTEND_PORT})"
    echo "  streamlit   Start Streamlit app (port ${STREAMLIT_PORT})"
    echo "  all         Start backend + frontend concurrently"
    echo ""
    echo "Environment variables:"
    echo "  BACKEND_PORT    Backend port (default: 8000)"
    echo "  FRONTEND_PORT   Frontend port (default: 5173)"
    echo "  STREAMLIT_PORT  Streamlit port (default: 8501)"
}

main() {
    local target="${1:-}"

    case "$target" in
        backend)    start_backend ;;
        frontend)   start_frontend ;;
        streamlit)  start_streamlit ;;
        all)        start_all ;;
        *)          show_help ;;
    esac
}

main "$@"
