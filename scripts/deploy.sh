#!/usr/bin/env bash
# WeatherFlow Deployment Script
# One-command deployment for various platforms
#
# Usage:
#   ./scripts/deploy.sh docker       # Deploy with Docker Compose (local/VPS)
#   ./scripts/deploy.sh fly          # Deploy to Fly.io
#   ./scripts/deploy.sh railway      # Deploy to Railway
#   ./scripts/deploy.sh stop         # Stop Docker Compose services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${BLUE}[INFO]${RESET} $*"; }
ok()    { echo -e "${GREEN}[OK]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

usage() {
    echo -e "${BOLD}WeatherFlow Deployment${RESET}"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  docker    Deploy with Docker Compose (local server or VPS)"
    echo "  fly       Deploy to Fly.io"
    echo "  railway   Deploy to Railway"
    echo "  stop      Stop Docker Compose services"
    echo "  status    Show status of running services"
    echo "  logs      Show logs from Docker Compose services"
    echo ""
}

check_command() {
    if ! command -v "$1" &>/dev/null; then
        error "$1 is required but not installed."
        exit 1
    fi
}

deploy_docker() {
    info "Deploying WeatherFlow with Docker Compose..."
    check_command docker

    cd "$PROJECT_DIR"

    info "Building and starting services..."
    docker compose -f docker-compose.prod.yml up --build -d

    echo ""
    ok "WeatherFlow is starting up!"
    echo ""
    echo -e "  ${BOLD}Streamlit App:${RESET}  http://localhost:${PORT:-80}"
    echo -e "  ${BOLD}API Health:${RESET}     http://localhost:${PORT:-80}/api/health"
    echo ""
    echo "Run '$0 logs' to view logs"
    echo "Run '$0 stop' to stop services"
}

deploy_fly() {
    info "Deploying WeatherFlow to Fly.io..."
    check_command flyctl

    cd "$PROJECT_DIR"

    # Check if app exists
    if ! flyctl status &>/dev/null 2>&1; then
        info "Creating Fly.io app..."
        flyctl launch --no-deploy --copy-config
    fi

    info "Deploying..."
    flyctl deploy

    echo ""
    ok "Deployed to Fly.io!"
    flyctl status
}

deploy_railway() {
    info "Deploying WeatherFlow to Railway..."
    check_command railway

    cd "$PROJECT_DIR"

    info "Deploying via Railway CLI..."
    railway up

    echo ""
    ok "Deployed to Railway!"
    echo "Check your Railway dashboard for the deployment URL."
}

stop_services() {
    info "Stopping WeatherFlow services..."
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.prod.yml down
    ok "Services stopped."
}

show_status() {
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.prod.yml ps
}

show_logs() {
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.prod.yml logs -f --tail=100
}

# ── Main ──

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

case "$1" in
    docker)  deploy_docker ;;
    fly)     deploy_fly ;;
    railway) deploy_railway ;;
    stop)    stop_services ;;
    status)  show_status ;;
    logs)    show_logs ;;
    *)
        error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
