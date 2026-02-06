#!/usr/bin/env bash
# WeatherFlow Comprehensive Build Script
#
# Builds all project artifacts: Python package, frontend, and documentation.
# Can be run with specific targets or builds everything by default.
#
# Usage:
#   ./scripts/build.sh              # Build everything
#   ./scripts/build.sh python       # Build Python package only
#   ./scripts/build.sh frontend     # Build frontend only
#   ./scripts/build.sh docs         # Build documentation only
#   ./scripts/build.sh --clean      # Clean before building

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

log_step() { echo -e "${BLUE}[BUILD]${RESET} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${RESET}    $1"; }
log_err()  { echo -e "${RED}[FAIL]${RESET}  $1"; }

# ── Python package ──────────────────────────────

build_python() {
    log_step "Building Python package..."
    cd "$ROOT_DIR"

    # Clean previous build
    rm -rf dist/ build/ *.egg-info weatherflow.egg-info/

    # Verify version consistency
    python scripts/check_version.py || {
        log_err "Version mismatch detected. Fix before building."
        return 1
    }

    # Build sdist and wheel
    python -m build

    log_ok "Python package built:"
    ls -lh dist/
}

# ── Frontend ────────────────────────────────────

build_frontend() {
    log_step "Building frontend..."
    cd "$ROOT_DIR/frontend"

    if [ ! -d "node_modules" ]; then
        log_step "Installing frontend dependencies..."
        npm ci
    fi

    npm run build

    log_ok "Frontend built in frontend/dist/"
}

# ── Documentation ───────────────────────────────

build_docs() {
    log_step "Building documentation..."
    cd "$ROOT_DIR"

    mkdocs build

    log_ok "Documentation built in site/"
}

# ── Clean ───────────────────────────────────────

clean_all() {
    log_step "Cleaning build artifacts..."
    cd "$ROOT_DIR"

    rm -rf dist/ build/ *.egg-info weatherflow.egg-info/
    rm -rf site/
    rm -rf frontend/dist/
    rm -rf htmlcov/ .coverage
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name '*.pyc' -delete 2>/dev/null || true

    log_ok "All build artifacts cleaned."
}

# ── Main ────────────────────────────────────────

main() {
    cd "$ROOT_DIR"
    echo -e "${BOLD}WeatherFlow Build System${RESET}"
    echo "========================"
    echo ""

    local targets=("$@")
    local do_clean=false

    # Handle --clean flag
    for i in "${!targets[@]}"; do
        if [[ "${targets[$i]}" == "--clean" ]]; then
            do_clean=true
            unset 'targets[$i]'
        fi
    done

    if $do_clean; then
        clean_all
    fi

    # Default: build everything
    if [ ${#targets[@]} -eq 0 ]; then
        targets=("python" "frontend" "docs")
    fi

    local failed=0
    for target in "${targets[@]}"; do
        case "$target" in
            python)   build_python   || failed=1 ;;
            frontend) build_frontend || failed=1 ;;
            docs)     build_docs     || failed=1 ;;
            clean)    clean_all ;;
            *)
                log_err "Unknown target: $target"
                echo "Valid targets: python, frontend, docs, clean"
                failed=1
                ;;
        esac
        echo ""
    done

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}${BOLD}Build completed successfully.${RESET}"
    else
        echo -e "${RED}${BOLD}Build failed.${RESET}"
        exit 1
    fi
}

main "$@"
