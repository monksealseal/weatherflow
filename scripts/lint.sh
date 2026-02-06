#!/usr/bin/env bash
# WeatherFlow Unified Linting Script
#
# Runs all code quality checks: formatting, imports, linting, type checking.
#
# Usage:
#   ./scripts/lint.sh              # Run all checks
#   ./scripts/lint.sh --fix        # Fix auto-fixable issues
#   ./scripts/lint.sh format       # Check formatting only
#   ./scripts/lint.sh types        # Run type checking only
#   ./scripts/lint.sh frontend     # Run frontend linting only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

PACKAGE="weatherflow"
FIX_MODE=false
FAILED=0
TOTAL=0
PASSED=0

log_step() { echo -e "${BLUE}[$((TOTAL + 1))]${RESET} $1"; }

run_check() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    log_step "$name"

    if "$@" 2>&1; then
        PASSED=$((PASSED + 1))
        echo -e "    ${GREEN}PASSED${RESET}"
    else
        FAILED=$((FAILED + 1))
        echo -e "    ${RED}FAILED${RESET}"
    fi
    echo ""
}

# ── Checks ──────────────────────────────────────

check_format() {
    if $FIX_MODE; then
        run_check "Black (formatting)" python -m black "$PACKAGE/" tests/
        run_check "isort (imports)"    python -m isort "$PACKAGE/" tests/
    else
        run_check "Black (formatting)" python -m black --check --diff "$PACKAGE/" tests/
        run_check "isort (imports)"    python -m isort --check-only --diff "$PACKAGE/" tests/
    fi
}

check_lint() {
    run_check "Flake8 (linting)" python -m flake8 "$PACKAGE/" tests/
}

check_types() {
    run_check "mypy (type checking)" python -m mypy "$PACKAGE/"
}

check_frontend() {
    if [ -d "$ROOT_DIR/frontend/node_modules" ]; then
        run_check "ESLint (frontend)" bash -c "cd '$ROOT_DIR/frontend' && npm run lint"
    else
        echo -e "${YELLOW}Skipping frontend lint: node_modules not installed.${RESET}"
        echo "  Run 'make install-frontend' first."
        echo ""
    fi
}

# ── Main ────────────────────────────────────────

main() {
    cd "$ROOT_DIR"
    echo -e "${BOLD}WeatherFlow Code Quality Checks${RESET}"
    echo "================================"
    echo ""

    local targets=("$@")

    # Handle --fix flag
    for i in "${!targets[@]}"; do
        if [[ "${targets[$i]}" == "--fix" ]]; then
            FIX_MODE=true
            unset 'targets[$i]'
        fi
    done
    # Re-index
    targets=("${targets[@]}")

    # Default: run all checks
    if [ ${#targets[@]} -eq 0 ]; then
        targets=("format" "lint" "types" "frontend")
    fi

    for target in "${targets[@]}"; do
        case "$target" in
            format)   check_format ;;
            lint)     check_lint ;;
            types)    check_types ;;
            frontend) check_frontend ;;
            *)
                echo -e "${RED}Unknown target: $target${RESET}"
                echo "Valid targets: format, lint, types, frontend"
                FAILED=$((FAILED + 1))
                ;;
        esac
    done

    # Summary
    echo "================================"
    echo -e "${BOLD}Results: ${PASSED}/${TOTAL} passed${RESET}"

    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}${FAILED} check(s) failed.${RESET}"
        if ! $FIX_MODE; then
            echo -e "${YELLOW}Tip: Run './scripts/lint.sh --fix' to auto-fix formatting issues.${RESET}"
        fi
        exit 1
    else
        echo -e "${GREEN}All checks passed.${RESET}"
    fi
}

main "$@"
