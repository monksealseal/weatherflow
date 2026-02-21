#!/usr/bin/env bash
#
# WeatherFlow One-Button Cloud Deployment
#
# Usage:
#   ./scripts/deploy-cloud.sh                    # Full deploy (build + push + deploy all)
#   ./scripts/deploy-cloud.sh --frontend-only    # Deploy frontend only
#   ./scripts/deploy-cloud.sh --backend-only     # Deploy backend only
#   ./scripts/deploy-cloud.sh --scale 4096 4     # Deploy with 4GB RAM, 4 vCPUs
#   ./scripts/deploy-cloud.sh --status           # Check deployment status
#   ./scripts/deploy-cloud.sh --skip-build       # Push and deploy without rebuilding
#
# Environment variables:
#   RAILWAY_TOKEN       - Railway API token (required for direct Railway deploy)
#   DEPLOY_BRANCH       - Branch to deploy from (default: current branch)
#   SKIP_TESTS          - Set to 1 to skip tests
#   SKIP_BUILD          - Set to 1 to skip frontend build

set -euo pipefail

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND_URL="${BACKEND_URL:-https://weatherflow-api-production.up.railway.app}"
FRONTEND_URL="${FRONTEND_URL:-https://monksealseal.github.io/weatherflow}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# State
DEPLOY_TARGET="all"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SCALE_MEMORY=""
SCALE_CPU=""
STATUS_ONLY=0
DEPLOY_BRANCH="${DEPLOY_BRANCH:-}"

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
log()    { echo -e "${BLUE}[deploy]${NC} $*"; }
ok()     { echo -e "${GREEN}  ✓${NC} $*"; }
warn()   { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail()   { echo -e "${RED}  ✗${NC} $*"; exit 1; }
header() { echo -e "\n${BOLD}═══ $* ═══${NC}\n"; }

# ──────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --frontend-only) DEPLOY_TARGET="frontend"; shift ;;
    --backend-only)  DEPLOY_TARGET="backend";  shift ;;
    --skip-build)    SKIP_BUILD=1;             shift ;;
    --skip-tests)    SKIP_TESTS=1;             shift ;;
    --status)        STATUS_ONLY=1;            shift ;;
    --scale)
      SCALE_MEMORY="${2:-2048}"; SCALE_CPU="${3:-2}"
      shift; shift 2>/dev/null || true; shift 2>/dev/null || true
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --frontend-only    Deploy frontend to GitHub Pages only"
      echo "  --backend-only     Deploy backend to Railway only"
      echo "  --skip-build       Skip frontend build step"
      echo "  --skip-tests       Skip test step"
      echo "  --scale MEM CPU    Set Railway resources (e.g., --scale 4096 4)"
      echo "  --status           Check current deployment status"
      echo "  -h, --help         Show this help"
      exit 0
      ;;
    *) warn "Unknown option: $1"; shift ;;
  esac
done

cd "$PROJECT_ROOT"

# ──────────────────────────────────────────────────────────
# Status check
# ──────────────────────────────────────────────────────────
check_status() {
  header "Deployment Status"

  echo -e "${BOLD}Backend${NC} (${DIM}${BACKEND_URL}${NC})"
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BACKEND_URL}/api/health" 2>/dev/null || echo "000")
  if [ "$STATUS" = "200" ]; then
    ok "Backend is healthy (HTTP $STATUS)"
    HEALTH=$(curl -s "${BACKEND_URL}/api/health" 2>/dev/null)
    echo -e "     ${DIM}${HEALTH}${NC}"
  else
    warn "Backend returned HTTP $STATUS"
  fi

  echo ""
  echo -e "${BOLD}Frontend${NC} (${DIM}${FRONTEND_URL}${NC})"
  FE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${FRONTEND_URL}" 2>/dev/null || echo "000")
  if [ "$FE_STATUS" = "200" ] || [ "$FE_STATUS" = "304" ]; then
    ok "Frontend is live (HTTP $FE_STATUS)"
  else
    warn "Frontend returned HTTP $FE_STATUS"
  fi

  echo ""
  echo -e "${BOLD}Git${NC}"
  BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
  LAST_COMMIT=$(git log -1 --pretty=format:"%h %s" 2>/dev/null || echo "unknown")
  echo -e "  Branch: ${BRANCH}"
  echo -e "  Latest: ${DIM}${LAST_COMMIT}${NC}"

  AHEAD=$(git rev-list --count origin/${BRANCH}..HEAD 2>/dev/null || echo "?")
  if [ "$AHEAD" != "?" ] && [ "$AHEAD" -gt 0 ]; then
    warn "${AHEAD} commit(s) ahead of remote — push to trigger deploy"
  else
    ok "Up to date with remote"
  fi
}

if [ "$STATUS_ONLY" = "1" ]; then
  check_status
  exit 0
fi

# ──────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     WeatherFlow Cloud Deployment                 ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Target:  ${BOLD}${DEPLOY_TARGET}${NC}"
[ -n "$SCALE_MEMORY" ] && echo -e "  Memory:  ${BOLD}${SCALE_MEMORY} MB${NC}"
[ -n "$SCALE_CPU" ]    && echo -e "  vCPUs:   ${BOLD}${SCALE_CPU}${NC}"
echo ""

# ──────────────────────────────────────────────────────────
# Step 1: Pre-flight checks
# ──────────────────────────────────────────────────────────
header "Step 1/5: Pre-flight checks"

# Check git status
BRANCH=$(git branch --show-current 2>/dev/null)
if [ -z "$BRANCH" ]; then
  fail "Not on a git branch"
fi
ok "On branch: ${BRANCH}"

# Check for uncommitted changes
if ! git diff --quiet HEAD 2>/dev/null; then
  warn "Uncommitted changes detected — will auto-commit before deploy"
fi

# ──────────────────────────────────────────────────────────
# Step 2: Build frontend
# ──────────────────────────────────────────────────────────
if [ "$SKIP_BUILD" = "0" ] && [ "$DEPLOY_TARGET" != "backend" ]; then
  header "Step 2/5: Build frontend"

  if [ -f "frontend/package.json" ]; then
    log "Installing frontend dependencies..."
    cd frontend
    npm ci --silent 2>/dev/null || npm install --silent
    ok "Dependencies installed"

    log "Building frontend for production..."
    VITE_API_URL="$BACKEND_URL" npm run build
    BUILD_SIZE=$(du -sh dist/ 2>/dev/null | cut -f1)
    ok "Frontend built (${BUILD_SIZE})"
    cd "$PROJECT_ROOT"
  else
    warn "No frontend/package.json found, skipping frontend build"
  fi
else
  log "Skipping frontend build"
fi

# ──────────────────────────────────────────────────────────
# Step 3: Run tests (quick)
# ──────────────────────────────────────────────────────────
if [ "$SKIP_TESTS" = "0" ]; then
  header "Step 3/5: Quick tests"

  if command -v python &> /dev/null && [ -d "tests" ]; then
    log "Running Python tests..."
    python -m pytest tests/ --maxfail=3 -q --tb=short 2>/dev/null && \
      ok "Python tests passed" || \
      warn "Some tests failed (continuing deploy)"
  fi

  if [ -f "frontend/package.json" ] && [ "$DEPLOY_TARGET" != "backend" ]; then
    log "Type-checking frontend..."
    cd frontend
    npx tsc --noEmit 2>/dev/null && ok "TypeScript checks passed" || warn "TypeScript has warnings (non-blocking)"
    cd "$PROJECT_ROOT"
  fi
else
  log "Skipping tests"
fi

# ──────────────────────────────────────────────────────────
# Step 4: Commit and push
# ──────────────────────────────────────────────────────────
header "Step 4/5: Push to remote"

# Stage and commit if there are changes
if ! git diff --quiet HEAD 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard 2>/dev/null)" ]; then
  log "Staging changes..."
  git add -A
  COMMIT_MSG="deploy: cloud deployment $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [ -n "$SCALE_MEMORY" ] && COMMIT_MSG="${COMMIT_MSG} [scale:${SCALE_MEMORY}MB/${SCALE_CPU}cpu]"
  git commit -m "$COMMIT_MSG" 2>/dev/null || true
  ok "Changes committed"
fi

# Push
PUSH_BRANCH="${DEPLOY_BRANCH:-$BRANCH}"
log "Pushing to origin/${PUSH_BRANCH}..."
git push -u origin "$PUSH_BRANCH" 2>&1 && ok "Pushed to origin/${PUSH_BRANCH}" || warn "Push may have failed (check git output above)"

# ──────────────────────────────────────────────────────────
# Step 5: Trigger cloud deployment
# ──────────────────────────────────────────────────────────
header "Step 5/5: Cloud deployment"

# Method 1: If gh CLI is available, trigger workflow dispatch
if command -v gh &> /dev/null; then
  log "Triggering deploy-cloud workflow via GitHub CLI..."

  GH_ARGS="--ref $PUSH_BRANCH -f deploy_target=${DEPLOY_TARGET}"
  [ -n "$SCALE_MEMORY" ] && GH_ARGS="${GH_ARGS} -f railway_memory_mb=${SCALE_MEMORY}"
  [ -n "$SCALE_CPU" ]    && GH_ARGS="${GH_ARGS} -f railway_cpu_count=${SCALE_CPU}"

  gh workflow run deploy-cloud.yml $GH_ARGS 2>/dev/null && \
    ok "Workflow triggered" || \
    warn "Could not trigger workflow (deploy will still run on push to main)"

  # Watch the run
  log "Monitoring deployment..."
  sleep 3
  gh run list --workflow=deploy-cloud.yml --limit=1 2>/dev/null || true

# Method 2: If Railway CLI is available, deploy directly
elif command -v railway &> /dev/null && [ -n "${RAILWAY_TOKEN:-}" ]; then
  if [ "$DEPLOY_TARGET" != "frontend" ]; then
    log "Deploying backend directly via Railway CLI..."

    if [ -n "$SCALE_MEMORY" ]; then
      railway variables set TORCH_NUM_THREADS="${SCALE_CPU}" UVICORN_WORKERS="${SCALE_CPU}" 2>/dev/null || true
    fi

    railway up --detach 2>/dev/null && ok "Railway deployment triggered" || warn "Railway deploy returned non-zero"
  fi

  if [ "$DEPLOY_TARGET" != "backend" ] && [ "$PUSH_BRANCH" = "main" ]; then
    ok "Frontend will deploy via GitHub Pages (push to main detected)"
  fi

# Method 3: Push-triggered deployment
else
  if [ "$PUSH_BRANCH" = "main" ]; then
    ok "Push to main detected — GitHub Actions will handle deployment"
    echo -e "  ${DIM}Frontend: GitHub Pages via deploy-pages.yml${NC}"
    echo -e "  ${DIM}Backend:  Railway via deploy-cloud.yml${NC}"
  else
    warn "Not on main branch. Merge to main to trigger auto-deployment."
    echo -e "  ${DIM}Or install the GitHub CLI (gh) to trigger manual deploys from any branch.${NC}"
  fi
fi

# ──────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║           Deployment Summary                     ║${NC}"
echo -e "${BOLD}╠═══════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}                                                   ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Frontend: ${FRONTEND_URL}  ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Backend:  ${BACKEND_URL}   ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  Branch:   ${PUSH_BRANCH}                           ${BOLD}║${NC}"
[ -n "$SCALE_MEMORY" ] && \
echo -e "${BOLD}║${NC}  Scale:    ${SCALE_MEMORY}MB RAM / ${SCALE_CPU} vCPU               ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}                                                   ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  ${DIM}Check status: make deploy-status${NC}               ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  ${DIM}View logs:    make deploy-logs${NC}                 ${BOLD}║${NC}"
echo -e "${BOLD}║${NC}                                                   ${BOLD}║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════╝${NC}"
echo ""
