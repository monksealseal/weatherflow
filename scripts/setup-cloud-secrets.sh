#!/usr/bin/env bash
#
# One-time setup: Configure GitHub Secrets for cloud deployment
#
# This script helps you set up the three secrets needed for automatic
# cloud deployment via GitHub Actions:
#
#   1. RAILWAY_TOKEN      - Your Railway API token
#   2. RAILWAY_PROJECT_ID - Your Railway project ID
#   3. RAILWAY_SERVICE_ID - Your Railway service ID
#
# After running this once, `make deploy` will work from anywhere.
#
# Prerequisites: gh CLI (GitHub CLI) must be installed and authenticated.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  WeatherFlow Cloud Deployment Setup${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""
echo "This sets up the GitHub Secrets needed for one-button deployment."
echo "You only need to run this once."
echo ""

# Check for gh CLI
if ! command -v gh &> /dev/null; then
  echo -e "${RED}GitHub CLI (gh) is required but not installed.${NC}"
  echo ""
  echo "Install it:"
  echo "  macOS:   brew install gh"
  echo "  Linux:   https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
  echo "  Windows: winget install --id GitHub.cli"
  echo ""
  echo "Then authenticate: gh auth login"
  echo ""
  echo -e "${YELLOW}Alternative: Set secrets manually at:${NC}"
  echo "  https://github.com/monksealseal/weatherflow/settings/secrets/actions"
  echo ""
  echo "Secrets to add:"
  echo "  RAILWAY_TOKEN      - From https://railway.app/account/tokens"
  echo "  RAILWAY_PROJECT_ID - From your Railway project URL"
  echo "  RAILWAY_SERVICE_ID - From your Railway service URL"
  exit 1
fi

# Check gh auth
if ! gh auth status &> /dev/null; then
  echo -e "${YELLOW}Not logged in to GitHub CLI. Running: gh auth login${NC}"
  gh auth login
fi

REPO="monksealseal/weatherflow"
echo -e "${BLUE}Repository: ${REPO}${NC}"
echo ""

# ──────────────────────────────────────────────
# Secret 1: Railway Token
# ──────────────────────────────────────────────
echo -e "${BOLD}Step 1/3: Railway API Token${NC}"
echo ""
echo "  Get your token from: https://railway.app/account/tokens"
echo "  Click 'Create Token', give it a name like 'weatherflow-deploy'"
echo ""
read -sp "  Paste your RAILWAY_TOKEN (input hidden): " RAILWAY_TOKEN
echo ""

if [ -z "$RAILWAY_TOKEN" ]; then
  echo -e "${YELLOW}  Skipped (no value entered)${NC}"
else
  gh secret set RAILWAY_TOKEN --repo "$REPO" --body "$RAILWAY_TOKEN"
  echo -e "${GREEN}  ✓ RAILWAY_TOKEN saved${NC}"
fi
echo ""

# ──────────────────────────────────────────────
# Secret 2: Railway Project ID
# ──────────────────────────────────────────────
echo -e "${BOLD}Step 2/3: Railway Project ID${NC}"
echo ""
echo "  Find this in your Railway dashboard URL:"
echo "  https://railway.app/project/YOUR_PROJECT_ID/..."
echo ""
read -p "  Enter RAILWAY_PROJECT_ID: " RAILWAY_PROJECT_ID
echo ""

if [ -z "$RAILWAY_PROJECT_ID" ]; then
  echo -e "${YELLOW}  Skipped (no value entered)${NC}"
else
  gh secret set RAILWAY_PROJECT_ID --repo "$REPO" --body "$RAILWAY_PROJECT_ID"
  echo -e "${GREEN}  ✓ RAILWAY_PROJECT_ID saved${NC}"
fi
echo ""

# ──────────────────────────────────────────────
# Secret 3: Railway Service ID
# ──────────────────────────────────────────────
echo -e "${BOLD}Step 3/3: Railway Service ID${NC}"
echo ""
echo "  Find this in your Railway service URL:"
echo "  https://railway.app/project/.../service/YOUR_SERVICE_ID"
echo ""
read -p "  Enter RAILWAY_SERVICE_ID: " RAILWAY_SERVICE_ID
echo ""

if [ -z "$RAILWAY_SERVICE_ID" ]; then
  echo -e "${YELLOW}  Skipped (no value entered)${NC}"
else
  gh secret set RAILWAY_SERVICE_ID --repo "$REPO" --body "$RAILWAY_SERVICE_ID"
  echo -e "${GREEN}  ✓ RAILWAY_SERVICE_ID saved${NC}"
fi

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
echo ""
echo "  You can now deploy with a single command:"
echo ""
echo -e "    ${BOLD}make deploy${NC}          # Build + push + deploy everything"
echo -e "    ${BOLD}make deploy-quick${NC}    # Push + deploy (skip build)"
echo -e "    ${BOLD}make scale-up${NC}        # Scale to 4GB RAM / 4 vCPU"
echo -e "    ${BOLD}make deploy-status${NC}   # Check live deployment health"
echo ""
echo "  Or trigger deployment with scaling from GitHub:"
echo ""
echo -e "    ${BOLD}gh workflow run deploy-cloud.yml \\${NC}"
echo -e "      ${BOLD}-f deploy_target=all \\${NC}"
echo -e "      ${BOLD}-f railway_memory_mb=4096 \\${NC}"
echo -e "      ${BOLD}-f railway_cpu_count=4${NC}"
echo ""
echo -e "  Verify secrets: ${BLUE}gh secret list --repo $REPO${NC}"
echo ""
