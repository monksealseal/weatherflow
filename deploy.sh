#!/bin/bash
# GCM Heroku Deployment Script
# Run this script to deploy the GCM web application to Heroku

set -e  # Exit on error

echo "=================================================="
echo "  GCM Heroku Deployment Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo -e "${RED}Error: Heroku CLI is not installed${NC}"
    echo ""
    echo "Please install Heroku CLI first:"
    echo "  macOS: brew tap heroku/brew && brew install heroku"
    echo "  Ubuntu: curl https://cli-assets.heroku.com/install.sh | sh"
    echo "  Windows: Download from https://devcenter.heroku.com/articles/heroku-cli"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Heroku CLI found${NC}"

# Check if logged in to Heroku
if ! heroku auth:whoami &> /dev/null; then
    echo -e "${YELLOW}⚠ Not logged in to Heroku${NC}"
    echo "Opening login page..."
    heroku login
else
    echo -e "${GREEN}✓ Logged in to Heroku as $(heroku auth:whoami)${NC}"
fi

# Ask for app name
echo ""
read -p "Enter Heroku app name (leave blank for auto-generated): " APP_NAME

# Create Heroku app
echo ""
echo "Creating Heroku app..."

if [ -z "$APP_NAME" ]; then
    HEROKU_OUTPUT=$(heroku create 2>&1)
else
    HEROKU_OUTPUT=$(heroku create "$APP_NAME" 2>&1)
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Heroku app created${NC}"
    echo "$HEROKU_OUTPUT"

    # Extract app name from output
    APP_NAME=$(echo "$HEROKU_OUTPUT" | grep -oP 'https://\K[^.]+' | head -1)
    APP_URL="https://${APP_NAME}.herokuapp.com"
else
    echo -e "${RED}Error creating app:${NC}"
    echo "$HEROKU_OUTPUT"
    exit 1
fi

# Verify we're on the correct branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "claude/build-gcm-physics-VaCFZ" ]; then
    echo -e "${YELLOW}⚠ Warning: Not on deployment branch${NC}"
    echo "Current branch: $CURRENT_BRANCH"
    echo "Switching to claude/build-gcm-physics-VaCFZ..."
    git checkout claude/build-gcm-physics-VaCFZ
fi

# Deploy to Heroku
echo ""
echo "Deploying to Heroku..."
echo "This may take a few minutes..."
echo ""

if git push heroku claude/build-gcm-physics-VaCFZ:main; then
    echo ""
    echo -e "${GREEN}✓ Deployment successful!${NC}"
else
    echo ""
    echo -e "${RED}✗ Deployment failed${NC}"
    echo "Check the error messages above"
    exit 1
fi

# Wait for app to be ready
echo ""
echo "Waiting for app to start..."
sleep 5

# Open the app
echo ""
echo "=================================================="
echo -e "${GREEN}  Deployment Complete!${NC}"
echo "=================================================="
echo ""
echo "App name: $APP_NAME"
echo "URL: $APP_URL"
echo ""
echo "Opening app in browser..."
heroku open -a "$APP_NAME"

echo ""
echo "Useful commands:"
echo "  View logs:    heroku logs --tail -a $APP_NAME"
echo "  Open app:     heroku open -a $APP_NAME"
echo "  Restart:      heroku restart -a $APP_NAME"
echo "  Scale up:     heroku ps:scale web=1 -a $APP_NAME"
echo ""
echo -e "${GREEN}Happy simulating! 🌍${NC}"
