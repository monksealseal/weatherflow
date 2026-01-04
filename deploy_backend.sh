#!/bin/bash

# WeatherFlow Backend Deployment Script
# This script helps set up the centralized backend on Railway

set -e  # Exit on error

echo "========================================="
echo "WeatherFlow Backend Deployment Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${YELLOW}Railway CLI not found.${NC}"
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
    echo -e "${GREEN}✓ Railway CLI installed${NC}"
fi

# Check Railway CLI version
echo "Railway CLI version:"
railway --version
echo ""

# Login to Railway
echo -e "${YELLOW}Logging in to Railway...${NC}"
echo "This will open your browser. Please sign in with GitHub."
railway login

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Railway login failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Logged in to Railway${NC}"
echo ""

# Initialize project
echo -e "${YELLOW}Creating Railway project...${NC}"
echo "This will create a new project linked to this repository."

railway init

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Failed to create Railway project${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Railway project created${NC}"
echo ""

# Set environment variables
echo -e "${YELLOW}Setting environment variables...${NC}"

railway variables set PORT=8000
railway variables set TORCH_NUM_THREADS=4
railway variables set PYTHON_VERSION=3.11

echo -e "${GREEN}✓ Environment variables set${NC}"
echo ""

# Deploy
echo -e "${YELLOW}Deploying to Railway...${NC}"
echo "This will deploy the backend and make it publicly available."

railway up

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Deployment failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Deployment successful${NC}"
echo ""

# Get the public URL
echo -e "${YELLOW}Generating public domain...${NC}"
DOMAIN=$(railway domain)

if [ -z "$DOMAIN" ]; then
    echo -e "${YELLOW}No domain found. Generating one...${NC}"
    railway domain
    DOMAIN=$(railway domain)
fi

echo -e "${GREEN}✓ Backend deployed at: https://$DOMAIN${NC}"
echo ""

# Test health endpoint
echo -e "${YELLOW}Testing health endpoint...${NC}"
sleep 5  # Wait for deployment to be ready

HEALTH_URL="https://$DOMAIN/api/health"
HEALTH_RESPONSE=$(curl -s "$HEALTH_URL" || echo "failed")

if [[ "$HEALTH_RESPONSE" == *"ok"* ]]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "Response: $HEALTH_RESPONSE"
    echo "Please check Railway logs: railway logs"
fi

echo ""
echo "========================================="
echo "Deployment Summary"
echo "========================================="
echo ""
echo "Backend URL: https://$DOMAIN"
echo "Health Check: https://$DOMAIN/api/health"
echo "API Options: https://$DOMAIN/api/options"
echo ""
echo "Next steps:"
echo "1. Update frontend/.env.production with:"
echo "   VITE_API_URL=https://$DOMAIN"
echo ""
echo "2. Update .github/workflows/deploy-pages.yml with:"
echo "   VITE_API_URL: https://$DOMAIN"
echo ""
echo "3. Commit and push changes:"
echo "   git add frontend/.env.production .github/workflows/deploy-pages.yml"
echo "   git commit -m 'Configure backend URL'"
echo "   git push origin main"
echo ""
echo "4. GitHub Pages will automatically redeploy with the new backend URL"
echo ""
echo "View logs: railway logs --tail 100"
echo "Open dashboard: railway open"
echo ""
echo -e "${GREEN}✓ Deployment complete!${NC}"
