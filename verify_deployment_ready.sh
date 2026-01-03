#!/bin/bash
# Verify deployment readiness

echo "=================================================="
echo "  GCM Deployment Readiness Check"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

# Check required files
echo "Checking required files..."

required_files=(
    "app.py"
    "Procfile"
    "requirements.txt"
    "runtime.txt"
    "templates/index.html"
    "static/css/style.css"
    "static/js/app.js"
    "gcm/__init__.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file (MISSING)"
        ((ERRORS++))
    fi
done

echo ""

# Check Procfile format
echo "Checking Procfile..."
if grep -q "web: gunicorn app:app" Procfile; then
    echo -e "  ${GREEN}✓${NC} Procfile format correct"
else
    echo -e "  ${RED}✗${NC} Procfile format incorrect"
    ((ERRORS++))
fi

echo ""

# Check runtime.txt
echo "Checking runtime.txt..."
if grep -q "python-3.11" runtime.txt; then
    echo -e "  ${GREEN}✓${NC} Python version specified"
else
    echo -e "  ${YELLOW}⚠${NC} Python version might be outdated"
    ((WARNINGS++))
fi

echo ""

# Check requirements.txt
echo "Checking requirements.txt..."
required_packages=("flask" "gunicorn" "numpy" "matplotlib")
for pkg in "${required_packages[@]}"; do
    if grep -qi "$pkg" requirements.txt; then
        echo -e "  ${GREEN}✓${NC} $pkg"
    else
        echo -e "  ${RED}✗${NC} $pkg (MISSING)"
        ((ERRORS++))
    fi
done

echo ""

# Check Git status
echo "Checking Git status..."
if git diff --quiet && git diff --cached --quiet; then
    echo -e "  ${GREEN}✓${NC} All changes committed"
else
    echo -e "  ${YELLOW}⚠${NC} Uncommitted changes"
    ((WARNINGS++))
fi

echo ""

# Check branch
echo "Checking Git branch..."
BRANCH=$(git branch --show-current)
if [ "$BRANCH" = "claude/build-gcm-physics-VaCFZ" ]; then
    echo -e "  ${GREEN}✓${NC} On deployment branch"
else
    echo -e "  ${YELLOW}⚠${NC} Not on deployment branch (current: $BRANCH)"
    ((WARNINGS++))
fi

echo ""

# Check GCM package structure
echo "Checking GCM package structure..."
gcm_modules=("core" "physics" "grid" "numerics" "io" "utils")
for module in "${gcm_modules[@]}"; do
    if [ -d "gcm/$module" ] && [ -f "gcm/$module/__init__.py" ]; then
        echo -e "  ${GREEN}✓${NC} gcm/$module"
    else
        echo -e "  ${RED}✗${NC} gcm/$module (MISSING)"
        ((ERRORS++))
    fi
done

echo ""

# Check .slugignore (optional but recommended)
echo "Checking .slugignore..."
if [ -f ".slugignore" ]; then
    echo -e "  ${GREEN}✓${NC} .slugignore present (optimized slug size)"
else
    echo -e "  ${YELLOW}⚠${NC} .slugignore missing (larger slug size)"
    ((WARNINGS++))
fi

echo ""

# Summary
echo "=================================================="
echo "  Summary"
echo "=================================================="
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "Your app is ready to deploy to Heroku!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./deploy.sh"
    echo "  2. Or manually: git push heroku claude/build-gcm-physics-VaCFZ:main"
    echo ""
else
    echo -e "${RED}✗ Found $ERRORS error(s)${NC}"
    echo ""
    echo "Please fix the errors above before deploying."
    exit 1
fi

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $WARNINGS warning(s)${NC}"
    echo "These won't prevent deployment but should be addressed."
    echo ""
fi

echo "=================================================="
