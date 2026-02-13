# WeatherFlow Build System
# Unified entry point for all build, test, lint, and deployment operations
#
# Usage:
#   make help          - Show all available targets
#   make install       - Install package in development mode
#   make test          - Run all tests
#   make lint          - Run all linters
#   make build         - Build all artifacts
#   make dev           - Start development servers
#
# Prerequisites:
#   - Python >= 3.8
#   - Node.js >= 18 (for frontend)
#   - Docker (optional, for container builds)

.PHONY: help install install-dev install-docs install-frontend install-all \
        test test-python test-frontend test-coverage \
        lint lint-python lint-frontend lint-types format format-check \
        build build-python build-frontend build-docs build-docker \
        dev dev-backend dev-frontend dev-streamlit \
        docker-build docker-up docker-down docker-shell \
        deploy deploy-docker deploy-fly deploy-railway deploy-stop deploy-status deploy-logs \
        clean clean-python clean-frontend clean-docs clean-docker \
        release-check version-check pre-commit-install pre-commit-run \
        ci-test ci-build

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

PYTHON       ?= python
PIP          ?= pip
NPM          ?= npm
DOCKER       ?= docker
COMPOSE      ?= docker compose

PACKAGE      := weatherflow
FRONTEND_DIR := frontend
DOCS_DIR     := docs
BUILD_DIR    := dist
COVERAGE_DIR := htmlcov

# Colors for terminal output
BLUE  := \033[0;34m
GREEN := \033[0;32m
RED   := \033[0;31m
BOLD  := \033[1m
RESET := \033[0m

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────

help: ## Show this help message
	@echo "$(BOLD)WeatherFlow Build System$(RESET)"
	@echo "========================"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET) make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ──────────────────────────────────────────────
# Installation
# ──────────────────────────────────────────────

install: ## Install package in development mode
	$(PIP) install -e .

install-dev: ## Install with development dependencies
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt

install-docs: ## Install with documentation dependencies
	$(PIP) install -e ".[docs]"
	$(PIP) install -r requirements-docs.txt

install-frontend: ## Install frontend dependencies
	cd $(FRONTEND_DIR) && $(NPM) ci

install-all: install-dev install-docs install-frontend ## Install everything (Python + frontend + docs)
	@echo "$(GREEN)All dependencies installed.$(RESET)"

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────

test: test-python ## Run all tests

test-python: ## Run Python tests with pytest
	$(PYTHON) -m pytest tests/ --maxfail=3 -q

test-frontend: ## Run frontend tests with Vitest
	cd $(FRONTEND_DIR) && $(NPM) run test

test-coverage: ## Run Python tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html:$(COVERAGE_DIR)
	@echo "$(GREEN)Coverage report: $(COVERAGE_DIR)/index.html$(RESET)"

test-all: test-python test-frontend ## Run both Python and frontend tests

# ──────────────────────────────────────────────
# Linting & Formatting
# ──────────────────────────────────────────────

lint: lint-python ## Run all linters

lint-python: ## Run Python linters (flake8)
	$(PYTHON) -m flake8 $(PACKAGE)/ tests/

lint-types: ## Run mypy type checking
	$(PYTHON) -m mypy $(PACKAGE)/

lint-frontend: ## Run frontend ESLint
	cd $(FRONTEND_DIR) && $(NPM) run lint

lint-all: lint-python lint-types lint-frontend ## Run all linters including types and frontend

format: ## Format Python code with black + isort
	$(PYTHON) -m black $(PACKAGE)/ tests/
	$(PYTHON) -m isort $(PACKAGE)/ tests/

format-check: ## Check Python formatting without making changes
	$(PYTHON) -m black --check $(PACKAGE)/ tests/
	$(PYTHON) -m isort --check-only $(PACKAGE)/ tests/

# ──────────────────────────────────────────────
# Building
# ──────────────────────────────────────────────

build: build-python ## Build all artifacts

build-python: clean-python ## Build Python package (sdist + wheel)
	$(PYTHON) -m build
	@echo "$(GREEN)Python package built in $(BUILD_DIR)/$(RESET)"
	@ls -lh $(BUILD_DIR)/

build-frontend: ## Build frontend for production
	cd $(FRONTEND_DIR) && $(NPM) run build
	@echo "$(GREEN)Frontend built in $(FRONTEND_DIR)/dist/$(RESET)"

build-docs: ## Build MkDocs documentation
	mkdocs build
	@echo "$(GREEN)Documentation built in site/$(RESET)"

build-all: build-python build-frontend build-docs ## Build everything
	@echo "$(GREEN)All artifacts built successfully.$(RESET)"

# ──────────────────────────────────────────────
# Development Servers
# ──────────────────────────────────────────────

dev: ## Start backend + frontend dev servers (requires two terminals or use dev-backend / dev-frontend separately)
	@echo "$(BOLD)Start development servers in separate terminals:$(RESET)"
	@echo "  Terminal 1: make dev-backend"
	@echo "  Terminal 2: make dev-frontend"
	@echo ""
	@echo "Or run the backend in the background:"
	@echo "  make dev-backend &"
	@echo "  make dev-frontend"

dev-backend: ## Start FastAPI backend server (port 8000)
	$(PYTHON) -m uvicorn weatherflow.server.app:app --host 0.0.0.0 --port 8000 --reload

dev-frontend: ## Start Vite frontend dev server (port 5173)
	cd $(FRONTEND_DIR) && $(NPM) run dev

dev-streamlit: ## Start Streamlit app (port 8501)
	streamlit run streamlit_app/Home.py

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────

docker-build: ## Build Docker image
	$(DOCKER) build -t weatherflow .

docker-up: ## Start docker-compose services (demo profile)
	$(COMPOSE) --profile demo up --build

docker-up-web: ## Start docker-compose web service
	$(COMPOSE) --profile web up --build

docker-down: ## Stop docker-compose services
	$(COMPOSE) down

docker-shell: ## Open interactive shell in Docker container
	$(COMPOSE) --profile shell run --rm gcm-shell

# ──────────────────────────────────────────────
# Production Deployment
# ──────────────────────────────────────────────

deploy: deploy-docker ## Deploy production stack (alias for deploy-docker)

deploy-docker: ## Deploy full stack with Docker Compose (Streamlit + API + nginx)
	$(COMPOSE) -f docker-compose.prod.yml up --build -d
	@echo "$(GREEN)WeatherFlow deployed at http://localhost:$${PORT:-80}$(RESET)"

deploy-fly: ## Deploy Streamlit app to Fly.io
	flyctl deploy

deploy-railway: ## Deploy API to Railway
	railway up

deploy-stop: ## Stop production Docker Compose services
	$(COMPOSE) -f docker-compose.prod.yml down

deploy-status: ## Show status of production services
	$(COMPOSE) -f docker-compose.prod.yml ps

deploy-logs: ## Show logs from production services
	$(COMPOSE) -f docker-compose.prod.yml logs -f --tail=100

# ──────────────────────────────────────────────
# Code Quality
# ──────────────────────────────────────────────

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# ──────────────────────────────────────────────
# Release
# ──────────────────────────────────────────────

release-check: version-check ## Run all pre-release checks
	@echo "$(BOLD)Running pre-release checks...$(RESET)"
	@echo ""
	@echo "$(BLUE)[1/6]$(RESET) Running tests..."
	$(PYTHON) -m pytest tests/ -q --tb=short
	@echo ""
	@echo "$(BLUE)[2/6]$(RESET) Checking code format..."
	$(PYTHON) -m black --check $(PACKAGE)/
	@echo ""
	@echo "$(BLUE)[3/6]$(RESET) Checking import order..."
	$(PYTHON) -m isort --check-only $(PACKAGE)/
	@echo ""
	@echo "$(BLUE)[4/6]$(RESET) Running linter..."
	$(PYTHON) -m flake8 $(PACKAGE)/
	@echo ""
	@echo "$(BLUE)[5/6]$(RESET) Building package..."
	$(PYTHON) -m build
	@echo ""
	@echo "$(BLUE)[6/6]$(RESET) Building frontend..."
	cd $(FRONTEND_DIR) && $(NPM) run build
	@echo ""
	@echo "$(GREEN)$(BOLD)All pre-release checks passed.$(RESET)"

version-check: ## Verify version consistency across all files
	@echo "Checking version consistency..."
	@$(PYTHON) scripts/check_version.py

# ──────────────────────────────────────────────
# CI Targets (used by GitHub Actions)
# ──────────────────────────────────────────────

ci-test: ## CI: Install deps and run tests
	$(PIP) install --upgrade pip
	$(PIP) install pytest pytest-cov
	$(PIP) install -e .
	$(PYTHON) -m pytest tests/ --maxfail=1 --cov=$(PACKAGE) --cov-report=term-missing

ci-build: ## CI: Build Python package
	$(PIP) install --upgrade pip
	$(PIP) install build
	$(PYTHON) -m build

ci-lint: ## CI: Run linting checks
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m black --check $(PACKAGE)/ tests/
	$(PYTHON) -m isort --check-only $(PACKAGE)/ tests/
	$(PYTHON) -m flake8 $(PACKAGE)/ tests/

ci-frontend-build: ## CI: Build frontend
	cd $(FRONTEND_DIR) && $(NPM) ci && $(NPM) run build

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

clean: clean-python clean-docs ## Clean all build artifacts

clean-python: ## Clean Python build artifacts
	rm -rf $(BUILD_DIR)/ build/ *.egg-info $(PACKAGE).egg-info/
	rm -rf $(COVERAGE_DIR)/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	@echo "$(GREEN)Python artifacts cleaned.$(RESET)"

clean-frontend: ## Clean frontend build artifacts
	rm -rf $(FRONTEND_DIR)/dist/ $(FRONTEND_DIR)/node_modules/
	@echo "$(GREEN)Frontend artifacts cleaned.$(RESET)"

clean-docs: ## Clean documentation build
	rm -rf site/
	@echo "$(GREEN)Docs artifacts cleaned.$(RESET)"

clean-docker: ## Remove Docker images and containers
	$(COMPOSE) down --rmi local --volumes --remove-orphans
	@echo "$(GREEN)Docker artifacts cleaned.$(RESET)"

clean-all: clean clean-frontend clean-docker ## Clean absolutely everything
	@echo "$(GREEN)All artifacts cleaned.$(RESET)"
