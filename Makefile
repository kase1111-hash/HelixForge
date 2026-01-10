# HelixForge Makefile
# Cross-Dataset Insight Synthesizer

.PHONY: help install install-dev test lint typecheck format clean docker-build docker-up docker-down run

# Default target
help:
	@echo "HelixForge - Cross-Dataset Insight Synthesizer"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Installation:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install all dependencies including dev"
	@echo ""
	@echo "Development:"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-int      Run integration tests only"
	@echo "  lint          Run linter (ruff)"
	@echo "  typecheck     Run type checker (mypy)"
	@echo "  format        Format code (ruff)"
	@echo "  check         Run all checks (lint + typecheck)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build Docker images"
	@echo "  docker-up     Start all services"
	@echo "  docker-down   Stop all services"
	@echo ""
	@echo "Running:"
	@echo "  run           Run the API server"
	@echo "  run-dev       Run API server in development mode"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         Remove cache and build artifacts"
	@echo "  clean-all     Remove all generated files"

# Python and pip
PYTHON := python3
PIP := pip3
PYTEST := pytest
RUFF := ruff
MYPY := mypy

# Directories
SRC_DIRS := agents api models utils
TEST_DIR := tests

# Installation targets
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pytest pytest-cov pytest-asyncio ruff mypy types-requests types-PyYAML

# Testing targets
test:
	$(PYTEST) $(TEST_DIR) -v --tb=short

test-unit:
	$(PYTEST) $(TEST_DIR) -v --tb=short --ignore=$(TEST_DIR)/test_integration.py

test-int:
	$(PYTEST) $(TEST_DIR)/test_integration.py -v --tb=short

test-cov:
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIRS) --cov-report=html --cov-report=term-missing

# Linting and type checking
lint:
	$(RUFF) check $(SRC_DIRS) $(TEST_DIR)

lint-fix:
	$(RUFF) check $(SRC_DIRS) $(TEST_DIR) --fix

typecheck:
	$(MYPY) $(SRC_DIRS) --config-file=pyproject.toml

format:
	$(RUFF) format $(SRC_DIRS) $(TEST_DIR)

check: lint typecheck

# Docker targets
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart

# Run targets
run:
	uvicorn api.server:app --host 0.0.0.0 --port 8000

run-dev:
	uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Cleanup targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

clean-all: clean
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	rm -rf *.egg-info/ 2>/dev/null || true
	rm -rf outputs/* 2>/dev/null || true
	rm -rf data/processed/* 2>/dev/null || true

# Database migrations (placeholder)
db-migrate:
	@echo "Running database migrations..."
	@echo "Note: Implement migrations with alembic or similar"

# Version management
version:
	@grep "^version" pyproject.toml | head -1

# Build distributable package
build:
	$(PYTHON) -m build

# Security scan
security-scan:
	@echo "Running security scan..."
	pip-audit 2>/dev/null || echo "Install pip-audit: pip install pip-audit"
	bandit -r $(SRC_DIRS) 2>/dev/null || echo "Install bandit: pip install bandit"
