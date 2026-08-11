# Developer entry points. `make ci` is the gate every stage must pass.
#
# Windows users without GNU make: scripts/dev.ps1 mirrors every target here.
#   .\scripts\dev.ps1 ci
#
# All Python commands run through `uv run`, which resolves the project
# environment without requiring an activated virtualenv.

.DEFAULT_GOAL := help
SHELL := /bin/sh

UV ?= uv
RUN := $(UV) run
PKG := src/multicam_tracker
COMPOSE ?= docker compose

.PHONY: help install install-vision lint format format-check typecheck \
        test-unit test-integration test-all coverage db-up db-down db-logs \
        db-reset clean ci

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

install: ## Create the venv and install core + dev dependencies
	$(UV) sync --extra dev

install-vision: ## Additionally install the CV stack (slow, platform-sensitive)
	$(UV) sync --extra dev --extra vision

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

lint: ## Run ruff lint checks
	$(RUN) ruff check .

format: ## Rewrite files with the ruff formatter
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

format-check: ## Verify formatting without rewriting (used by CI)
	$(RUN) ruff format --check .

typecheck: ## Run mypy in strict mode over $(PKG)
	$(RUN) mypy

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test-unit: ## Run unit tests only (fast, no docker)
	$(RUN) pytest tests/unit -m "not integration"

test-integration: ## Run integration tests (requires docker)
	$(RUN) pytest tests/integration -m integration

test-all: ## Run the whole suite with the coverage gate
	$(RUN) pytest

coverage: ## Run the suite and write an HTML coverage report
	$(RUN) pytest --cov-report=html
	@echo "Report written to htmlcov/index.html"

# ---------------------------------------------------------------------------
# Local database
# ---------------------------------------------------------------------------

db-up: ## Start Postgres+pgvector and wait for the healthcheck to pass
	$(COMPOSE) up -d --wait postgres

db-down: ## Stop Postgres, preserving the data volume
	$(COMPOSE) down

db-reset: ## Stop Postgres and DESTROY the data volume
	$(COMPOSE) down -v

db-logs: ## Tail the Postgres logs
	$(COMPOSE) logs -f postgres

# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

ci: lint format-check typecheck test-all ## Everything CI runs, in CI's order

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
