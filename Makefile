.PHONY: help install install-prod sync clean test test-unit test-integration test-cov lint format check steer stitch sae localise init pre-commit notebook notebook-strip app

# Default target
help:
	@echo "T2I-Interp Toolkit - Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install          Install all dependencies (including dev tools)"
	@echo "  make install-prod     Install production dependencies only"
	@echo "  make sync             Install all optional dependencies (dev, ray, notebook)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Format code with ruff"
	@echo "  make check           Run all checks (lint + format check)"
	@echo "  make init            Install pre-commit hooks"
	@echo "  make pre-commit      Run pre-commit on all files"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-cov        Run tests with coverage report"
	@echo ""
	@echo "Workflows (each forwards extra Hydra overrides):"
	@echo "  make steer           Run steering workflow         (STEER_ARGS=...)"
	@echo "  make stitch          Run stitching workflow        (STITCH_ARGS=...)"
	@echo "  make sae             Run SAE workflow              (SAE_ARGS=...)"
	@echo "  make localise        Run localisation workflow     (LOC_ARGS=...)"
	@echo ""
	@echo "Notebooks:"
	@echo "  make notebook        Launch Jupyter Lab in notebooks/"
	@echo "  make notebook-strip  Strip output cells from notebooks/*.ipynb"
	@echo ""
	@echo "Playground:"
	@echo "  make app             Launch the Streamlit playground at localhost:8501"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove cache and build artifacts"

# Installation targets
install:
	uv sync --extra dev

install-prod:
	uv sync

sync:
	uv sync --all-extras

# Linting and formatting
lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lint
	uv run ruff format --check .

# Pre-commit hooks
init:
	uv run pre-commit install

pre-commit:
	uv run pre-commit run --all-files

# Testing targets
test:
	uv run pytest tests/

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-cov:
	uv run pytest tests/ --cov=t2i_interp --cov=utils --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Workflow targets — each forwards `*_ARGS` as Hydra overrides.
# Example: make steer STEER_ARGS="model=sdxl_turbo alpha=20"
STEER_ARGS ?=
STITCH_ARGS ?=
SAE_ARGS ?=
LOC_ARGS ?=

steer:
	t2i-steer $(STEER_ARGS)

stitch:
	t2i-stitch $(STITCH_ARGS)

sae:
	t2i-sae $(SAE_ARGS)

localise:
	t2i-localise $(LOC_ARGS)

# Notebooks
notebook:
	uv run jupyter lab --notebook-dir=notebooks

notebook-strip:
	uv run nbstripout notebooks/*.ipynb
	@echo "Stripped output cells from all notebooks."

# Streamlit playground (no-code GUI for the four workflows)
app:
	uv run streamlit run app/streamlit_app.py

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/
	@echo "Cleaned up cache and build artifacts"

