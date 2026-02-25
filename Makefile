.PHONY: help install install-dev clean test test-unit test-integration test-cov lint format check train infer init pre-commit

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
	@echo "Experiments:"
	@echo "  make train           Run training pipeline"
	@echo "  make infer           Run inference pipeline"
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

# Experiment targets
DATASET ?= nirmalendu01/fairface-trainval-race-balanced-200
ACCESSOR ?= model.unet_2.down_attn_blocks[0].self_attn_out
RUN_NAME ?= test_run
STEPS ?= 1000

train:
	python -m scripts.train_pipeline \
		--training_fn KSteer.fit \
		--run_name $(RUN_NAME) \
		--dataset $(DATASET) \
		--accessor_path '$(ACCESSOR)' \
		--input_dim $$((4096*320)) \
		--hidden_dim 4096 \
		--output_dim 7 \
		--steps $(STEPS) \
		--lr 1e-5 \
		--refresh_batch_size 64 \
		--out_batch_size 16 \
		--training_device cuda:0 \
		--data_device cpu \
		--autocast_dtype bfloat16 \
		--preprocess_fn scripts.train_pipeline:preprocess_fn \
		--gt_processing_fn scripts.train_pipeline:race_processing_fn \
		--wandb_config t2i_interp/reporting/config.yaml

train-script:
	bash scripts/train_pipeline.sh

infer:
	bash scripts/infer_pipeline.sh

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

