# T2I-Interp Toolkit

A **text-to-image interpretation** toolkit for analyzing and steering diffusion models using Sparse Autoencoders (SAEs) and activation steering.

## Features

- **Activation Steering**: Learn mappers/classifiers from UNet activations and apply interventions
- **SAE Analysis**: Train and analyze sparse autoencoders on diffusion model activations  
- **Flexible Hooks**: Hook into any layer of diffusion models (UNet, FLUX transformers, encoders)
- **Clean Workflows**: Generator-based training API with live logging (tqdm, W&B, file logs)
- **Organized Outputs**: Structured run folders with metadata, checkpoints, and artifacts

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (including dev tools like ruff, pytest)
make install

# Or manually with uv
uv sync --extra dev

# Activate the virtual environment (optional - you can use 'uv run' instead)
source .venv/bin/activate
```

### Using pip

```bash
pip install -e ".[dev]"
```

## Using uv

**Option 1: Activate the environment** (traditional way)
```bash
source .venv/bin/activate
# Now you can run commands directly
ruff check .
pytest tests/
```

**Option 2: Use `uv run`** (no activation needed)
```bash
uv run ruff check .
uv run pytest tests/
```

The Makefile uses `uv run` so you can use `make` commands without activating the environment.

### Optional: Authentication

```bash
huggingface-cli login  # For dataset access
wandb login            # For experiment tracking
```

## Quick Start

### Training a Steering Vector

```bash
python -m scripts.train_pipeline \
  --training_fn KSteer.fit \
  --run_name steer_fairface_test \
  --dataset nirmalendu01/fairface-trainval-race-balanced-200 \
  --accessor_path 'model.unet_2.down_attn_blocks[0].self_attn_out' \
  --input_dim 1310720 \
  --hidden_dim 4096 \
  --output_dim 7 \
  --steps 1000 \
  --lr 1e-5 \
  --refresh_batch_size 64 \
  --out_batch_size 16 \
  --training_device cuda:0 \
  --data_device cpu \
  --autocast_dtype bfloat16 \
  --preprocess_fn scripts.train_pipeline:preprocess_fn \
  --gt_processing_fn scripts.train_pipeline:race_processing_fn \
  --wandb_config reporting/config.yaml
```

### Using the Training Script

For training on multiple layers, use `scripts/train_pipeline.sh`:

```bash
bash scripts/train_pipeline.sh
```

## Development

### Setup Development Environment

```bash
make install  # Installs all dependencies including dev tools
```

### Running Tests

```bash
make test           # Run all tests
make test-unit      # Run unit tests only
make test-integration  # Run integration tests only
make test-cov       # Run tests with coverage report
```

### Code Quality

```bash
make lint           # Run ruff linter
make format         # Format code with ruff
make check          # Run all checks (lint + type check)
```

### Running Experiments

```bash
# Training experiment
make train DATASET=nirmalendu01/fairface-trainval-race-balanced-200

# Inference experiment  
make infer RUN_NAME=my_experiment
```

## Project Structure

```
.
├── t2Interp/              # Core library
│   ├── T2I.py            # Main model wrapper
│   ├── accessors.py      # Layer access utilities
│   ├── intervention.py   # Steering interventions
│   ├── sae.py            # Sparse autoencoder
│   └── ...
├── utils/                # Utilities
│   ├── buffer.py         # Activation buffers
│   ├── training.py       # Training loops
│   └── ...
├── reporting/            # Logging and tracking
│   ├── base.py          # Base updater interface
│   └── wandb.py         # W&B integration
├── scripts/             # Training scripts
│   ├── train_pipeline.py
│   └── infer_pipeline.py
├── tests/               # Test suite
└── config/              # Model configurations
```

## Core Concepts

### Accessors

Accessors are dotted paths to model submodules:
- `model.unet_2.down_attn_blocks[0].self_attn_out`
- `model.unet_2.mid_cross_attn_out`

See `t2Interp/accessors.py` for the full API.

### Workflows

All workflows expose a consistent API:

```python
def fit(...) -> Generator[TrainUpdate, None, Output]:
    """
    Yields: TrainUpdate objects with step metrics
    Returns: Output object with final results and checkpoints
    """
```

Available workflows:
- `KSteer.fit` - Steering vector training
- `SAE.fit` - Sparse autoencoder training

### Output Structure

Each run creates a structured directory:

```
runs/<run_name>/
├── run_metadata.json    # Hyperparameters, config
├── logs/
│   └── run.log         # Training logs
├── artifacts/          # Checkpoints, models
└── viz/                # Visualizations (optional)
```

## Tips

- **Precision**: Use `--autocast_dtype bfloat16` for stable mixed-precision training
- **Batch Sizes**: Increase `--out_batch_size` for better GPU utilization
- **Unique Run Names**: Append layer names to `--run_name` for easier tracking
- **File Logs**: Check `runs/<run_name>/logs/run.log` for detailed per-run logs

## License

MIT
