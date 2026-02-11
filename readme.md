# T2I-Interp Toolkit

A **text-to-image interpretation** toolkit for analyzing and steering diffusion models using Sparse Autoencoders (SAEs) and activation steering.

## Features

- **SAE Analysis**: Train and analyze sparse autoencoders on diffusion model activations using the new `.edit()` based `SAEManagerEdit`
- **Activation Steering**: Learn mappers/classifiers from model activations and apply interventions
- **Feature Scaling**: Ablate or amplify individual SAE features with `SAEFeatureScalingIntervention`
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

# Activate the virtual environment
source .venv/bin/activate
```

### Using pip

```bash
pip install -e ".[dev]"
```

### Optional: Authentication

```bash
huggingface-cli login  # For model/dataset access
wandb login            # For experiment tracking
```

## Quick Start

### Using SAEManagerEdit (recommended)

```python
from t2Interp import T2IModel, SAEManagerEdit

# Load model
model = T2IModel("stabilityai/sdxl-turbo", device="cuda", dtype="float16")

# Create SAE manager and add a trained SAE
manager = SAEManagerEdit(model)
sae_block = manager.add_sae(
    target_accessor=model.unet.mid_block.attentions[0],
    sae=my_trained_sae,
    name="mid_attn_sae",
)

# Apply edits to wire SAE into computation flow
manager.apply_edits()

# Trace SAE activations during generation
with manager.edited_model.generate("A mountain landscape", validate=False, scan=False):
    encoder_out = sae_block.encoder_out.value.save()

# encoder_out now contains the SAE feature activations
print(encoder_out.shape)
```

### Feature Ablation / Amplification

```python
from t2Interp import SAEFeatureScalingIntervention, run_intervention

# Ablate specific features (set to 0)
ablation = SAEFeatureScalingIntervention(
    model=model,
    accessors=[sae_block.encoder_out],
    feature_indices=[42, 128, 256],
    scale=0.0,  # 0.0 = ablate, >1.0 = amplify
)

# Run with intervention
output = run_intervention(model, ["A mountain"], interventions=[ablation])
```

### Backward-compatible API

```python
# Old SAEManager interface still works via SAEManagerEdit
manager = SAEManagerEdit(model)
blocks = manager.add_saes_to_model(
    sae_list=[(model.unet.conv_out, sae, "conv_sae")],
    diff=True,  # SAE processes (output - input) residuals
)

# Run with cache
cache = manager.run_with_cache("A mountain", num_inference_steps=1, seed=42)
```

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

## Development

### Running Tests

```bash
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests only (requires GPU/model)
make test-cov          # Run tests with coverage report
```

### Code Quality

```bash
make lint              # Run ruff linter
make format            # Format code with ruff
make check             # Run all checks (lint + type check)
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
├── t2Interp/                # Core library
│   ├── T2I.py               # Main T2IModel wrapper
│   ├── accessors.py         # ModuleAccessor for layer I/O access
│   ├── blocks.py            # SAEBlock, TransformerBlock, UnetTransformerBlock
│   ├── intervention.py      # Intervention classes (steering, scaling, feature ablation)
│   ├── sae_edit.py          # SAEManagerEdit (.edit() based - recommended)
│   ├── sae.py               # SAEManager (hook-based - deprecated)
│   ├── mapper.py            # MLPMapper, MLPMapperTwoHeads
│   ├── concept_search.py    # KSteer, CAA steering classes
│   ├── unet.py              # UNet component wrappers
│   ├── clip_encoder.py      # CLIP encoder wrapper
│   ├── t5_encoder.py        # T5 encoder wrapper
│   └── flux_*.py            # FLUX transformer support
├── utils/                   # Utilities
│   ├── buffer.py            # Activation buffers
│   ├── training.py          # Training loops
│   ├── inference.py         # Inference utilities
│   └── output_manager.py    # Output management
├── reporting/               # Logging and tracking
│   ├── base.py              # Base updater interface
│   └── wandb.py             # W&B integration
├── scripts/                 # Training/inference scripts
│   ├── train_pipeline.py
│   └── infer_pipeline.py
├── tests/                   # Test suite
│   ├── unit/                # Unit tests (no model required)
│   └── integration/         # Integration tests (requires model)
├── notebooks/               # Example walkthroughs
│   ├── walkthrough.ipynb    # Original walkthrough
│   └── walkthrough-2.ipynb  # SAEManagerEdit walkthrough
├── config/                  # Model configurations
└── dictionary_learning/     # SAE training submodule
```

## Core Concepts

### SAEManagerEdit vs SAEManager

| Feature | SAEManagerEdit (new) | SAEManager (deprecated) |
|---------|---------------------|------------------------|
| Approach | `.edit()` based | Hook-based |
| Tracing | Direct value access | Separate activation dict |
| Interventions | All intervention classes | Hook-only interventions |
| Diff mode | Supported | Supported |
| Multiple SAEs | Supported | Supported |

### Interventions

| Class | Description | Compatible with SAEManagerEdit |
|-------|-------------|-------------------------------|
| `SAEFeatureScalingIntervention` | Scale/ablate specific SAE features | Yes (recommended) |
| `AddVectorIntervention` | Add steering vector to activations | Yes |
| `ReplaceIntervention` | Replace activations with a vector | Yes |
| `ScalingAttentionIntervention` | Scale attention heads | Yes |
| `EncoderAttentionIntervention` | Intervene on encoder attention | Yes |
| `FeatureIntervention` | Scale features via hooks | No (use SAEFeatureScaling) |

### Accessors

ModuleAccessors provide unified I/O access to model submodules:

```python
from t2Interp import ModuleAccessor, IOType

# Access the output of a module
accessor = ModuleAccessor(model.unet.conv_out, "conv_out", IOType.OUTPUT)

# In a trace context:
value = accessor.value       # read
accessor.value = new_value   # write
```

### Output Structure

Each run creates a structured directory:

```
runs/<run_name>/
├── run_metadata.json    # Hyperparameters, config
├── logs/
│   └── run.log          # Training logs
├── artifacts/           # Checkpoints, models
└── viz/                 # Visualizations (optional)
```

## Tips

- **Precision**: Use `--autocast_dtype bfloat16` for stable mixed-precision training
- **Batch Sizes**: Increase `--out_batch_size` for better GPU utilization
- **Unique Run Names**: Append layer names to `--run_name` for easier tracking
- **File Logs**: Check `runs/<run_name>/logs/run.log` for detailed per-run logs
- **Mac/MPS**: The toolkit supports MPS devices for Mac development

## License

MIT
