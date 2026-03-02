# T2I-Interp Toolkit

A **text-to-image interpretation** toolkit for analyzing and steering diffusion models using Sparse Autoencoders (SAEs), activation steering, stitching, and layer-wise localization.

## Features

- **Activation Steering**: Learn concept mappers/classifiers from UNet activations and apply targeted interventions.
- **SAE Analysis**: Train and analyze sparse autoencoders on diffusion model activations.
- **Latent Stitching**: Collect and stitch latents from different layers to manipulate generated outputs.
- **Localization Sweeps**: Sweep through UNet cross-attention heads and trace perturbations via hooking.
- **Hydra Configuration**: Easily manage configurations, multi-run sweeps, and dynamic architecture parameters via `hydra-core`.
- **Organized Outputs**: Fully structured local run configurations bundled securely as an installable Python package.

---

## Installation

The T2I-Interp toolkit is completely pip-installable, ensuring all executable workflows are immediately available from the terminal.

### Global Editable Installation (Recommended)

To install the toolkit and all subpackages (`t2i_interp`, `config`, `scripts`) into your active Python environment:

```bash
# From the repository root
pip install -e .
```

This registers the local configurations and exposes the four primary workflow commands globally via console script entrypoints.

### Authentication

If evaluating on HF datasets or pushing your own logged sweeps:

```bash
huggingface-cli login  # For dataset access
wandb login            # For experiment tracking
```

---

## Workflows and Executables

Leveraging the power of Hydra, this toolkit runs four primary configurable scripts. All YAML configurations are locally contained within the `config/` directory. 

### Experiment Tracking with Weights & Biases (W&B)

All 4 workflows natively support pushing their generated grids, images, and config dictionaries up to a W&B Dashboard. To enable this, simply override `wandb.project` when executing any run:

```bash
# Single run tracking
t2i-localise wandb.project="attention-ablation" wandb.name="baseline-sweep"

# Multi-run sweeps tracked into W&B automatically
t2i-steer -m layer_name="unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2","unet.mid_block.attentions.0.transformer_blocks.0.attn2" wandb.project="steer-module-sweep"
```

*Note: W&B initialization defaults can also be permanently saved inside `config/wandb.yaml`.*

### 1. Concept Steering (`t2i-steer`)

Train and apply targeted MLP steering maps.

```bash
# Run with default config
t2i-steer

# Override prompt and batch size
t2i-steer prompt="a cinematic shot of a happy professor" refresh_batch_size=64

# Hydra Multi-run Sweeping over target modules
t2i-steer -m layer_name="unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2","unet.mid_block.attentions.0.transformer_blocks.0.attn2"
```
**Config path:** `config/steer/run.yaml`

### 2. Stitching (`t2i-stitch`)

Collect latents across multiple layers, train an `MLPMapper` to stitch them together, and run modified generation.

```bash
# Run with default config
t2i-stitch

# Override the specific concept prompt
t2i-stitch prompt="A red car turning into a blue car"
```
**Config path:** `config/stitch/run.yaml`

### 3. Sparse Autoencoders (`t2i-sae`)

Explore activation landscapes and generate grid-visualizations of structural concepts automatically discovered via imported SAE checkpoints.

```bash
# Run with default config
t2i-sae

# Override the number of plotted features and inference steps
t2i-sae n_top_features=6 num_inference_steps=2
```
*Note: Any loaded SAE checkpoints and spatial dimensions can be freely modified or swapped out from `config/sae/run.yaml` via the `saes` dict structure.*

### 4. Attention Localization (`t2i-localise`)

Apply concept-erasure perturbation loops across model cross-attention heads, generating individual verification grids per layer sweep.

```bash
# Run with default config
t2i-localise

# Run a sweep generating varied guidance scales
t2i-localise -m guidance_scale=0.0,2.0,5.0
```
**Config path:** `config/localisation/run.yaml`

---

## Interactive Notebooks

All workflows have complimentary **Jupyter Notebooks** available in the `notebooks/` directory. These are beneficial for learning how the lower-level PyTorch modules (`T2IModel`, `UNetAlterHook`, `SAEManager`, `Stitcher`, etc.) compose the backend interpretation loops.

- `steer.ipynb` - Step-by-step steering verification and logic.
- `stitch.ipynb` - Pipeline demonstration mapping layers pairwise.
- `sae.ipynb` - Interactive activation capturing and grid creation.
- `walkthrough.ipynb` / `localisation.ipynb` - Demonstration of altering UNet attention modules dynamically.

Before running notebooks directly, verify the package has been fully installed or ensure your kernel is instantiated from the repository root.

---

## Project Structure

```
T2I_Interp_toolkit/
├── config/              # Hydra YAML configurations
│   ├── steer/run.yaml
│   ├── stitch/run.yaml
│   ├── sae/run.yaml
│   └── localisation/run.yaml
├── scripts/             # Core execution entry points
│   ├── run_steer.py     # mapped to 't2i-steer'
│   ├── run_stitch.py    # mapped to 't2i-stitch'
│   ├── run_sae.py       # mapped to 't2i-sae'
│   └── run_localisation.py # mapped to 't2i-localise'
├── t2i_interp/          # Core model logic and hook wrappers
│   ├── T2I.py           # Base text-to-image framework
│   ├── intervention.py  # Alteration hooks and MLP manipulation
│   └── sae.py           # SAE abstraction layers
├── notebooks/           # Interactive learning examples
└── pyproject.toml       # Package metadata and entry points
```

---

## Contributing

We welcome community feedback, pull requests, and ideas for extending the core Text-To-Image interpretability structures.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/T2I_Interp_toolkit.git
cd T2I_Interp_toolkit

# Install in editable mode with development tools
pip install -e ".[dev]"
```

Please run standardized formatting mechanisms (like `ruff`) and execute included tests (`pytest`) before attempting to open PRs.

---

## Open Issues

*(Placeholder for future tracking)*
- [ ] Incorporate comprehensive unit-tests for the text-to-image steering pipelines.
- [ ] Expand mapping algorithms alongside default MLPMapper methodologies.
- [ ] Implement integrated experiment dashboard outputs for Hydra multi-run generation grids.

## License

MIT
