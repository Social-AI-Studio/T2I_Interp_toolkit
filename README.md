# T2I-Interp Toolkit

A text-to-image interpretability toolkit for steering, SAE analysis, stitching, and cross-attention localisation in diffusion models.

## Citation

If you use this toolkit in your research, please cite our paper:

> **DreamReader: An Interpretability Toolkit for Text-to-Image Models**  
> Nirmalendu Prakash, Narmeen Oozeer, Michael Lan, Luka Samkharadze, Phillip Howard, Roy Ka-Wei Lee, Dhruv Nathawani, Shivam Raval, Amirali Abdullah (2026).  
> [arXiv:2603.13299](https://arxiv.org/abs/2603.13299)

```bibtex
@misc{prakash2026dreamreaderinterpretabilitytoolkittexttoimage,
      title={DreamReader: An Interpretability Toolkit for Text-to-Image Models}, 
      author={Nirmalendu Prakash and Narmeen Oozeer and Michael Lan and Luka Samkharadze and Phillip Howard and Roy Ka-Wei Lee and Dhruv Nathawani and Shivam Raval and Amirali Abdullah},
      year={2026},
      eprint={2603.13299},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.13299}
}
```

## Features

- Activation steering over UNet modules
- Sparse autoencoder (SAE) analysis workflows
- Latent stitching across layers
- Cross-attention localisation sweeps
- Hydra-driven config and multirun support

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). Install it first:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# or via Homebrew
brew install uv
```

Then from the repository root:

```bash
make install        # dev environment (uv sync --extra dev)
# or
make install-prod   # runtime only
make sync           # dev + ray + notebook extras
```

This creates a local `.venv/` with all dependencies pinned by `uv.lock`.

Optional auth for datasets / experiment tracking:

```bash
huggingface-cli login
wandb login
```

## CLI Quickstart

After `make install`, either activate the venv or prefix commands with `uv run`:

```bash
source .venv/bin/activate    # then t2i-steer, t2i-stitch, ...
# or
uv run t2i-steer             # no activation needed
```

Both invocation styles are supported:

```bash
t2i steer
t2i-steer
```

Primary workflows:

```bash
# Steering
t2i-steer
t2i-steer prompt="a cinematic shot of a happy professor" refresh_batch_size=64
t2i-steer -m layer_names="[unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2,unet.mid_block.attentions.0.transformer_blocks.0.attn2]"

# Stitch
t2i-stitch
t2i-stitch prompt="A red car turning into a blue car"

# SAE
t2i-sae
t2i-sae n_top_features=6 num_inference_steps=2

# Localisation
t2i-localise
t2i-localise -m guidance_scale=0.0,2.0,5.0
```

Or via Makefile shortcuts (defaults from each workflow's `run.yaml`):

```bash
make steer
make stitch
make sae
make localise
```

W&B override example:

```bash
t2i-localise wandb.project="attention-ablation" wandb.name="baseline-sweep"
```

## Config Locations

- `t2i_interp/config/steer/run.yaml`
- `t2i_interp/config/stitch/run.yaml`
- `t2i_interp/config/sae/run.yaml`
- `t2i_interp/config/localisation/run.yaml`
- `t2i_interp/config/wandb.yaml`

## Notebooks

- `notebooks/steer.ipynb`
- `notebooks/stitch.ipynb`
- `notebooks/sae.ipynb`
- `notebooks/localisation.ipynb`

## Project Structure

```text
T2I_Interp_toolkit/
├── t2i_interp/
│   ├── cli.py                 # unified `t2i` entry point
│   ├── accessors/             # ModuleAccessor / ModelWrapper
│   ├── hooks/                 # capture / alter hooks
│   ├── config/                # Hydra YAMLs (steer, stitch, sae, localisation)
│   ├── scripts/               # run_steer / run_stitch / run_sae / run_localisation
│   ├── reporting/             # W&B integration, sweep reports
│   ├── utils/                 # T2I helpers, metrics, plotting, training
│   ├── linear_steering.py     # CAA, KSteer, LoReFT
│   ├── loreft.py              # LoReFTLayer
│   ├── sae.py                 # SAEManager
│   ├── stitch.py              # Stitcher (mapper, graft, diffusion lens)
│   └── t2i.py                 # T2IModel pipeline wrapper
├── dictionary_learning/       # vendored SAE training library
├── bash/                      # convenience sweep launchers
├── notebooks/                 # workflow walkthroughs
├── tests/                     # unit + integration
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Development

```bash
git clone https://github.com/Social-AI-Studio/T2I_Interp_toolkit.git
cd T2I_Interp_toolkit
make install
make init           # install pre-commit hooks (one-time)
```

Before opening a PR:

```bash
make format         # ruff format + ruff check --fix
make lint           # ruff check (no fixes)
make check          # lint + format-check (CI-equivalent)
make test           # pytest tests/
make test-cov       # with coverage report
```

All Makefile targets:

```bash
make help
```

## License

MIT
