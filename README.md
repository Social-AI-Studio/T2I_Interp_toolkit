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

## Interactive playground (Streamlit)

No-code GUI for the four workflows. Launches a local web app at
`http://localhost:8501` with one page per workflow + a fingerprint browser:

```bash
make app
```

Each page exposes the same config knobs as the CLI (model preset, device,
dtype, prompts, intervention strength, etc.), runs the underlying
`t2i-*` command, streams the log live, and shows the generated images +
the run's reproducibility fingerprint. Pages:

- **Localisation** — pick a UNet layer + head, scale it, see the effect.
- **Steering** — train CAA / K-Steer / LoReFT, sweep alpha.
- **Stitching** — train an MLP mapper across activation spaces.
- **SAE** — discover top-activating sparse features and modulate them.
- **Fingerprints** — browse every past run's hash, model, seed, git SHA.

## Model presets

Pick a model with one Hydra override — its CFG scale, denoising steps, and
dtype defaults compose in automatically:

```bash
t2i-steer model=sd15          # Stable Diffusion 1.5, CFG-guided, 30 steps
t2i-steer model=sdxl          # SDXL base, CFG-guided, 30 steps
t2i-steer model=sdxl_turbo    # SDXL-Turbo, CFG-free, 4 steps
```

Same syntax for the other three workflows:

```bash
t2i-stitch model=sdxl
t2i-sae model=sdxl_turbo
t2i-localise model=sd15
```

Add a new preset by dropping a YAML in
[t2i_interp/config/model/](t2i_interp/config/model/).

## Reproducibility & run fingerprints

Every workflow run writes a `fingerprint.json` next to its output images:

```json
{
  "fingerprint_hash": "875af5b2e5d8223e",
  "workflow": "steer",
  "model_id": "stabilityai/sdxl-turbo",
  "dataset_id": "nirmalendu01/spectacles-bias-prompts-headshot",
  "seed": 42,
  "intervention": {
    "steer_type": "loreft",
    "alpha": 1.0,
    "layer_names": ["unet.up_blocks.1.attentions.0.transformer_blocks.0.attn2"]
  },
  "git_sha": "f44e1c2…",
  "git_dirty": false,
  "config": { "...full resolved Hydra config..." }
}
```

The 16-char `fingerprint_hash` is **machine-independent** — the same logical
experiment from your laptop and a CUDA cluster produces the same hash. When a
W&B run is active, the fingerprint is also uploaded as an artifact and surfaced
in the run summary as `fingerprint/hash`, so you can filter sweeps by it. The
JSON is written *before model load*, so even crashed runs leave a record of
what was attempted.

To set a seed:

```bash
t2i-steer seed=42
```

This seeds Python `random`, NumPy, and Torch (CPU + CUDA) globally before
model load.

## Reproducing Figure 2 (SDXL-Turbo + LoReFT spectacles)

Figure 2 is a **sweep** over hook-site groups (down / mid / up cross-attention
blocks), reporting CLIP score, FID and LPIPS per cell. A single-cell run uses
the loreft config defaults:

```bash
t2i-steer --config-name=steer/loreft model=sdxl_turbo
```

To reproduce the actual sweep, launch a Hydra multirun over the three layer
groups (the same partitioning the paper plots: `unet.down_blocks` /
`unet.mid_block` / `unet.up_blocks`):

```bash
bash bash/run_loreft_macro_sweep.sh
```

That script sets `model=sdxl_turbo` and sweeps `layer_names` across all
down / mid / up cross-attn blocks in one `-m` invocation, logging
CLIP / FID / LPIPS to W&B so the per-cell metrics from the paper are
directly comparable.

Sweep alpha alone (within one layer group):

```bash
t2i-steer --config-name=steer/loreft model=sdxl_turbo -m alpha=5,10,20
```

## Apple Silicon (MPS)

Override device and dtype:

```bash
t2i-steer model=sdxl_turbo device=mps dtype=bfloat16
```

`bfloat16` is more numerically stable on MPS than `float16` for SDXL-Turbo.

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
