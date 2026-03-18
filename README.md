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

```bash
# from repository root
pip install -e .
```

Optional auth for datasets/experiment tracking:

```bash
huggingface-cli login
wandb login
```

## CLI Quickstart

Both command styles are supported:

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
│   ├── cli.py
│   ├── config/
│   │   ├── steer/
│   │   ├── stitch/
│   │   ├── sae/
│   │   └── localisation/
│   ├── scripts/
│   │   ├── run_steer.py
│   │   ├── run_stitch.py
│   │   ├── run_sae.py
│   │   └── run_localisation.py
│   └── utils/
├── bash/
├── notebooks/
└── pyproject.toml
```

## Development

```bash
git clone https://github.com/Social-AI-Studio/T2I_Interp_toolkit.git
cd T2I_Interp_toolkit
pip install -e ".[dev]"
```

Before opening a PR:

```bash
ruff check .
ruff format .
pytest
```

## License

MIT
