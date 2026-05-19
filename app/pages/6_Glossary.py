"""Glossary — vocabulary every page of this app uses."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Glossary • T2I-Interp", layout="wide")

st.title("Glossary")

st.markdown(
    "Quick reference for terms used on the other pages and in the paper. "
    "Most have entries in Table 4 of the paper too."
)

# ── Diffusion basics ─────────────────────────────────────────────────────────

st.subheader("Diffusion basics")
st.markdown(
    """
- **T2I** — text-to-image. Models like SD 1.5, SDXL, SDXL-Turbo, FLUX.
- **UNet** — the network that does the actual denoising. Has *down blocks*,
  a *mid block*, and *up blocks*. Each block contains attention layers
  that cross-attend between the noisy image and the text prompt.
- **CFG (classifier-free guidance)** — at each denoising step the model
  runs twice (with/without the prompt) and the outputs are mixed by a
  scalar called `guidance_scale`. Higher = more prompt adherence + more
  artifacts. SDXL-Turbo skips CFG entirely (`guidance_scale=0`).
- **Inference steps** — how many denoising steps. SD 1.5 typically uses
  30-50, SDXL-Turbo distills to 4. More steps = slower, slightly better
  quality, diminishing returns past 30.
- **VAE** — the autoencoder that maps between pixel space (512×512×3) and
  latent space (64×64×4). The UNet operates in latent space; the VAE
  decoder turns latents into images at the end.
"""
)

# ── Attention machinery ─────────────────────────────────────────────────────

st.subheader("Attention machinery")
st.markdown(
    """
- **attn1** — self-attention inside the image. Attends image patches to other
  image patches. Determines spatial coherence.
- **attn2** — *cross*-attention from image to text. Each image patch
  attends to every text token. **This is where the prompt influences
  the image** — and where most interpretability work lives.
- **Head** — multi-head attention splits each layer into N parallel heads
  (typically 8 for SD 1.5). Different heads often specialise (e.g. one
  head binds "color words", another binds "object shape"). Localisation
  scales individual heads to test this.
- **Hook site** — a specific module path (e.g.
  `unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2`) where the
  toolkit attaches a forward hook to capture or modify activations.
"""
)

# ── Steering vocabulary ─────────────────────────────────────────────────────

st.subheader("Steering vocabulary")
st.markdown(
    """
- **Steering vector** — a learned direction in activation space that, when
  added to a layer's output, nudges generation toward a target concept.
- **Alpha (`α`)** — the strength multiplier on the steering vector at
  inference. `α=0` = no effect; `α≈10` is a strong push; negative `α`
  inverts the direction.
- **CAA** — Contrastive Activation Addition. Compute the steering vector
  as `mean(positive_acts) − mean(negative_acts)` from labelled examples.
  Simple, fast, training-free.
- **K-Steer** — Trains an MLP classifier on activations to predict the
  concept, then uses its weights as the steering direction. Slightly
  more expressive than CAA for multi-class steering.
- **LoReFT** — Low-rank Representation Fine-tuning. A tiny rank-r adapter
  (~thousands of parameters) is trained to add a delta to activations.
  More expressive than a single vector and works at multiple sites at
  once. Used for the paper's headline spectacles result.
"""
)

# ── SAE vocabulary ──────────────────────────────────────────────────────────

st.subheader("SAE (Sparse Autoencoder) vocabulary")
st.markdown(
    """
- **SAE** — an over-complete autoencoder trained to reconstruct activations
  using a *sparse* combination of features (typically only K out of
  ~5000 features active per token). Goal: convert dense, polysemantic
  activations into sparse, interpretable ones.
- **Top-K** — the sparsity constraint where only the K largest activations
  pass through (default K=10 for sdxl-unbox). Forces the model to use
  the most-relevant features only.
- **Hidden dim** — the SAE's feature dictionary size (5120 for sdxl-unbox).
  Each feature is one direction in the dictionary.
- **Strength** — scaling factor applied to a single feature at generation
  time. Positive = amplify the concept; negative = suppress it. Used to
  test what each feature encodes.
"""
)

# ── Stitching vocabulary ────────────────────────────────────────────────────

st.subheader("Stitching vocabulary")
st.markdown(
    """
- **Mapper** — a small neural network (typically a 2-layer MLP) that takes
  activations from `layer_a` and outputs activations shaped like `layer_b`.
- **Source / target** — `model_a` owns `layer_a` (source of activations);
  `model_b` owns `layer_b` (where mapped activations get injected).
- **Single-model vs cross-model** — `model_a == model_b`: map between
  internal layers of the same model. `model_a != model_b`: transfer
  activations across different models (e.g. base SD 1.5 → LoRA fine-tune
  of SD 1.5). Paper §3.3 case study uses the cross-model variant.
- **Inject steps** — which denoising steps to apply the mapped activation.
  Usually just `[0]` (the first step) is enough; mapping at every step
  is more expensive and often unnecessary.
"""
)

# ── Reproducibility vocabulary ──────────────────────────────────────────────

st.subheader("Reproducibility vocabulary")
st.markdown(
    """
- **Fingerprint** — a 16-char SHA hash of (model + dataset + seed +
  intervention). Two runs with the same fingerprint represent the *same
  logical experiment*. Machine-independent — same hash on your laptop and
  a CUDA cluster. Paper §2 / Reproducibility paragraph.
- **Hydra** — the config framework all four workflows use. Lets you
  override any config field from the CLI: `t2i-steer alpha=20`.
- **Hydra preset** — a YAML file under `config/model/` that bundles a set
  of overrides. `model=sdxl_turbo` swaps in `model_key`, `dtype`,
  `guidance_scale`, `num_inference_steps`, etc. all at once.
"""
)
