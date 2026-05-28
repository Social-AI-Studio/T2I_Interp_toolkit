"""Glossary. Vocabulary every page of this app uses."""

from __future__ import annotations

import streamlit as st

from app.lib import render_app_footer

st.set_page_config(page_title="Glossary • T2I-Interp", layout="wide")

st.title("Glossary")
st.caption("Quick reference for terms used across the app and in the paper.")

st.markdown(
    "Quick reference for terms used on the other pages and in the paper. "
    "Most have entries in Table 4 of the paper too."
)

# ── Diffusion basics ─────────────────────────────────────────────────────────

st.subheader("Diffusion basics")
st.markdown(
    """
- **T2I**: text-to-image. Models like SD 1.5, SDXL, SDXL-Turbo, FLUX.
- **UNet**: the network that does the actual denoising. Has *down blocks*,
  a *mid block*, and *up blocks*. Each block contains attention layers
  that cross-attend between the noisy image and the text prompt.
- **CFG (classifier-free guidance)**: at each denoising step the model
  runs twice (with and without the prompt). The two outputs get mixed by
  a scalar called `guidance_scale`. Higher means more prompt adherence
  plus more artifacts. SDXL-Turbo skips CFG entirely (`guidance_scale=0`).
- **Inference steps**: how many denoising steps. SD 1.5 typically uses 30
  to 50. SDXL-Turbo distills down to 4. More steps means slower, slightly
  better quality, with diminishing returns past 30.
- **VAE**: the autoencoder that maps between pixel space (512x512x3) and
  latent space (64x64x4). The UNet operates in latent space. The VAE
  decoder turns latents into images at the end.
"""
)

# ── Attention machinery ─────────────────────────────────────────────────────

st.subheader("Attention machinery")
st.markdown(
    """
- **attn1**: self-attention inside the image. Attends image patches to
  other image patches. Determines spatial coherence.
- **attn2**: *cross*-attention from image to text. Each image patch
  attends to every text token. This is where the prompt influences the
  image, and where most interpretability work lives.
- **Head**: multi-head attention splits each layer into N parallel heads
  (typically 8 for SD 1.5). Different heads often specialise. One head
  might bind colour words, another might bind object shape. Localisation
  scales individual heads to test this.
- **Hook site**: a specific module path (e.g.
  `unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2`) where
  the toolkit attaches a forward hook to capture or modify activations.
"""
)

# ── Steering vocabulary ─────────────────────────────────────────────────────

st.subheader("Steering vocabulary")
st.markdown(
    """
- **Steering vector**: a learned direction in activation space that, when
  added to a layer's output, nudges generation toward a target concept.
- **Alpha (α)**: the strength multiplier on the steering vector at
  inference. `α=0` means no effect. `α=10` is a strong push. Negative
  alpha inverts the direction.
- **CAA**: Contrastive Activation Addition. Compute the steering vector
  as `mean(positive_acts) - mean(negative_acts)` from labelled examples.
  Simple, fast, training-free.
- **K-Steer**: trains an MLP classifier on activations to predict the
  concept, then uses its weights as the steering direction. Slightly
  more expressive than CAA for multi-class steering.
- **LoReFT**: Low-rank Representation Fine-tuning. A tiny rank-r adapter
  (thousands of parameters) is trained to add a delta to activations.
  More expressive than a single vector. Works at multiple sites at once.
  Used for the paper's headline spectacles result.
"""
)

# ── SAE vocabulary ──────────────────────────────────────────────────────────

st.subheader("SAE (Sparse Autoencoder) vocabulary")
st.markdown(
    """
- **SAE**: an over-complete autoencoder trained to reconstruct
  activations using a *sparse* combination of features (typically only
  K out of about 5000 features active per token). The goal is to convert
  dense, polysemantic activations into sparse, interpretable ones.
- **Top-K**: the sparsity constraint where only the K largest
  activations pass through (default K=10 for sdxl-unbox). Forces the
  model to use only the most relevant features.
- **Hidden dim**: the SAE's feature dictionary size (5120 for sdxl-unbox).
  Each feature is one direction in the dictionary.
- **Strength**: scaling factor applied to a single feature at generation
  time. Positive amplifies the concept. Negative suppresses it. Used to
  test what each feature encodes.
"""
)

# ── Stitching vocabulary ────────────────────────────────────────────────────

st.subheader("Stitching vocabulary")
st.markdown(
    """
- **Mapper**: a small neural network (typically a 2-layer MLP) that
  takes activations from `layer_a` and outputs activations shaped like
  `layer_b`.
- **Source and target**: `model_a` owns `layer_a` (source of
  activations). `model_b` owns `layer_b` (where mapped activations get
  injected).
- **Single-model vs cross-model**: when `model_a == model_b`, you map
  between internal layers of the same model. When they differ, you
  transfer activations across different models (e.g. base SD 1.5 to a
  LoRA fine-tune of SD 1.5). Paper §3.3 case study uses the cross-model
  variant.
- **Inject steps**: which denoising steps to apply the mapped
  activation. Usually just `[0]` (the first step) is enough. Mapping at
  every step is more expensive and often unnecessary.
"""
)

# ── Reproducibility vocabulary ──────────────────────────────────────────────

st.subheader("Reproducibility vocabulary")
st.markdown(
    """
- **Fingerprint**: a 16-character SHA hash of (model, dataset, seed,
  intervention). Two runs with the same fingerprint represent the same
  logical experiment. Machine-independent. Same hash on your laptop and
  on a CUDA cluster. Paper §2 reproducibility paragraph.
- **Hydra**: the config framework all four workflows use. Lets you
  override any config field from the CLI: `t2i-steer alpha=20`.
- **Hydra preset**: a YAML file under `config/model/` that bundles a
  set of overrides. `model=sdxl_turbo` swaps in `model_key`, `dtype`,
  `guidance_scale`, `num_inference_steps`, and so on, all at once.
"""
)

render_app_footer()
