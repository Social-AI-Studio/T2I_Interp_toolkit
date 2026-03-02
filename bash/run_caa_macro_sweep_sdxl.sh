#!/bin/bash
# Launch multirun: CAA (Contrastive Activation Addition) steering for SDXL
# Same layer structure as run_loreft_macro_sweep_sdxl.sh
#
#   down_blocks.1  (CrossAttnDownBlock2D) — 2 attentions × 2  transformer_blocks = 4  layers
#   down_blocks.2  (CrossAttnDownBlock2D) — 2 attentions × 10 transformer_blocks = 20 layers
#   mid_block      (UNetMidBlock2DCrossAttn) — 1 attention × 10 transformer_blocks = 10 layers
#   up_blocks.0    (CrossAttnUpBlock2D)   — 3 attentions × 10 transformer_blocks = 30 layers
#   up_blocks.1    (CrossAttnUpBlock2D)   — 3 attentions × 2  transformer_blocks = 6  layers
#
# Sweeps over both layer blocks AND alpha values (Cartesian product).
# delete_cache=false so tars are reused across alpha jobs for the same block
# (alpha only affects inference, not collection or training).
#
# Run from project root: bash run_caa_macro_sweep_sdxl.sh
# Override alphas:       bash run_caa_macro_sweep_sdxl.sh "alpha=5,20,50"

set -e

# Remove any local dev diffusers checkout from PYTHONPATH to avoid import conflicts
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "LCM_LoRA" | tr '\n' ':' | sed 's/:$//')

# SDXL down blocks with cross-attention
LAYERS_DOWN=$(echo \
  unet.down_blocks.1.attentions.{0..1}.transformer_blocks.{0..1}.attn2 \
  unet.down_blocks.2.attentions.{0..1}.transformer_blocks.{0..9}.attn2 \
  | tr ' ' ',')

# SDXL mid block
LAYERS_MID=$(echo \
  unet.mid_block.attentions.0.transformer_blocks.{0..9}.attn2 \
  | tr ' ' ',')

# SDXL up blocks with cross-attention
LAYERS_UP=$(echo \
  unet.up_blocks.0.attentions.{0..2}.transformer_blocks.{0..9}.attn2 \
  unet.up_blocks.1.attentions.{0..2}.transformer_blocks.{0..1}.attn2 \
  | tr ' ' ',')

t2i-steer --config-name=steer/caa -m \
  "model_key=stabilityai/stable-diffusion-xl-base-1.0" \
  "layer_names=[${LAYERS_DOWN}],[${LAYERS_MID}],[${LAYERS_UP}]" \
  "alpha=5,10,20" \
  "wandb.project=steer-caa-macro-sweep-sdxl" \
  "delete_cache=false" \
  "save_dir=./latents_cache/caa_steering_sdxl" \
  "output_dir=./output_images/caa_steer_sdxl" \
  "$@"
