#!/bin/bash
# Launch multirun: cross-attn LoReFT steering for SDXL
# SDXL UNet cross-attention structure (stabilityai/stable-diffusion-xl-base-1.0):
#
#   down_blocks.0  (DownBlock2D)          — no cross-attention
#   down_blocks.1  (CrossAttnDownBlock2D) — 2 attentions × 2  transformer_blocks = 4  layers
#   down_blocks.2  (CrossAttnDownBlock2D) — 2 attentions × 10 transformer_blocks = 20 layers
#   mid_block      (UNetMidBlock2DCrossAttn) — 1 attention × 10 transformer_blocks = 10 layers
#   up_blocks.0    (CrossAttnUpBlock2D)   — 3 attentions × 10 transformer_blocks = 30 layers
#   up_blocks.1    (CrossAttnUpBlock2D)   — 3 attentions × 2  transformer_blocks = 6  layers
#   up_blocks.2    (UpBlock2D)            — no cross-attention
#
# Jobs are grouped by block (3 jobs: down, mid, up) mirroring the SD1.5 sweep.
# capture_step_index=0 is forced to avoid multi-GB tars at SDXL's higher spatial resolutions.
#
# Run from project root: bash run_loreft_macro_sweep_sdxl.sh

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

t2i-steer --config-name=steer/loreft -m \
  "model_key=stabilityai/stable-diffusion-xl-base-1.0" \
  "layer_names=[${LAYERS_DOWN}],[${LAYERS_MID}],[${LAYERS_UP}]" \
  "wandb.project=steer-loreft-macro-sweep-sdxl" \
  "capture_step_index=0" \
  "num_inference_steps=30" \
  "steer_steps=30" \
  "delete_cache=true" \
  "save_dir=./latents_cache/loreft_steering_sdxl" \
  "output_dir=./output_images/loreft_steer_sdxl" \
  "$@"
