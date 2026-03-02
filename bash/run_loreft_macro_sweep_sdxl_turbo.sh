#!/bin/bash
# Launch multirun: cross-attn LoReFT steering for SDXL-Turbo (4-step distilled)
# Architecture is identical to SDXL base — same layer names apply.
#
# Key differences vs standard SDXL sweep:
#   - guidance_scale=0.0   (turbo is CFG-free)
#   - conditional_only=false  (no unconditional half without CFG)
#   - num_inference_steps=4 / steer_steps=4  (~7.5x faster collection)
#   - capture_step_index="all"  (4 steps is small enough for full-step tars)
#
# Run from project root:         bash run_loreft_macro_sweep_sdxl_turbo.sh
# Override alpha at call site:   ALPHA=2.0 bash run_loreft_macro_sweep_sdxl_turbo.sh

set -e

# Edit-strength scale (1.0 = full trained edit, 0.0 = no edit, >1 amplifies)
ALPHA="${ALPHA:-10.0}"

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
  "model_key=stabilityai/sdxl-turbo" \
  "layer_names=[${LAYERS_DOWN}],[${LAYERS_MID}],[${LAYERS_UP}]" \
  "wandb.project=steer-loreft-macro-sweep-sdxl-turbo" \
  "guidance_scale=0.0" \
  "conditional_only=false" \
  "capture_step_index=all" \
  "num_inference_steps=4" \
  "steer_steps=4" \
  "alpha=${ALPHA}" \
  "delete_cache=true" \
  "save_dir=./latents_cache/loreft_steering_sdxl_turbo" \
  "output_dir=./output_images/loreft_steer_sdxl_turbo" \
  "$@"
