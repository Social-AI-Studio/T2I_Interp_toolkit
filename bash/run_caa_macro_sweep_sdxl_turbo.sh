#!/bin/bash

set -e

LAYERS=(
  unet.up_blocks.0.attentions.0.transformer_blocks.0.attn2
)

t2i-steer --config-name=steer/caa -m \
  "model_key=stabilityai/sdxl-turbo" \
  "layer_names=[${LAYERS}]" \
  "alpha=5,10,20" \
  "wandb.project=steer-caa-macro-sweep-sdxl-turbo" \
  "guidance_scale=0.0" \
  "conditional_only=false" \
  "capture_step_index=all" \
  "num_inference_steps=4" \
  "steer_steps=4" \
  "delete_cache=false" \
  "save_dir=./latents_cache/caa_steering_sdxl_turbo" \
  "output_dir=./output_images/caa_steer_sdxl_turbo" \
  "$@"
