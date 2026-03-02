#!/bin/bash
# Launch multirun: LoReFT steering on all up_blocks cross-attn layers
# Each job trains and steers on ONE layer, and all jobs upload to a shared W&B table.
#
# Run from project root: bash run_loreft_sweep.sh

set -e

# Remove any local dev diffusers checkout from PYTHONPATH to avoid import conflicts
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "LCM_LoRA" | tr '\n' ':' | sed 's/:$//')

# Up blocks cross-attn (9 layers: up_blocks 1,2,3 × 3 attentions each)
LAYERS_UP="\
unet.up_blocks.1.attentions.0.transformer_blocks.0.attn2,\
unet.up_blocks.1.attentions.1.transformer_blocks.0.attn2,\
unet.up_blocks.1.attentions.2.transformer_blocks.0.attn2,\
unet.up_blocks.2.attentions.0.transformer_blocks.0.attn2,\
unet.up_blocks.2.attentions.1.transformer_blocks.0.attn2,\
unet.up_blocks.2.attentions.2.transformer_blocks.0.attn2,\
unet.up_blocks.3.attentions.0.transformer_blocks.0.attn2,\
unet.up_blocks.3.attentions.1.transformer_blocks.0.attn2,\
unet.up_blocks.3.attentions.2.transformer_blocks.0.attn2"

# Run a Hydra multirun (-m) using the loreft config.
# Sweep over layer_name using the comma-separated list.
t2i-steer --config-name=steer/loreft -m \
  "layer_names=[${LAYERS_UP}]" \
  "wandb.project=steer-loreft-sweep" \
  "delete_cache=true" \
  "$@"
