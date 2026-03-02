#!/bin/bash
# Launch multirun: down_blocks + mid_block + up_blocks cross-attn CAA steering
# Each job steers at all respective layers simultaneously.
# Results logged to W&B table via WandbMultirunCallback.
#
# Run from project root: bash run_steer_sweep.sh

set -e

# Remove any local dev diffusers checkout from PYTHONPATH to avoid import conflicts
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "LCM_LoRA" | tr '\n' ':' | sed 's/:$//')

# Down blocks cross-attn (6 layers: down_blocks 0,1,2 × 2 attentions each)
LAYERS_DOWN="\
unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2,\
unet.down_blocks.0.attentions.1.transformer_blocks.0.attn2,\
unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2,\
unet.down_blocks.1.attentions.1.transformer_blocks.0.attn2,\
unet.down_blocks.2.attentions.0.transformer_blocks.0.attn2,\
unet.down_blocks.2.attentions.1.transformer_blocks.0.attn2"

# Mid block cross-attn (1 layer)
LAYERS_MID="unet.mid_block.attentions.0.transformer_blocks.0.attn2"

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

# Sweep only layer_names (3 jobs, no cross-product).
# output_dir is auto-suffixed per job in run_steer.py from the block component.
t2i steer -m \
  "layer_names=[${LAYERS_DOWN}],[${LAYERS_MID}],[${LAYERS_UP}]" \
  "wandb.project=steer-module-sweep" \
  "delete_cache=true" \
  "$@"
