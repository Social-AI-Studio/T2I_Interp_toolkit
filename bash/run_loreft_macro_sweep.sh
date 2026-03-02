#!/bin/bash
# Launch multirun: down_blocks + mid_block + up_blocks cross-attn LoReFT steering
# Each job trains and steers at all respective layers simultaneously.
# Results logged to W&B table via WandbMultirunCallback.
#
# Run from project root:         bash run_loreft_macro_sweep.sh
# Override alpha at call site:   ALPHA=2.0 bash run_loreft_macro_sweep.sh

set -e

# Edit-strength scale (1.0 = full trained edit, 0.0 = no edit, >1 amplifies)
ALPHA="${ALPHA:-10.0}"

# Remove any local dev diffusers checkout from PYTHONPATH to avoid import conflicts
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "LCM_LoRA" | tr '\n' ':' | sed 's/:$//')

# SD 1.5: down_blocks.1 and down_blocks.2, each with 2 attentions x 1 transformer block (4 layers)
LAYERS_DOWN=$(echo unet.down_blocks.1.attentions.{0..1}.transformer_blocks.0.attn2 unet.down_blocks.2.attentions.{0..1}.transformer_blocks.0.attn2 | tr ' ' ',')

# SD 1.5: mid_block has 1 attention x 1 transformer block (1 layer)
LAYERS_MID=$(echo unet.mid_block.attentions.0.transformer_blocks.0.attn2 | tr ' ' ',')

# SD 1.5: up_blocks.1 and up_blocks.2, each with 3 attentions x 1 transformer block (6 layers)
LAYERS_UP=$(echo unet.up_blocks.1.attentions.{0..2}.transformer_blocks.0.attn2 unet.up_blocks.2.attentions.{0..2}.transformer_blocks.0.attn2 | tr ' ' ',')

# Sweep only layer_names (3 jobs).
# output_dir is auto-suffixed per job in run_steer.py from the block component.
t2i-steer --config-name=steer/loreft -m \
  "model_key=runwayml/stable-diffusion-v1-5" \
  "layer_names=[${LAYERS_DOWN}],[${LAYERS_MID}],[${LAYERS_UP}]" \
  "wandb.project=steer-loreft-macro-sweep-sd15" \
  "alpha=${ALPHA}" \
  "delete_cache=true" \
  "$@"
