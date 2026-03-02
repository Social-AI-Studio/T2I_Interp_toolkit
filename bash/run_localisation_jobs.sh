#!/bin/bash
# Multi-job launch script for localisation

# Example: Launching Hydra multirun across multiple specific cross-attention layers and heads.
# By passing comma-separated values to the Hydra `-m` (multirun) flag, it will automatically
# spawn a separate job for every combination of layer and head!

LAYERS="up_blocks_1_attentions_0_transformer_blocks_0_attn2_out,\
up_blocks_1_attentions_1_transformer_blocks_0_attn2_out,\
up_blocks_1_attentions_2_transformer_blocks_0_attn2_out,\
up_blocks_2_attentions_0_transformer_blocks_0_attn2_out,\
up_blocks_2_attentions_1_transformer_blocks_0_attn2_out,\
up_blocks_2_attentions_2_transformer_blocks_0_attn2_out,\
up_blocks_3_attentions_0_transformer_blocks_0_attn2_out,\
up_blocks_3_attentions_1_transformer_blocks_0_attn2_out,\
up_blocks_3_attentions_2_transformer_blocks_0_attn2_out"

HEADS="0,1,2,3,4,5,6,7"

# Note: You can also specify environment variables to distribute across GPUs
# e.g., export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Launching localisation sweeps..."

t2i localise -m \
    target_layer=$LAYERS \
    target_heads=$HEADS \
    wandb.project="localisation-sweep" \
    factor=0.0 \
    prompt="a photo of a unicorn"
