#!/usr/bin/env bash
set -euo pipefail

# Edit once:
# ACCESSOR='model.unet_2.down_attn_blocks[0].self_attn_out'
# DATASET='nirmalendu01/fairface-trainval-race-balanced-200'
DATASET='nirmalendu01/socialcounterfactuals-200'

# Example grid
declare -a ACCESSORS=(
  "model.text_encoder_2.blocks[0].out_"
  "model.text_encoder_2.blocks[1].out_"
  "model.text_encoder_2.blocks[2].out_"
  "model.text_encoder_2.blocks[3].out_"
  "model.text_encoder_2.blocks[4].out_"
  "model.text_encoder_2.blocks[5].out_"
  "model.text_encoder_2.blocks[6].out_"
  "model.text_encoder_2.blocks[7].out_"
  "model.text_encoder_2.blocks[8].out_"
  "model.text_encoder_2.blocks[9].out_"
  "model.text_encoder_2.blocks[10].out_"
  "model.text_encoder_2.blocks[11].out_"
  # "model.unet_2.down_attn_blocks[0].self_attn_out"
  # "model.unet_2.down_attn_blocks[0].cross_attn_out"
  # "model.unet_2.down_attn_blocks[1].self_attn_out"
  # "model.unet_2.down_attn_blocks[1].cross_attn_out"
  # "model.unet_2.down_attn_blocks[2].self_attn_out"
  # "model.unet_2.down_attn_blocks[2].cross_attn_out"
  # "model.unet_2.down_attn_blocks[3].self_attn_out"
  # "model.unet_2.down_attn_blocks[3].cross_attn_out"
  # "model.unet_2.down_attn_blocks[4].self_attn_out"
  # "model.unet_2.down_attn_blocks[4].cross_attn_out"
  # "model.unet_2.down_attn_blocks[5].self_attn_out"
  # "model.unet_2.down_attn_blocks[5].cross_attn_out"

)
# INPUT_DIM=$((4096*320))
INPUT_DIM=$((768*77))
# MAPPER_KW="{\"input_dim\": ${INPUT_DIM}, \"hidden_dim\": 4096, \"output_dim\": 7}"

run_name='steer_mlp_train'
# hidden_dim=4096
hidden_dim=256
train_steps=30
DENOISER_STEPS='[10]' 
autocast_dtype='bfloat16'
MAPPER_KW="{\"input_dim\": ${INPUT_DIM}, \"hidden_dim\": ${hidden_dim}, \"output_dims\": [7,2]}"
USE_MEMMAP=1
CACHE_ACTIVATIONS=1
GT_COLS="['race','gender']"
DATASET_COL="caption" 
for ACCESSOR in "${ACCESSORS[@]}"; do
  # Parse key=val pairs into vars
#   eval "$cfg"

  python -m scripts.train_pipeline \
    --training_fn scripts.train_pipeline:run_ksteer_fit \
    --dataset "$DATASET" \
    --val_split "validation" \
    --denoiser_steps "${DENOISER_STEPS}" \
    --accessor_path "$ACCESSOR" \
    --run_name "$run_name" \
    --train_steps "${train_steps}" \
    --refresh_batch_size 64 \
    --out_batch_size 4 \
    --training_device cuda:0 \
    --data_device cpu \
    --autocast_dtype "${autocast_dtype}" \
    --outputs_root runs \
    --wandb_config reporting/config.yaml \
    --gt_processing_fn scripts.train_pipeline:gt_processing_fn \
    --mapper twohead_mlp \
    --mapper-kwargs "${MAPPER_KW}" \
    --loss torch.nn.CrossEntropyLoss \
    --updaters wandb \
    --updaters file \
    --ground_truth_column "${GT_COLS}" \
    --dataset_column "${DATASET_COL}" \
    --optim torch.optim.Adam --optim-kwargs '{"lr":1e-5}' \
    $( (( USE_MEMMAP )) && echo "--use_memmap" ) \
    $( (( CACHE_ACTIVATIONS )) && echo "--cache_activations" ) 
done