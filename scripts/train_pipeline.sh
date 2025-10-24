#!/usr/bin/env bash
set -euo pipefail

# Edit once:
# ACCESSOR='model.unet_2.down_attn_blocks[0].self_attn_out'
DATASET='nirmalendu01/fairface-trainval-race-balanced-200'

# Example grid
declare -a ACCESSORS=(
  "model.unet_2.down_attn_blocks[0].self_attn_out"
  "model.unet_2.down_attn_blocks[0].cross_attn_out"
)
INPUT_DIM=$((4096*320))
MAPPER_KW="{\"input_dim\": ${INPUT_DIM}, \"hidden_dim\": 4096, \"output_dim\": 7}"
run_name='steer_mlp_train'
hidden_dim=4096
train_steps=100
DENOISER_STEPS='[10,12,15]' 
autocast_dtype='bfloat16'

for ACCESSOR in "${ACCESSORS[@]}"; do
  # Parse key=val pairs into vars
#   eval "$cfg"

  python -m scripts.train_pipeline \
    --training_fn scripts.train_pipeline:run_ksteer_fit \
    --dataset "$DATASET" \
    --denoiser-steps "${DENOISER_STEPS}" \
    --accessor_path "$ACCESSOR" \
    --run_name "$run_name" \
    --train_steps "${train_steps}" \
    --refresh_batch_size 64 \
    --out_batch_size 16 \
    --training_device cuda:0 \
    --data_device cpu \
    --autocast_dtype "${autocast_dtype}" \
    --outputs_root runs \
    --wandb_config reporting/config.yaml \
    --preprocess_fn scripts.train_pipeline:preprocess_fn \
    --gt_processing_fn scripts.train_pipeline:race_processing_fn \
    --mapper mlp \
    --mapper-kwargs "${MAPPER_KW}" \
    --loss torch.nn.CrossEntropyLoss \
    --updaters wandb \
    --updaters file \
    --optim torch.optim.Adam --optim-kwargs '{"lr":1e-5}'
done
