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
run_name='steer_mlp_train'
hidden_dim=4096
steps=100
lr=1e-5
autocast_dtype='bfloat16'

for ACCESSOR in "${ACCESSORS[@]}"; do
  # Parse key=val pairs into vars
#   eval "$cfg"

  python -m scripts.train_pipeline \
    --training_fn KSteer.fit \
    --dataset "$DATASET" \
    --accessor_path "$ACCESSOR" \
    --run_name "$run_name" \
    --input_dim $((4096*320)) \
    --hidden_dim "${hidden_dim}" \
    --output_dim 7 \
    --steps "${steps}" \
    --lr "${lr}" \
    --refresh_batch_size 64 \
    --out_batch_size 16 \
    --training_device cuda:0 \
    --data_device cpu \
    --autocast_dtype "${autocast_dtype}" \
    --outputs_root runs \
    --wandb_config reporting/config.yaml \
    --preprocess_fn scripts.train_pipeline:preprocess_fn \
    --gt_processing_fn scripts.train_pipeline:race_processing_fn
done
