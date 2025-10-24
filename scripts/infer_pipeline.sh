#!/usr/bin/env bash
set -euo pipefail

DATASET='nirmalendu01/fairface-trainval-race-balanced-200'            # e.g. "yourname/debiasdb" or a local path
DATASET_SPLIT="val"
MODEL_REPO="CompVis/stable-diffusion-v1-4"
MODEL_DEVICE="cuda:0"
MODEL_DTYPE="bfloat16"

ACCESSOR_PATH="model.unet_2.down_attn_blocks[0].self_attn_out"        # <-- replace with your valid accessor expression

WORKFLOW="steering"               # or localisation | stitching | sae
RUN_BASE="infer_race_steering_mlp"

# Updaters: choose any subset, order doesn't matter
WANDB_CONFIG="reporting/config.yaml"
WANDB_RUN_PREFIX="${RUN_BASE}"
LOG_DIR="./logs"                    # file logger directory
OUT_ROOT="./runs"                 # OutputManager root

# Be careful with shell quoting for JSON.
INPUT_DIM=$((4096*320))
MAPPER_KW="{\"input_dim\": ${INPUT_DIM}, \"hidden_dim\": 4096, \"output_dim\": 7}"
MAPPER_CKPT="./runs/steer_mlp_train_steering_20251021-050122/artifacts/best_ckpt.pt"

# Other knobs
STEER_STEPS=1
DENOISING_STEP=0
AUTODTYPE="bfloat16"
REFRESH_BS=64
OUT_BS=4
D_SUBMODULE=$((4096*320))
DATA_DEVICE="cpu"
USE_MEMMAP=0
CACHE_ACTIVATIONS=0

# ---------- THE ALPHA GRID ----------
ALPHAS=(0.25 0.5 0.75 1.0 1.5 2.0)
DENOISER_STEP=10
# Make sure log dir exists
mkdir -p "${LOG_DIR}"

# ---------- LOOP ----------
for A in "${ALPHAS[@]}"; do
  # Sanitize alpha for filenames (e.g., 0.25 -> p0_25)
  A_TAG="a${A//./p}"
  RUN_NAME="${WANDB_RUN_PREFIX}_${A_TAG}"
  LOG_PATH="${LOG_DIR}/${RUN_NAME}.jsonl"

  echo ">>> Running alpha=${A}  run_name=${RUN_NAME}"

  python -m scripts.infer_pipeline \
    --workflow "${WORKFLOW}" \
    --infer_fn scripts.infer_pipeline:run_steering \
    --run_name "${RUN_NAME}" \
    --dataset "${DATASET}" \
    --dataset_split "${DATASET_SPLIT}" \
    --dataset_column "race" \
    --target_idx 1 \
    --preprocess_fn scripts.infer_pipeline:race_preprocess_fn \
    --model_repo "${MODEL_REPO}" \
    --model_dtype "${MODEL_DTYPE}" \
    --accessor_path "${ACCESSOR_PATH}" \
    --mapper "mlp" \
    --mapper-kwargs "${MAPPER_KW}" \
    --mapper_ckpt "${MAPPER_CKPT}" \
    --steer_steps "${STEER_STEPS}" \
    --alpha "${A}" \
    --denoiser_step "${DENOISER_STEP}" \
    --inference_steps 50 \
    --device "${MODEL_DEVICE}" \
    --data_device "${DATA_DEVICE}" \
    --autocast_dtype "${AUTODTYPE}" \
    --refresh_batch_size "${REFRESH_BS}" \
    --out_batch_size "${OUT_BS}" \
    --d_submodule "${D_SUBMODULE}" \
    $( (( USE_MEMMAP )) && echo "--use_memmap" ) \
    $( (( CACHE_ACTIVATIONS )) && echo "--cache_activations" ) \
    --outputs_root "${OUT_ROOT}" \
    --wandb_config "${WANDB_CONFIG}" \
    --log-file "${LOG_PATH}" \
    --updaters file \
    --seed 42 \
    --subset 10

done
