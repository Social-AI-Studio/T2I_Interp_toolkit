#!/bin/bash
# Train cross-model mapper between SD 1.5 (base) and LCM-LoRA SD 1.5 activations.
# Captures mid_block cross-attn at step 0 from both models and trains a mapper.
#
# model_a: runwayml/stable-diffusion-v1-5  (standard diffusion)
# model_b: latent-consistency/lcm-lora-sdv1-5  (LCM-LoRA, 4-step)
#
# ── Usage ──────────────────────────────────────────────────────────────────────
# Train mapper (default):
#   CUDA_VISIBLE_DEVICES=0 bash run_stitch_dmd2_sweep.sh
#
# Single-pair steer_contrast:
#   MODE=steer_contrast \
#   STEER_POS="photo of a black man" \
#   STEER_NEG="photo of a white man" \
#   STEER_APPLY="photo of a man" \
#   bash run_stitch_dmd2_sweep.sh
#
# Multi-pair steer_contrast (bash loop, NOT Hydra -m):
#   Hydra -m computes Cartesian product — N pos × N neg × N apply = N^3 jobs.
#   For N paired triplets, define STEER_PAIRS as a bash array and loop.
#   Edit the STEER_PAIRS block below, then:
#
#   MODE=steer_contrast bash run_stitch_dmd2_sweep.sh
#
#   Each pair gets its own output_dir (steer_0, steer_1, ...).
#   After all pairs finish a combined grid is saved to output_images/lcm_stitch/.
#
# GPU memory for 20k samples: float16 ~8 GB, float32 ~16 GB

set -e

# ── Model / layer config ──────────────────────────────────────────────────────
BASE_MODEL="runwayml/stable-diffusion-v1-5"
LCM_LORA="latent-consistency/lcm-lora-sdv1-5"

# SD 1.5 mid_block: 1 attention × 1 transformer block
# Activation shape: (B, 64, 1280) → flat dim = 81920
LAYER_MID="unet.mid_block.attentions.0.transformer_blocks.0.attn2"
DIM_MID=81920   # 64 * 1280

# ── Runtime / training config ─────────────────────────────────────────────────
MAPPER_PATH="${MAPPER_PATH:-./output_images/lcm_stitch/mapper.pt}"

# Mode: "train", "steer_contrast", or "steer_transfer"
# Auto-detect: if a mapper checkpoint already exists and MODE is not set,
# skip collection + training and go straight to steer_contrast.
if [[ -z "${MODE}" ]] && [[ -f "${MAPPER_PATH}" ]]; then
    MODE="steer_contrast"
    echo "[auto] Mapper found at ${MAPPER_PATH} → defaulting to MODE=steer_contrast (skip collection)"
else
    MODE="${MODE:-train}"
fi

# Collect activations into GPU memory — skips all tar files.
# Collection runs max_samples × 2 models pipeline calls (e.g. 200×2=400 at 1 step each).
COLLECT_TO_MEMORY="${COLLECT_TO_MEMORY:-true}"

# (tar-file path only) pre-load tars into GPU RAM after writing.
USE_GPU_CACHE="${USE_GPU_CACHE:-false}"
GPU_CACHE_DTYPE="${GPU_CACHE_DTYPE:-float16}"  # float16 (~8 GB) or float32 (~16 GB) at 20k samples

STEER_ALPHA="${STEER_ALPHA:-10.0}"

# Guidance scale for model_b (LCM-LoRA) generation.
# LCM-LoRA needs ~1.0; plain SD1.5 would need 7.0–7.5.
GUIDANCE_SCALE_B="${GUIDANCE_SCALE_B:-1.0}"

# ── steer_contrast pairs ──────────────────────────────────────────────────────
# For multi-pair runs: define STEER_PAIRS as a bash array.
# Each entry: "positive concept|negative concept|apply prompt"
# Apply prompt is optional — leave empty to use cfg.prompts.
#
# IMPORTANT: Do NOT use Hydra -m for multi-pair steer_contrast.
# Hydra's BasicSweeper does Cartesian product, not zip — N values per param
# gives N^3 jobs. The bash loop below gives exactly N jobs (one per pair).
#
# Example (uncomment to use):
# STEER_PAIRS=(
#   "photo of a black man|photo of a white man|photo of a person"
#   "photo of a happy woman|photo of a sad woman|photo of a person"
# )
STEER_PAIRS=("${STEER_PAIRS[@]}")

# CLI convenience: pass multiple pairs from the command line without editing the script.
# Format: STEER_PAIRS_STR="pos1|neg1|apply1;pos2|neg2|apply2"
#   separator between pairs:    semicolon  ;
#   separator within a pair:    pipe       |
# Example:
#   STEER_PAIRS_STR="photo of a black man|photo of a white man|photo of a person;photo of a happy woman|photo of a sad woman|photo of a person"
if [[ -n "${STEER_PAIRS_STR:-}" ]]; then
    IFS=';' read -ra _cli_pairs <<< "${STEER_PAIRS_STR}"
    STEER_PAIRS+=("${_cli_pairs[@]}")
fi

# Single-pair shortcut: set STEER_POS/NEG/APPLY env vars.
# Appended to STEER_PAIRS if non-empty.
STEER_POS="${STEER_POS:-}"
STEER_NEG="${STEER_NEG:-}"
STEER_APPLY="${STEER_APPLY:-}"
if [[ -n "${STEER_POS}" ]]; then
    STEER_PAIRS+=("${STEER_POS}|${STEER_NEG}|${STEER_APPLY}")
fi

# ── steer_transfer only ───────────────────────────────────────────────────────
# steer_contrast computes the direction on-the-fly; no files needed.
# steer_transfer is for injecting a pre-computed .pt vector.
STEER_VEC_PATH="${STEER_VEC_PATH:-}"
REF_ACT_PATH="${REF_ACT_PATH:-}"

# ── Shared output base dir ────────────────────────────────────────────────────
OUTPUT_BASE="${OUTPUT_BASE:-./output_images/lcm_stitch}"

# Remove any local dev diffusers checkout from PYTHONPATH to avoid import conflicts
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "LCM_LoRA" | tr '\n' ':' | sed 's/:$//')

# ── Shared Hydra overrides (used in every t2i-stitch call) ───────────────────
SHARED_OVERRIDES=(
  "--config-name=stitch/run"
  "model_key=${BASE_MODEL}"
  "model_key_b=${BASE_MODEL}"
  "lora_b.repo=${LCM_LORA}"
  "lora_b.scheduler=LCMScheduler"
  "layer_a=${LAYER_MID}"
  "layer_b=${LAYER_MID}"
  "input_dim=${DIM_MID}"
  "hidden_dim=2048"
  "output_dim=${DIM_MID}"
  "dataset_name=nirmalendu01/laion-coco-aesthetic-text-only"
  "prompt_col_a=text"
  "prompt_col_b=text"
  "capture_step_index=0"
  "num_inference_steps_a=1"
  "num_inference_steps_b=4"
  "num_inference_steps=4"
  "guidance_scale=7.5"
  "guidance_scale_b=${GUIDANCE_SCALE_B}"
  "conditional_only=true"
  "max_samples=20000"
  "save_dir=./latents_cache/lcm_mapper"
  "inject_steps=null"
  "num_steps=100000"
  "lr=1e-4"
  "collect_to_memory=${COLLECT_TO_MEMORY}"
  "use_gpu_cache=${USE_GPU_CACHE}"
  "gpu_cache_dtype=${GPU_CACHE_DTYPE}"
  "mapper_path=${MAPPER_PATH}"
  "steer_alpha=${STEER_ALPHA}"
  "wandb.project=stitch-lcm-sd15"
)

# ── steer_contrast: bash loop over explicit pairs ─────────────────────────────
if [[ "${MODE}" == "steer_contrast" ]] && [[ ${#STEER_PAIRS[@]} -gt 0 ]]; then
    STEER_JOB_DIRS=()

    for i in "${!STEER_PAIRS[@]}"; do
        IFS='|' read -r pos neg apply <<< "${STEER_PAIRS[$i]}"
        JOB_DIR="${OUTPUT_BASE}/steer_${i}"
        STEER_JOB_DIRS+=("${JOB_DIR}")

        echo ""
        echo "── steer_contrast job ${i}: pos='${pos}' neg='${neg}' apply='${apply}' ──"
        echo "   output_dir=${JOB_DIR}"

        t2i-stitch "${SHARED_OVERRIDES[@]}" \
          "mode=steer_contrast" \
          "output_dir=${JOB_DIR}" \
          "steer_pos_prompts=[${pos}]" \
          "steer_neg_prompts=[${neg}]" \
          ${apply:+"steer_apply_prompts=[${apply}]"} \
          "$@"
    done

    # After all pairs: aggregate steer_meta.json files → combined grid
    if [[ ${#STEER_JOB_DIRS[@]} -gt 1 ]]; then
        echo ""
        echo "── Aggregating ${#STEER_JOB_DIRS[@]} steer jobs into combined grid ──"
        python - "${STEER_JOB_DIRS[@]}" << 'PYEOF'
import sys, json, os
sys.path.insert(0, ".")
from t2i_interp.reporting.sweep_reports import build_steer_grid_from_jobs

job_dirs = sys.argv[1:]
job_results = []
for d in job_dirs:
    meta_path = os.path.join(d, "steer_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            job_results.append({**json.load(f), "output_dir": d})
    else:
        print(f"[aggregate] Warning: no steer_meta.json in {d}")

if job_results:
    grid_path = os.path.join(os.path.dirname(job_dirs[0]), "steer_sweep_grid.png")
    saved = build_steer_grid_from_jobs(job_results, grid_path)
    if saved:
        print(f"[aggregate] Combined grid → {saved}")
PYEOF
    fi

# ── steer_transfer / train: single job ───────────────────────────────────────
else
    t2i-stitch "${SHARED_OVERRIDES[@]}" \
      "mode=${MODE}" \
      "output_dir=${OUTPUT_BASE}" \
      ${STEER_VEC_PATH:+"steer_vec_path=${STEER_VEC_PATH}"} \
      ${REF_ACT_PATH:+"ref_act_path=${REF_ACT_PATH}"} \
      "$@"
fi
