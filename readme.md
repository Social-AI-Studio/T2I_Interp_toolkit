Here’s a **ready-to-drop `README.md`** that matches your `workflow.sh` launcher (no sweep script). It explains how to run, what the script does, and a few pro tips (unique run names, logs, etc.).

---

# T2I-Interp Toolkit

This repo provides **text-to-image interpretation** workflows built on Diffusers UNet hooks. You launch a workflow from the CLI, get live status via updaters (tqdm/W&B/file logs), and a clean on-disk **run folder** with metadata, reports, and artifacts.

---

## Workflows

* **`steering`** — learns a mapper/classifier from UNet activations to a target label and applies it as a “steer”.
* (Optional) `localisation`, `stitching`, `sae` — follow the same generator-style fit API and plug into the same Trainer/logging/output layers.

Each workflow exposes:

```python
def fit(...) -> Generator[TrainUpdate, None, Output]
```

It **yields** `TrainUpdate` (step logs) and finally **returns** an `Output` (e.g., best checkpoint + metadata).

---

## Install

```bash
# from repo root
pip install -e .

# optional: auth for datasets and W&B
huggingface-cli login
wandb login
```

> If you prefer not to install, invoke with `PYTHONPATH=/path/to/repo` and run `python -m scripts.run_workflow ...`.

---

## Datasets & Accessors

* **Dataset**: HF repo or local dataset with `train`/`val` splits.
* **Accessor**: dotted/indexed path to a submodule on `model`, e.g.
  `model.unet_2.down_attn_blocks[0].self_attn_out`

---

## Pre/Post-processing functions

Pass callables by **import path** (`module:func`). In your setup you export them from `scripts/run_workflow.py`:

```
--preprocess_fn scripts.run_workflow:preprocess_fn
--gt_processing_fn scripts.run_workflow:race_processing_fn
```

(If you move them elsewhere, update the import paths accordingly.)

---

## Run with `train_pipeline.sh` (recommended)

Use the provided `scripts/train_pipeline.sh` to loop over multiple **accessors**:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASET='nirmalendu01/fairface-trainval-race-balanced-200'

declare -a ACCESSORS=(
  "model.unet_2.down_attn_blocks[0].self_attn_out"
  "model.unet_2.down_attn_blocks[0].cross_attn_out"
)

run_name='steer_mlp_train'
hidden_dim=4096
steps=1000
lr=1e-5
autocast_dtype='bfloat16'

for ACCESSOR in "${ACCESSORS[@]}"; do
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
```

### Launch (with logging)

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_pipeline.sh > run.log 2>&1 &
```

* `nohup … &` runs it in the background.
* `> run.log 2>&1` captures **stdout and stderr** (Loguru’s console sink is stderr).
* Per-run **file logs** still go to: `runs/<run_name>/logs/run.log`.

### Make each run name unique (optional but helpful)

Right now every accessor uses the same `--run_name`. To distinguish them, append a short slug:

```bash
slug=$(echo "$ACCESSOR" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')
python -m scripts.train_pipeline \
  --run_name "${run_name}-${slug}" \
  ...
```

That gives you separate output folders (and W&B runs) per accessor.

---

## Direct CLI (no script)

You can also call the runner directly once:

```bash
python -m scripts.train_pipeline \
  --training_fn KSteer.fit \
  --run_name steer_fairface_sd14 \
  --dataset nirmalendu01/fairface-trainval-race-balanced-200 \
  --accessor_path 'model.unet_2.down_attn_blocks[0].self_attn_out' \
  --input_dim $((4096*320)) --hidden_dim 4096 --output_dim 7 \
  --steps 200 --refresh_batch_size 64 --out_batch_size 16 \
  --training_device cuda:0 --data_device cpu --autocast_dtype bfloat16 \
  --preprocess_fn scripts.train_pipeline:preprocess_fn \
  --gt_processing_fn scripts.train_pipeline:race_processing_fn \
  --wandb_config reporting/config.yaml
```

---

## Status updaters

The Trainer calls each updater’s `.log(update)` per step and `.done()` once at the end.

* **`SimpleUpdater`** — `tqdm` progress bar (console).
* **`WandbUpdater`** — pushes metrics to Weights & Biases.
* **`SimpleFileLogger`** — writes to `runs/<run_name>/logs/run.log` (Loguru).

Bound fields (e.g., `step`, `loss`, `val_loss`) are printed by the file logger because the file sink uses a format including `{extra}`.

---

## Output structure & metadata

Each run is recorded under `runs/<run_name>/`:

```
runs/<run_name>/
  run_metadata.json      # canonical metadata (dataset, accessor, hyperparams, etc.)
  viz/                   # JS/HTML visualizers (optional)
  report/                # static PDFs/PNGs/HTML (optional)
  artifacts/             # checkpoints, memmaps, arrays…
```

The **OutputManager** creates this layout and writes `run_metadata.json`.
Callbacks (e.g., `save_best_ckpt`) can persist your best model to `artifacts/`.

---

## Tips & troubleshooting

* **Imports**: install the repo (`pip install -e .`) or use `python -m` from the repo root so `from t2Interp...` works.
* **Accessor typos**: double-check `--accessor_path` strings—errors show at startup.
* **No file logs?** Ensure `SimpleFileLogger` is added once and that `.done()` is called once after training (not per step). Check `runs/<run_name>/logs/run.log`.
* **Precision**: `--autocast_dtype bfloat16` is generally stable; you can keep model weights in fp32.
* **Throughput**: increase `--out_batch_size`; CPU slices are pinned before H2D for fast transfers.

---

## Minimal end-to-end sanity check

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m scripts.train_pipeline \
  --training_fn KSteer.fit \
  --run_name test_run \
  --dataset nirmalendu01/fairface-trainval-race-balanced-200 \
  --accessor_path 'model.unet_2.down_attn_blocks[0].self_attn_out' \
  --input_dim $((4096*320)) --hidden_dim 1024 --output_dim 7 \
  --steps 20 --refresh_batch_size 32 --out_batch_size 16 \
  --training_device cuda:0 --data_device cpu --autocast_dtype bfloat16 \
  --preprocess_fn scripts.train_pipeline:preprocess_fn \
  --gt_processing_fn scripts.train_pipeline:race_processing_fn
```

You should see:

* W&B run created (if configured),
* console progress (tqdm),
* per-run file logs at `runs/test_run/logs/run.log`,
* `run_metadata.json` and (if applicable) a saved best checkpoint under `artifacts/`.

---
