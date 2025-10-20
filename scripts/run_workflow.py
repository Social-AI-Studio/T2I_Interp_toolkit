import os
import argparse
import importlib
from dataclasses import asdict
from typing import Optional, Callable, List, Any, Dict

import torch as th

from t2Interp.T2I import T2IModel
from t2Interp.concept_search import KSteer 
from t2Interp.mapper import MLPMapper
from reporting.config_loader import load_config, wandb_init_kwargs
from utils.runningstats import WandbUpdater
from utils.training import TrainingSpec, Training
from utils.output_manager import OutputManager
from utils.utils import preprocess_image

RACE_LABELS = {
    "East Asian": 0, "Indian": 1, "Black": 2, "White": 3,
    "Middle Eastern": 4, "Latino_Hispanic": 5, "Southeast Asian": 6,
}

def preprocess_fn(x):
    # your lambda: preprocess_image(x, 512)
    return preprocess_image(x, 512)

def race_processing_fn(x):
    # your lambda: th.tensor(race_labels[x], dtype=th.long)
    return th.tensor(RACE_LABELS[x], dtype=th.long)

def import_callable(path: Optional[str]) -> Optional[Callable]:
    """Import a callable from 'pkg.module:func_name' or return None."""
    if not path:
        return None
    if ":" not in path:
        raise ValueError(f"Expected 'module.path:callable', got '{path}'")
    mod, name = path.split(":", 1)
    fn = getattr(importlib.import_module(mod), name)
    if not callable(fn):
        raise TypeError(f"{path} is not callable")
    return fn


def build_workflow(workflow: str, model) -> Any:
    """Return the workflow object based on name."""
    wf = workflow.lower()
    if wf == "steering":
        return KSteer(model)
    # wrappers for other workflows - to be implemented
    else:
        raise ValueError(f"Unknown workflow '{workflow}'")
    
def main():
    p = argparse.ArgumentParser(description="Run T2I workflows with standard outputs.")
    # Core run config
    p.add_argument("--workflow", type=str, default="steering",
                   choices=["steering", "localisation", "stitching", "sae"])
    p.add_argument("--run_name", type=str, default="training_race_steering_mlp")
    p.add_argument("--dataset", type=str, required=True,
                   help="HF dataset repo or local path (expects train/val splits).")
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--model_repo", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--model_device", type=str, default="cuda:0")
    p.add_argument("--model_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])

    # Accessor
    p.add_argument("--accessor_path", type=str, required=True,
                   help="Python expression evaluated on model to get submodule, e.g. "
                        "'model.unet_2.down_attn_blocks[0].self_attn_out'.")

    # Mapper & loss/opt
    p.add_argument("--input_dim", type=int, default=4096*320)
    p.add_argument("--hidden_dim", type=int, default=4096)
    p.add_argument("--output_dim", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-5)

    # Training/activation params
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--denoising_step", type=int, default=0)
    p.add_argument("--training_device", type=str, default="cuda:0")
    p.add_argument("--data_device", type=str, default="cpu")
    p.add_argument("--autocast_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--refresh_batch_size", type=int, default=64)
    p.add_argument("--out_batch_size", type=int, default=16)
    p.add_argument("--d_submodule", type=int, default=4096*320)

    # Dataset columns & processing
    p.add_argument("--dataset_column", type=str, default="image")
    p.add_argument("--ground_truth_column", type=str, default="race")
    p.add_argument("--use_val", action="store_true", default=True)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--preprocess_fn", type=str, default=None,
                   help="Import path 'pkg.mod:fn' to preprocess inputs.")
    p.add_argument("--gt_processing_fn", type=str, default=None,
                   help="Import path 'pkg.mod:fn' to process ground truth.")

    # Output & caching
    p.add_argument("--use_memmap", action="store_true", default=True)
    p.add_argument("--cache_activations", action="store_true", default=True)
    p.add_argument("--log_steps", type=int, default=5)
    p.add_argument("--outputs_root", type=str, default="./runs")
    p.add_argument("--no_symlink_latest", action="store_true", default=False)

    # W&B
    p.add_argument("--wandb_config", type=str, default="reporting/config.yaml")
    p.add_argument("--wandb_run_name", type=str, default=None)

    args = p.parse_args()

    # ---- Build model ----
    dtype_map = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
    model = T2IModel(args.model_repo, device=args.model_device, dtype=dtype_map[args.model_dtype])
    model.pipeline.set_progress_bar_config(disable=True)

    # ---- Resolve accessor (evaluate expression safely) ----
    try:
        accessor = eval(args.accessor_path, {"model": model})
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate accessor_path: {args.accessor_path}") from e

    # ---- Mapper / loss / opt ----
    mapper = MLPMapper(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim)
    loss_fn = th.nn.CrossEntropyLoss()
    optimizers = [th.optim.Adam(mapper.parameters(), lr=args.lr)]

    # ---- Pre/Post processing callables ----
    preprocess_fn = import_callable(args.preprocess_fn)
    gt_processing_fn = import_callable(args.gt_processing_fn)

    # ---- kwargs passed to workflow.fit ----
    autocast_dtype = {"float16": th.float16, "bfloat16": th.bfloat16, "float32": th.float32}[args.autocast_dtype]
    workflow_kwargs: Dict[str, Any] = {
        "preprocess_fn": preprocess_fn,
        "gt_processing_fn": gt_processing_fn,
        "subset": args.subset,
        "val_split": args.val_split,
        "dataset_column": args.dataset_column,
        "ground_truth_column": args.ground_truth_column,
        "use_val": args.use_val,
        "steps": args.steps,
        "denoising_step": args.denoising_step,
        "training_device": args.training_device,
        "data_device": args.data_device,
        "autocast_dtype": autocast_dtype,
        "d_submodule": args.d_submodule,
        "log_steps": args.log_steps,
        "refresh_batch_size": args.refresh_batch_size,
        "out_batch_size": args.out_batch_size,
        "use_memmap": args.use_memmap,
        "cache_activations": args.cache_activations,
        "workflow": args.workflow,
        "run_name": args.run_name,
    }

    # ---- W&B ----
    wb_cfg = load_config(args.wandb_config)
    wb_cfg["wandb"].update({"run_name": args.wandb_run_name or args.run_name})
    init_kwargs = wandb_init_kwargs(wb_cfg)
    stats_updaters = [WandbUpdater(init_kwargs=init_kwargs)]

    # ---- Output Manager ----
    # Many OutputManager impls expect a config; if it accepts plain kwargs, this works:
    out_manager = OutputManager(
        root_dir=args.outputs_root,
        run_name=args.run_name,
        workflow=args.workflow,
        make_latest_symlink=(not args.no_symlink_latest),
    )

    # Two handy callbacks that many OutputManager variants expose:
    # - write_metadata(result, **kwargs)
    # - save_best_ckpt(result, **kwargs)
    callbacks = []
    if hasattr(out_manager, "write_metadata"):
        callbacks.append(out_manager.write_metadata)
    if hasattr(out_manager, "save_best_ckpt"):
        callbacks.append(out_manager.save_best_ckpt)

    # ---- Choose workflow and run ----
    workflow = build_workflow(args.workflow, model)

    spec = TrainingSpec(
        name=args.run_name,
        fn=workflow.fit,
        stats_updaters=stats_updaters,
        callback_fns=callbacks,
        args=(args.dataset, accessor, mapper, loss_fn, optimizers),
        kwargs=workflow_kwargs,
    )

    Training(spec).run_trainer()


if __name__ == "__main__":
    main()