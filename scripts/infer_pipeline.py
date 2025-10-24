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
from utils.runningstats import WandbUpdater, SimpleFileLogger
from utils.inference import InferenceSpec, Inference
from utils.output_manager import OutputManager
from utils.utils import preprocess_image
import json

RACE_LABELS={"East Asian":"East Asian","Indian":"Indian","Black":"Black","White":"White","Middle Eastern":"Middle Eastern","Latino_Hispanic": "Latino Hispanic","Southeast Asian":"Southeast Asian"}
race_preprocess_fn = lambda x: f"A photo of a {RACE_LABELS[x]} person."

MAPPER_REGISTRY = {
    "mlp": MLPMapper,  
}

def parse_json(s: str | None) -> Dict[str, Any]:
    return {} if not s else json.loads(s)

def resolve_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)

def build_mapper(spec: str, mapper_kwargs: Dict[str, Any]):
    cls = MAPPER_REGISTRY.get(spec) if spec in MAPPER_REGISTRY else resolve_dotted(spec)
    # Don’t pass dims from bash; infer here:
    return cls(**mapper_kwargs)

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
    
def main():
    p = argparse.ArgumentParser(description="Run T2I workflows with standard outputs.")
    p.add_argument("--workflow", type=str, default="steering",
                   choices=["steering", "localisation", "stitching", "sae"])
    p.add_argument("--infer_fn", type=str, default=None,
                   help="inference function to run.")
    p.add_argument("--run_name", type=str, default="infer_race_steering_mlp")
    
    p.add_argument("--model_repo", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--model_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])

    # Accessor
    p.add_argument("--accessor_path", type=str, required=True,
                   help="Python expression evaluated on model to get submodule, e.g. "
                        "'model.unet_2.down_attn_blocks[0].self_attn_out'.")
    
    # Mapper params
    p.add_argument("--mapper", default="mlp", help="mapper name (e.g. mlp) or dotted path")
    p.add_argument("--mapper-kwargs", default="{}", help='JSON dict for mapper dims, e.g. \'{"input_dim": 4096*320, "hidden_dim": 4096, "output_dim": 7}\'')
    p.add_argument("--mapper_ckpt", type=str, default=None, help="Path to mapper checkpoint to load.")

    # Training/activation params
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--target_idx", type=int, default=0)
    p.add_argument("--denoising_step", type=int, default=0)
    p.add_argument("--data_device", type=str, default="cpu")
    p.add_argument("--autocast_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--refresh_batch_size", type=int, default=64)
    p.add_argument("--out_batch_size", type=int, default=16)
    p.add_argument("--d_submodule", type=int, default=4096*320)

    # Dataset columns & processing
    p.add_argument("--dataset", type=str, required=True,
                   help="HF dataset repo or local path (expects train/val splits).")
    p.add_argument("--dataset_split", type=str, default="val")
    p.add_argument("--dataset_column", type=str, default="prompt")
    p.add_argument("--ground_truth_column", type=str, default="race")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--preprocess_fn", type=str, default=None,
                   help="Import path 'pkg.mod:fn' to preprocess inputs.")
    p.add_argument("--gt_processing_fn", type=str, default=None,
                   help="Import path 'pkg.mod:fn' to process ground truth.")

    # Output & caching
    p.add_argument("--use_memmap", action="store_true", default=True)
    p.add_argument("--cache_activations", action="store_true", default=True)
    p.add_argument("--outputs_root", type=str, default="./runs")
    p.add_argument("--no_symlink_latest", action="store_true", default=False)

     # Updaters
    p.add_argument("--wandb_config", type=str, default="reporting/config.yaml")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--updaters", action="append", default=['file'], help="Add a logger: wandb | file ")
    p.add_argument("--log-file", default="./logs/train.jsonl", help="Path for file logger")

    args = p.parse_args()

    # ---- Build model ----
    dtype_map = {"float16": th.float16, "bfloat16": th.bfloat16, "float32": th.float32}
    model = T2IModel(args.model_repo, device=args.device, dtype=dtype_map[args.model_dtype])
    model.pipeline.set_progress_bar_config(disable=True)

    # ---- Resolve accessor (evaluate expression safely) ----
    try:
        accessor = eval(args.accessor_path, {"model": model})
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate accessor_path: {args.accessor_path}") from e

    # ---- Pre/Post processing callables ----
    preprocess_fn = import_callable(args.preprocess_fn)
    gt_processing_fn = import_callable(args.gt_processing_fn)
    infer_fn = import_callable(args.infer_fn)

    # ---- kwargs passed to workflow.fit ----
    autocast_dtype = {"float16": th.float16, "bfloat16": th.bfloat16, "float32": th.float32}[args.autocast_dtype]
    workflow_kwargs: Dict[str, Any] = {
        "preprocess_fn": preprocess_fn,
        "gt_processing_fn": gt_processing_fn,
        "subset": args.subset,
        "dataset_split": args.dataset_split,
        "dataset_column": args.dataset_column,
        "ground_truth_column": args.ground_truth_column,
        "num_inference_steps": args.steps,
        "target_idx": args.target_idx,
        "alpha": args.alpha,
        "denoising_step": args.denoising_step,
        "data_device": args.data_device,
        "autocast_dtype": autocast_dtype,
        "d_submodule": args.d_submodule,
        "refresh_batch_size": args.refresh_batch_size,
        "out_batch_size": args.out_batch_size,
        "use_memmap": args.use_memmap,
        "cache_activations": args.cache_activations,
        "run_name": args.run_name,
        "dataset": args.dataset,
    }

    # stats_updaters=[]
    wb_cfg = load_config(args.wandb_config)
    wb_cfg["wandb"].update({"run_name": args.wandb_run_name or args.run_name})
    # if "wandb" in args.updaters:
    #     init_kwargs = wandb_init_kwargs(wb_cfg)
    #     stats_updaters.append(WandbUpdater(init_kwargs=init_kwargs))
    # if "file" in args.updaters:
    #     filelogger = SimpleFileLogger(log_path=args.log_file, args=args, kwargs=workflow_kwargs)
    #     stats_updaters.append(filelogger)

    mapper_kwargs = parse_json(args.mapper_kwargs)
    mapper = build_mapper(args.mapper, mapper_kwargs).to(device=args.device, dtype=dtype_map[args.model_dtype])
    
    # ---- Output Manager ----
    out_manager = OutputManager(
        root_dir=args.outputs_root,
        run_name=args.run_name,
        workflow=args.workflow,
        make_latest_symlink=(not args.no_symlink_latest),
    )

    # - write_metadata(result, **kwargs)
    # - save_best_ckpt(result, **kwargs)
    callbacks = []
    if hasattr(out_manager, "write_metadata"):
        callbacks.append(out_manager.write_metadata)
    if hasattr(out_manager, "write_to_wandb"):
        callbacks.append(out_manager.write_to_wandb)

    spec = InferenceSpec(
        name=args.run_name,
        inference_fn = infer_fn,
        callback_fns=callbacks,
        kwargs={**workflow_kwargs,"wb_cfg": wb_cfg, "model": model, "accessor": accessor, "mapper": mapper},
    )

    Inference(spec).run_inference()


if __name__ == "__main__":
    main()