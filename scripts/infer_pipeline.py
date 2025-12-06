import argparse
import importlib
import json
from collections.abc import Callable, Generator
from itertools import tee
from typing import Any

import torch as th

from dictionary_learning.utils import hf_dataset_to_generator
from reporting.config_loader import load_config, wandb_init_kwargs
from t2Interp.accessors import ModuleAccessor
from t2Interp.concept_search import CAA, KSteer
from t2Interp.mapper import MLPMapper
from t2Interp.T2I import T2IModel
from utils.inference import Inference, InferenceSpec
from utils.output import Output
from utils.output_manager import OutputManager
from utils.runningstats import SimpleFileLogger, Update, WandbUpdater
from utils.text_image_buffer import _build_buffer
from utils.text_img_util import OutputAlterHook, replace_policy, run_with_hook
from utils.utils import ActivationConfig, BatchIterator, gen_images_from_prompts

RACE_LABELS = {
    "East Asian": "East Asian",
    "Indian": "Indian",
    "Black": "Black",
    "White": "White",
    "Middle Eastern": "Middle Eastern",
    "Latino_Hispanic": "Latino Hispanic",
    "Southeast Asian": "Southeast Asian",
}
race_preprocess_fn = lambda x: f"A photo of a {RACE_LABELS[x]} person."

MAPPER_REGISTRY = {
    "mlp": MLPMapper,
}


def parse_json(s: str | None) -> dict[str, Any]:
    return {} if not s else json.loads(s)


def resolve_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def build_mapper(spec: str, mapper_kwargs: dict[str, Any]):
    cls = MAPPER_REGISTRY.get(spec) if spec in MAPPER_REGISTRY else resolve_dotted(spec)
    # Don’t pass dims from bash; infer here:
    return cls(**mapper_kwargs)


def run_steering(
    model: T2IModel, dataset, accessor: ModuleAccessor, mapper: th.nn.Module, **kwargs
) -> Generator[Update, None, Output]:
    # def cache_path(dataset,split,subset):
    #     base = Path("data") / dataset / accessor.attr_name / split
    #     return str(base / str(subset) if subset is not None else base)

    # def log(msg:str):
    #     update = Update(info=msg)
    #     yield update

    # prompts, buffer = tee(BatchIterator(hf_dataset_to_generator(dataset,**kwargs),batch_size=kwargs.get("out_batch_size",1)))
    assert hasattr(kwargs, "data_loader_kwargs"), "data_loader_kwargs must be provided in kwargs"
    d_sub = kwargs.data_loader_kwargs.get("d_submodule", None)
    # target_idx = kwargs.pop("target_idx", None)
    # assert target_idx is not None, "target_idx must be provided in kwargs"

    dataset_split = kwargs.data_loader_kwargs.get("dataset_split", "val")
    use_memmap = kwargs.data_loader_kwargs.get("use_memmap", False)

    cfg = ActivationConfig(
        autocast_dtype=kwargs.get("autocast_dtype", th.float32),
        data_loader_kwargs=kwargs.get("data_loader_kwargs", {}),
    )
    prompts, gen = tee(hf_dataset_to_generator(dataset, **kwargs.data_loader_kwargs))
    buffer = _build_buffer(gen, model, accessor, dataset, dataset_split, cfg)

    # if use_memmap and os.path.exists(cache_path(dataset,dataset_split,kwargs.get('subset',None))):
    #     log(f"Using existing memmap at {cache_path(dataset,dataset_split,kwargs.get('subset',None))} for steering activations")
    #     buffer = ShardedActivationMemmapDataset(cache_path(dataset,dataset_split,kwargs.get('subset',None)),**kwargs)
    # elif use_memmap:
    #     buffer = t2IActivationBuffer(hf_dataset_to_generator(dataset,**kwargs), model, accessor,d_submodule=d_sub, **kwargs)
    #     log(f"Creating memmap at {cache_path(dataset,dataset_split,kwargs.get('subset',None))} for steering activations")
    #     buffer = convert_buffer_to_memap(buffer,memmap_dir=cache_path(dataset,dataset_split,kwargs.get('subset',None)), **kwargs)
    # else:
    #     buffer = t2IActivationBuffer(hf_dataset_to_generator(dataset,**kwargs), model, accessor,d_submodule=d_sub, **kwargs)

    prompts = BatchIterator(prompts, batch_size=kwargs.data_loader_kwargs.get("out_batch_size", 1))

    # use_cache = kwargs.data_loader_kwargs.get("cache_activations", False)
    # if use_cache:
    #     buffer = CachedActivationIterator(buffer, **kwargs)

    if kwargs.get("steer_method", "KSteer") == "KSteer":
        steer = KSteer(model)
    elif kwargs.get("steer_method", "KSteer") == "CAA":
        steer = CAA(model)
    generate_kwargs = {}
    if "guidance_scale" in kwargs:
        generate_kwargs["guidance_scale"] = kwargs["guidance_scale"]
    imgs = []
    baseline_imgs = []

    p_iter = iter(prompts)  # don't recreate iter() each loop
    b_iter = iter(buffer)

    i = 0
    while True:
        try:
            ps = next(p_iter)
            activations = next(b_iter)
        except StopIteration:
            break  # either side exhausted → we're done

        yield Update(info=f"Steering on batch {i}")

        steered_activation = steer.steer(activations, mapper=mapper, **kwargs)

        hook_obj = OutputAlterHook(
            replace_policy(steered_activation),
            step_index=kwargs.get("denoiser_step", 0),
        )

        # If these are constant per run, assert once outside loop
        seed = kwargs.get("seed", None)
        num_inference_steps = kwargs.get("num_inference_steps", None)
        assert seed is not None, "seed must be provided in kwargs"
        assert num_inference_steps is not None, "num_inference_steps must be provided in kwargs"

        output = run_with_hook(
            model,
            ps,
            accessor.module,
            hook_obj,
            accessor.io_type,
            **{**{"seed": seed, "num_inference_steps": num_inference_steps}, **generate_kwargs},
        )

        baselines = gen_images_from_prompts(model, ps, **{**kwargs, **generate_kwargs})

        # extend vs append: extend flattens per-batch image lists
        imgs.extend(output.images)
        baseline_imgs.extend(baselines)

        i += 1
    return Output(preds=imgs, baselines=baseline_imgs)


def import_callable(path: str | None) -> Callable | None:
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
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Workflow
    p.add_argument(
        "--workflow",
        type=str,
        default="steering",
        choices=["steering", "localisation", "stitching", "sae"],
    )
    p.add_argument("--infer_fn", type=str, default=None, help="inference function to run.")
    p.add_argument("--run_name", type=str, default="infer_race_steering_mlp")

    p.add_argument("--model_repo", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--model_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"]
    )

    # Accessor
    p.add_argument(
        "--accessor_path",
        type=str,
        required=True,
        help="Python expression evaluated on model to get submodule, e.g. "
        "'model.unet_2.down_attn_blocks[0].self_attn_out'.",
    )

    # Mapper params
    p.add_argument("--mapper", default="mlp", help="mapper name (e.g. mlp) or dotted path")
    p.add_argument(
        "--mapper-kwargs",
        default="{}",
        help='JSON dict for mapper dims, e.g. \'{"input_dim": 4096*320, "hidden_dim": 4096, "output_dim": 7}\'',
    )
    p.add_argument(
        "--mapper_ckpt", type=str, default=None, help="Path to mapper checkpoint to load."
    )

    # Training/activation params
    p.add_argument("--steer_steps", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--target_idx", type=int, default=0)
    p.add_argument("--denoiser_step", type=int, default=10)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--data_device", type=str, default="cpu")
    p.add_argument(
        "--autocast_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    p.add_argument("--refresh_batch_size", type=int, default=64)
    p.add_argument("--out_batch_size", type=int, default=16)
    p.add_argument("--d_submodule", type=int, default=4096 * 320)

    # Dataset columns & processing
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HF dataset repo or local path (expects train/val splits).",
    )
    p.add_argument("--dataset_split", type=str, default="val")
    p.add_argument("--dataset_column", type=str, default="prompt")
    p.add_argument("--ground_truth_column", type=str, default="race")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument(
        "--preprocess_fn",
        type=str,
        default=None,
        help="Import path 'pkg.mod:fn' to preprocess inputs.",
    )
    p.add_argument(
        "--gt_processing_fn",
        type=str,
        default=None,
        help="Import path 'pkg.mod:fn' to process ground truth.",
    )

    p.add_argument("--steer_method", type=str, default="KSteer", choices=["KSteer", "CAA"])
    # Output & caching
    p.add_argument("--use_memmap", action="store_true", default=False)
    p.add_argument("--cache_activations", action="store_true", default=False)
    p.add_argument("--outputs_root", type=str, default="./runs")
    p.add_argument("--no_symlink_latest", action="store_true", default=False)

    # Updaters
    p.add_argument("--wandb_config", type=str, default="reporting/config.yaml")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--updaters", action="append", default=["file"], help="Add a logger: wandb | file "
    )
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
    autocast_dtype = {"float16": th.float16, "bfloat16": th.bfloat16, "float32": th.float32}[
        args.autocast_dtype
    ]

    workflow_kwargs: dict[str, Any] = {
        "num_inference_steps": args.inference_steps,
        "train_steps": args.train_steps,
        "training_device": args.training_device,
        "autocast_dtype": autocast_dtype,
        "log_steps": args.log_steps,
        "alpha": args.alpha,
        "steer_steps": args.steer_steps,
        "steer_method": args.steer_method,
        "seed": args.seed,
        "workflow": args.workflow,
        "run_name": args.run_name,
        "data_loader_kwargs": {
            "refresh_batch_size": args.refresh_batch_size,
            "out_batch_size": args.out_batch_size,
            "use_memmap": args.use_memmap,
            "cache_activations": args.cache_activations,
            "d_submodule": args.d_submodule,
            "preprocess_fn": preprocess_fn,
            "gt_processing_fn": gt_processing_fn,
            "use_val": args.use_val,
            "dataset_split": args.dataset_split,
            "subset": args.subset,
            "data_device": args.data_device,
            "denoiser_step": args.denoiser_step,
            "dataset_column": args.dataset_column,
            "ground_truth_column": args.ground_truth_column,
        },
    }

    # stats_updaters=[]
    wb_cfg = load_config(args.wandb_config)
    wb_cfg["wandb"].update({"run_name": args.wandb_run_name or args.run_name})
    init_kwargs = wandb_init_kwargs(wb_cfg)
    # if "wandb" in args.updaters:
    #
    #     stats_updaters.append(WandbUpdater(init_kwargs=init_kwargs))
    # if "file" in args.updaters:
    #     filelogger = SimpleFileLogger(log_path=args.log_file, args=args, kwargs=workflow_kwargs)
    #     stats_updaters.append(filelogger)

    stats_updaters = []
    if "wandb" in args.updaters:
        stats_updaters.append(WandbUpdater(init_kwargs=init_kwargs))
    if "file" in args.updaters:
        filelogger = SimpleFileLogger(log_path=args.log_file, kwargs=workflow_kwargs)
        stats_updaters.append(filelogger)

    mapper_kwargs = parse_json(args.mapper_kwargs)
    mapper = build_mapper(args.mapper, mapper_kwargs).to(
        device=args.device, dtype=dtype_map[args.model_dtype]
    )

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
        inference_fn=infer_fn,
        stats_updaters=stats_updaters,
        callback_fns=callbacks,
        kwargs={
            **workflow_kwargs,
            "model": model,
            "accessor": accessor,
            "mapper": mapper,
            "wandb_init_kwargs": init_kwargs,
        },
    )

    Inference(spec).run_inference()


if __name__ == "__main__":
    main()
