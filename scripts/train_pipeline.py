import argparse
import ast
import importlib
import json
from collections.abc import Callable
from typing import Any

import torch as th

from reporting.config_loader import load_config, wandb_init_kwargs
from t2Interp.concept_search import KSteer
from t2Interp.mapper import MLPMapper, MLPMapperTwoHeads
from t2Interp.T2I import T2IModel
from utils.output_manager import OutputManager
from utils.runningstats import SimpleFileLogger, WandbUpdater
from utils.training import Training, TrainingSpec

# RACE_LABELS = {
#     "East Asian": 0, "Indian": 1, "Black": 2, "White": 3,
#     "Middle Eastern": 4, "Latino_Hispanic": 5, "Southeast Asian": 6,
# }

RACE_LABELS = {
    "white": 0,
    "black": 1,
    "asian": 2,
    "indian": 3,
    "latino": 4,
    "middle eastern": 5,
    None: 6,
}

GENDER_LABELS = {"male": 0, "female": 1, None: 2}

MAPPER_REGISTRY = {"mlp": MLPMapper, "twohead_mlp": MLPMapperTwoHeads}

LOSS_REGISTRY = {
    "cross_entropy": th.nn.CrossEntropyLoss,
    "mse": th.nn.MSELoss,
}

OPTIM_REGISTRY = {
    "adam": th.optim.Adam,
    "adamw": th.optim.AdamW,
    "sgd": th.optim.SGD,
}


def _norm_str(x: str | None) -> str | None:
    if not isinstance(x, str):
        return None
    s = x.strip().lower()
    return s if s else None


# def preprocess_fn(x):
#     return preprocess_image(x, 512)


def reduce_fn(x, last_indices=None):
    # must be (B,ctx,dim)
    assert x.dim() == 3, "reduce_fn expects 3D tensor"
    B, S, H = x.shape
    idx = last_indices.view(B, 1, 1).expand(B, 1, H).to(x.device)
    return x.gather(1, idx).squeeze(1)


def race_processing_fn(x):
    return th.tensor(RACE_LABELS[x], dtype=th.long)


def gender_processing_fn(x):
    return th.tensor(GENDER_LABELS[x], dtype=th.long)


def gt_processing_fn(x):
    return race_processing_fn(_norm_str(x[0])), gender_processing_fn(_norm_str(x[1]))


# def steering_classifier_trainer(model):
#     steer = KSteer(model)
#     return steer.fit


def run_ksteer_fit(**kwargs):
    model = kwargs.get("model")
    assert model is not None, "kwargs must include 'model' key"
    # ksteer_init = ksteer_init or {}
    # fit_kwargs = fit_kwargs or {}
    ksteer = KSteer(model=model)
    return ksteer.fit(**kwargs)


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


def resolve_dotted(path: str):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def parse_json(s: str | None) -> dict[str, Any]:
    return {} if not s else json.loads(s)


def build_loss(spec: str, kwargs: dict[str, Any]):
    cls = LOSS_REGISTRY.get(spec) if spec in LOSS_REGISTRY else resolve_dotted(spec)
    return cls(**kwargs)


def build_mapper(spec: str, mapper_kwargs: dict[str, Any]):
    cls = MAPPER_REGISTRY.get(spec) if spec in MAPPER_REGISTRY else resolve_dotted(spec)
    # Don’t pass dims from bash; infer here:
    return cls(**mapper_kwargs)


def build_optimizers(specs: list[str], kwargs_list: list[dict[str, Any]], params):
    optims = []
    for spec, kw in zip(specs, kwargs_list):
        cls = OPTIM_REGISTRY.get(spec) if spec in OPTIM_REGISTRY else resolve_dotted(spec)
        optims.append(cls(params, **kw))
    return optims


# def build_workflow(workflow: str, model) -> Any:
#     """Return the workflow object based on name."""
#     wf = workflow.lower()
#     if wf == "steering":
#         return KSteer(model)
#     # wrappers for other workflows - to be implemented
#     else:
#         raise ValueError(f"Unknown workflow '{workflow}'")


def main():
    p = argparse.ArgumentParser(description="Run T2I workflows with standard outputs.")
    # Core run config
    p.add_argument(
        "--workflow",
        type=str,
        default="steering",
        choices=["steering", "localisation", "stitching", "sae"],
    )
    p.add_argument("--training_fn", type=str, default=None, help="training function to run.")
    p.add_argument("--run_name", type=str, default="training_race_steering_mlp")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HF dataset repo or local path (expects train/val splits).",
    )
    p.add_argument("--val_split", type=str, default="val")
    p.add_argument("--model_repo", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--model_device", type=str, default="cuda:0")
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

    # Mapper & loss/opt
    p.add_argument("--mapper", default="mlp", help="mapper name (e.g. mlp) or dotted path")
    p.add_argument(
        "--mapper-kwargs",
        default="{}",
        help='JSON dict for mapper dims, e.g. \'{"input_dim": 4096*320, "hidden_dim": 4096, "output_dim": 7}\'',
    )
    p.add_argument("--loss", required=True, help="loss name (e.g. cross_entropy) or dotted path")
    p.add_argument("--loss-kwargs", default="{}", help="JSON dict for loss ctor")
    p.add_argument(
        "--optim",
        action="append",
        required=True,
        help="optimizer name or dotted path; repeat for multiple",
    )
    p.add_argument(
        "--optim-kwargs",
        action="append",
        default=[],
        help="JSON dict for each optimizer; repeat to match --optim",
    )
    # p.add_argument("--input_dim", type=int, default=4096*320)
    # p.add_argument("--hidden_dim", type=int, default=4096)
    # p.add_argument("--output_dim", type=int, default=7)
    # p.add_argument("--lr", type=float, default=1e-5)

    # Training/activation params
    p.add_argument("--train_steps", type=int, default=200)
    p.add_argument("--denoiser_steps", type=lambda s: json.loads(s), default=None)
    p.add_argument("--training_device", type=str, default="cuda:0")
    p.add_argument("--data_device", type=str, default="cpu")
    p.add_argument(
        "--autocast_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    p.add_argument("--refresh_batch_size", type=int, default=64)
    p.add_argument("--out_batch_size", type=int, default=16)
    p.add_argument("--d_submodule", type=int, default=4096 * 320)

    # Dataset columns & processing
    p.add_argument("--dataset_column", type=str, default="image")
    p.add_argument("--ground_truth_column", type=str, default="race")
    p.add_argument("--use_val", action="store_true", default=True)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument(
        "--preprocess_fn",
        type=str,
        default=None,
        help="Import path 'pkg.mod:fn' to preprocess inputs.",
    )
    p.add_argument(
        "--reduce_fn",
        type=str,
        default=None,
        help="Import path 'pkg.mod:fn' to reduce captured activations.",
    )
    p.add_argument(
        "--gt_processing_fn",
        type=str,
        default=None,
        help="Import path 'pkg.mod:fn' to process ground truth.",
    )

    # Output & caching
    p.add_argument("--use_memmap", action="store_true", default=True)
    p.add_argument("--cache_activations", action="store_true", default=True)
    p.add_argument("--log_steps", type=int, default=5)
    p.add_argument("--outputs_root", type=str, default="./runs")
    p.add_argument("--no_symlink_latest", action="store_true", default=False)

    # Updaters
    p.add_argument("--wandb_config", type=str, default="reporting/config.yaml")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--updaters", action="append", default=["file"], help="Add a logger: wandb | file "
    )
    p.add_argument("--log-file", default="logs/train.jsonl", help="Path for file logger")

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
    # mapper = MLPMapper(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim)
    # loss_fn = th.nn.CrossEntropyLoss()
    # optimizers = [th.optim.Adam(mapper.parameters(), lr=args.lr)]
    mapper_kwargs = parse_json(args.mapper_kwargs)
    loss_kwargs = parse_json(args.loss_kwargs)
    ground_truth_column = ast.literal_eval(args.ground_truth_column)
    optim_kwargs_list = [parse_json(s) for s in (args.optim_kwargs or [])]

    while len(optim_kwargs_list) < len(args.optim):
        optim_kwargs_list.append({})

    mapper = build_mapper(args.mapper, mapper_kwargs).to(
        args.model_device, dtype=getattr(th, dtype_map[args.model_dtype])
    )
    loss_fn = build_loss(args.loss, loss_kwargs)
    optimizers = build_optimizers(args.optim, optim_kwargs_list, mapper.parameters())

    # ---- Pre/Post processing callables ----
    preprocess_fn = import_callable(args.preprocess_fn)
    reduce_fn = import_callable(args.reduce_fn)
    gt_processing_fn = import_callable(args.gt_processing_fn)
    training_fn = import_callable(args.training_fn)

    # ---- kwargs passed to workflow.fit ----
    autocast_dtype = {"float16": th.float16, "bfloat16": th.bfloat16, "float32": th.float32}[
        args.autocast_dtype
    ]
    workflow_kwargs: dict[str, Any] = {
        # "preprocess_fn": preprocess_fn,
        # "gt_processing_fn": gt_processing_fn,
        # "subset": args.subset,
        # "val_split": args.val_split,
        # "dataset_column": args.dataset_column,
        # "ground_truth_column": ground_truth_column,
        # "use_val": args.use_val,
        "train_steps": args.train_steps,
        # "denoiser_steps": args.denoiser_steps,
        "training_device": args.training_device,
        # "data_device": args.data_device,
        "autocast_dtype": autocast_dtype,
        # "d_submodule": args.d_submodule,
        "log_steps": args.log_steps,
        # "refresh_batch_size": args.refresh_batch_size,
        # "out_batch_size": args.out_batch_size,
        # "use_memmap": args.use_memmap,
        # "cache_activations": args.cache_activations,
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
            "val_split": args.val_split,
            "subset": args.subset,
            "data_device": args.data_device,
            "denoiser_steps": args.denoiser_steps,
            "ground_truth_column": ground_truth_column,
            "dataset_column": args.dataset_column,
            "reduce_fn": reduce_fn,
        },
    }

    stats_updaters = []
    if "wandb" in args.updaters:
        wb_cfg = load_config(args.wandb_config)
        wb_cfg["wandb"].update({"run_name": args.wandb_run_name or args.run_name})
        init_kwargs = wandb_init_kwargs(wb_cfg)
        stats_updaters.append(WandbUpdater(init_kwargs=init_kwargs))
    if "file" in args.updaters:
        filelogger = SimpleFileLogger(log_path=args.log_file, kwargs=workflow_kwargs)
        stats_updaters.append(filelogger)

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
    if hasattr(out_manager, "save_best_ckpt"):
        callbacks.append(out_manager.save_best_ckpt)

    spec = TrainingSpec(
        name=args.run_name,
        fn=training_fn,
        stats_updaters=stats_updaters,
        callback_fns=callbacks,
        # args=(args.dataset, accessor, mapper, loss_fn, optimizers, model),
        args=[],
        kwargs={
            **workflow_kwargs,
            "model": model,
            "dataset": args.dataset,
            "accessor": accessor,
            "mapper": mapper,
            "loss_fn": loss_fn,
            "optimizers": optimizers,
        },
    )

    Training(spec).run_trainer()


if __name__ == "__main__":
    main()
